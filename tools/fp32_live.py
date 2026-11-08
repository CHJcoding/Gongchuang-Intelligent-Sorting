# -*- coding: utf-8 -*-
"""fp32(ONNX/ORT) 作为分数天花板，对照 v2 int8 在现场帧上的差距。"""
import sys, glob, os
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "/root/vision_ws")
from vision_sorter.yolov8_bpu import letterbox_image, decode_outputs, letterbox_forward

CLASSES = ["hollow_cube", "tetrahedron", "pentagonal_prism", "triangular_prism",
           "cone", "cylinder", "hemisphere", "cube"]
sess = ort.InferenceSession("/root/vision_ws/fp32_best.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

def infer(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    lb, ratio, dw, dh = letterbox_image(bgr, 640, 640)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    out = sess.run(None, {iname: x})
    boxes, scores, cls = decode_outputs(out, 640, 640, len(CLASSES), strides=(8, 16, 32),
                                        conf_thres=0.02, iou_thres=0.45)
    if len(scores) == 0:
        return [], boxes
    boxes = letterbox_forward(boxes, ratio, dw, dh)
    return list(zip(scores.tolist(), cls.tolist(), boxes.tolist())), boxes

paths = sorted(glob.glob("/tmp/burst_*.jpg")) + sorted(glob.glob("/tmp/f[0-3].jpg")) + sorted(glob.glob("/tmp/live_*.jpg"))
print("frame            n  top1            top2            bbox")
for p in paths:
    res = infer(p)
    if res is None:
        print("%-14s read-fail" % os.path.basename(p)); continue
    arr, _ = res
    arr = sorted(arr, key=lambda z: -z[0])[:2]
    s = "  ".join("%s %.2f" % (CLASSES[c], sc) for sc, c, _ in arr) or "-"
    bb = ("(%.0f,%.0f)-(%.0f,%.0f)" % tuple(arr[0][2][:2] + arr[0][2][2:])) if arr else ""
    print("%-14s %2d %-32s %s" % (os.path.basename(p), len(arr), s, bb))
