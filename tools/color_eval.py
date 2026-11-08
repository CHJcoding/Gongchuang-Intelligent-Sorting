# -*- coding: utf-8 -*-
"""绿区间候选扫描：真实 YOLO 框 + 真实 classify_color，块内投票率与空盘误判。"""
import sys, glob, os, json
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "/root/vision_ws")
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
import color_classifier as CC
from vision_sorter.yolov8_bpu import letterbox_image, decode_outputs, letterbox_forward

CLASSES = ["hollow_cube","tetrahedron","pentagonal_prism","triangular_prism",
           "cone","cylinder","hemisphere","cube"]
sess = ort.InferenceSession("/root/vision_ws/fp32_best.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
CACHE = {}

def dets(path, conf=0.20):
    if path in CACHE: return CACHE[path]
    bgr = cv2.imread(path)
    lb, ratio, dw, dh = letterbox_image(bgr, 640, 640)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = np.ascontiguousarray(x.transpose(2,0,1)[None])
    out = sess.run(None, {iname: x})
    boxes, scores, cls = decode_outputs(out, 640, 640, len(CLASSES), strides=(8,16,32),
                                        conf_thres=conf, iou_thres=0.45)
    if len(scores):
        boxes = letterbox_forward(boxes, ratio, dw, dh)
    r = [(float(s), int(c), b) for s,c,b in zip(scores.tolist(), cls.tolist(), boxes)]
    CACHE[path] = (bgr, r)
    return CACHE[path]

def roi_of(bgr, box):
    h, w = bgr.shape[:2]
    x1,y1,x2,y2 = [float(v) for v in box[:4]]
    mx = (x2-x1)*C.GRASP_MASK_MARGIN; my = (y2-y1)*C.GRASP_MASK_MARGIN
    x1 = max(0, int(x1-mx)); y1 = max(0, int(y1-my))
    x2 = min(w-1, int(x2+mx)); y2 = min(h-1, int(y2+my))
    return bgr[y1:y2, x1:x2]

def sweep(name, lo, hi, s_min=None, v_min=None, disabled=None):
    C.COLOR_RANGES_HSV["green"] = (tuple(lo), tuple(hi))
    if disabled is not None: C.COLOR_DISABLED = set(disabled)
    print("### %s green=%s..%s CHROMATIC_S_MIN=%s" % (name, lo, hi, C.CHROMATIC_S_MIN))
    for p in sorted(glob.glob("/tmp/green_0*.jpg")):
        bgr, ds = dets(p)
        ds = sorted(ds, key=lambda z: -z[0])[:2]
        if not ds: print("  %s 无检测" % os.path.basename(p)); continue
        for sc, c, b in ds:
            roi = roi_of(bgr, b)
            col, cf = CC.classify_color(roi)
            print("  %s %-16s %.2f -> %-6s conf=%.2f  %s" % (
                os.path.basename(p), CLASSES[c], sc, str(col), cf, CC.color_diag(roi)))
    bad = 0; tot = 0
    for p in sorted(glob.glob("/root/vision_ws/dataset/20261031_1126/raw/negative/*.jpg"))[:25]:
        bgr, ds = dets(p)
        if not ds: continue
        tot += 1
        for sc, c, b in ds:
            roi = roi_of(bgr, b)
            col, cf = CC.classify_color(roi)
            if col == "green":
                bad += 1
                print("  空盘误判 %s %-14s %.2f %s" % (os.path.basename(p), CLASSES[c], sc, CC.color_diag(roi)))
    print("  空盘帧=%d 被框=%d 判成绿=%d" % (tot, sum(1 for _ in [0]), bad))

CANDS = [
    ("cur",      (74,80,50),  (106,255,255)),
    ("A_66_92",  (66,60,60),  (92,255,255)),
    ("B_62_96",  (62,50,50),  (96,255,255)),
    ("C_68_90",  (68,70,60),  (90,255,255)),
]
for name, lo, hi in CANDS:
    sweep(name, lo, hi)
