# -*- coding: utf-8 -*-
"""别人的 YOLO11(6类, 输出 [1,10,8400]) 在板端 ORT fp32 跑我们的现场帧/标注集。"""
import sys, glob, os, time
import numpy as np, cv2, onnxruntime as ort

NAMES = ["green_tri_pyramid","green_cube","pink_hex_prism",
         "pink_tri_prism","blue_cylinder","blue_penta_prism"]
S = ort.InferenceSession("/root/vision_ws/other/best.onnx", providers=["CPUExecutionProvider"])
IN = S.get_inputs()[0].name
OUT = S.get_outputs()[0].name

def letterbox(img, size=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(size/h, size/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size-nh)//2, (size-nw)//2
    out = np.full((size,size,3), color, np.uint8)
    out[top:top+nh, left:left+nw] = res
    return out, r, left, top

def nms(boxes, scores, iou=0.45):
    idx = np.argsort(-scores); keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1: break
        b = boxes[i]; rest = boxes[idx[1:]]
        xx1 = np.maximum(b[0], rest[:,0]); yy1 = np.maximum(b[1], rest[:,1])
        xx2 = np.minimum(b[2], rest[:,2]); yy2 = np.minimum(b[3], rest[:,3])
        inter = np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None)
        a1 = (b[2]-b[0])*(b[3]-b[1]); a2 = (rest[:,2]-rest[:,0])*(rest[:,3]-rest[:,1])
        iou_v = inter/np.maximum(a1+a2-inter,1e-9)
        idx = idx[1:][iou_v <= iou]
    return keep

def infer(path, conf=0.25):
    bgr = cv2.imread(path)
    lb, r, dw, dh = letterbox(bgr)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = np.ascontiguousarray(x.transpose(2,0,1)[None])
    t0 = time.time(); out = S.run([OUT], {IN: x})[0]; dt = (time.time()-t0)*1000
    p = out[0]                       # [10,8400]
    cls_raw = p[4:]                  # [6,8400]
    cls_id = cls_raw.argmax(0)
    cls_sc = cls_raw.max(0)          # YOLO11 是单标签训练，直接取最大 logit 当分数(已是 sigmoid 后?)
    m = cls_sc >= conf
    if not m.any(): return [], dt
    b = p[:4, m].T                   # cx,cy,w,h (640 空间)
    sc = cls_sc[m]; ci = cls_id[m]
    boxes = np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], 1)
    keep = nms(boxes, sc)
    boxes = boxes[keep]; sc = sc[keep]; ci = ci[keep]
    boxes[:, [0,2]] = (boxes[:, [0,2]] - dw)/r
    boxes[:, [1,3]] = (boxes[:, [1,3]] - dh)/r
    return [(float(s), int(c), bb) for s, c, bb in zip(sc, ci, boxes)], dt

print("score-space probe: cls max range = %.3f .. %.3f" % (0, 0))
paths = sorted(glob.glob("/tmp/green_0*.jpg"))
print("=== 现场全绿帧 ===")
for p in paths:
    ds, dt = infer(p)
    ds = sorted(ds, key=lambda z: -z[0])[:5]
    print(" %-14s %4.0fms n=%d  %s" % (os.path.basename(p), dt, len(ds),
          "  ".join("%s %.2f(%.0f,%.0f)" % (NAMES[c], s, b[0], b[1]) for s, c, b in ds)))
print("=== 我们的标注集（每类 5 张） ===")
root = "/root/vision_ws/dataset/20261031_1126/raw"
for d in sorted(os.listdir(root)):
    fs = sorted(glob.glob(os.path.join(root, d, "*.jpg")))[:5]
    for p in fs:
        ds, dt = infer(p)
        ds = sorted(ds, key=lambda z: -z[0])[:3]
        print(" %-6s %-20s %4.0fms %s" % (d, os.path.basename(p), dt,
              "  ".join("%s %.2f" % (NAMES[c], s) for s, c, b in ds) or "-"))
