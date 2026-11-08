import sys, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import yolov8_bpu as Y
NAMES = ["BQ","LK","SLZ","SMT","WLZ","YZ","YZHU","cube"]
img = cv2.imread("/tmp/val_imgs/023.jpg")
h, w = img.shape[:2]
gt = []
for ln in open("/tmp/val_lbls/023.txt"):
    p = ln.split()
    if len(p) >= 5:
        c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        gt.append((c, (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
det = Y.Yolov8Bpu("/root/vision_ws/gc_best_1029.bin", NAMES, conf_thres=0.25, verbose=False)
res = det.detect(img)
print("检出 %d 个（GT %d 个）" % (len(res), len(gt)))
def iou(a,b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1]); x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0,x2-x1), max(0,y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0
for d in res:
    best, bi = 0, -1
    for i,g in enumerate(gt):
        v = iou((d.x1,d.y1,d.x2,d.y2), (g[1],g[2],g[3],g[4]))
        if v > best: best, bi = v, i
    gname = NAMES[gt[bi][0]] if bi >= 0 else "-"
    print("  %-5s 分数=%.3f 框=(%4.0f,%3.0f)-(%4.0f,%3.0f) 尺寸=%.0fx%.0f | 最匹配 %s IoU=%.2f"
          % (NAMES[d.cls_id] if d.cls_id < len(NAMES) else d.cls_id, d.score,
             d.x1, d.y1, d.x2, d.y2, d.x2-d.x1, d.y2-d.y1, gname, best))