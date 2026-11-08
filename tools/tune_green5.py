# -*- coding: utf-8 -*-
"""每颗绿料的色相归属 + 含青绿瓣的候选区间评估。"""
import glob, sys
import cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
import color_classifier as CC

FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))

def parts(hsv):
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    m = ((h>=30)&(h<=125)&(s>=60)&(v>=40)).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m); boxes=[]
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 400:
            keep[lab==i]=255
            boxes.append((i, st[i,0], st[i,1], st[i,2], st[i,3], st[i,cv2.CC_STAT_AREA]))
    return keep, boxes

img = cv2.imread(FRAMES[0]); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
part, boxes = parts(hsv)
print("逐颗（框 x,y,w,h）：H中位 / 落在 H>100 的像素占比 / 面积")
for (i, x, y, w, h, a) in boxes:
    sel = (part>0) & (np.repeat(np.arange(hsv.shape[0])[:,None], hsv.shape[1], 1) >= y) & \
          (np.repeat(np.arange(hsv.shape[0])[:,None], hsv.shape[1], 1) < y+h) & \
          (np.repeat(np.arange(hsv.shape[1])[None,:], hsv.shape[0], 0) >= x) & \
          (np.repeat(np.arange(hsv.shape[1])[None,:], hsv.shape[0], 0) < x+w)
    hh = hsv[:,:,0][sel]
    if len(hh)==0: continue
    print("  (%3d,%3d,%2dx%2d) area=%4d  H中位=%3d  H>100占比=%4.0f%%  Hp90=%3d"
          % (x,y,w,h,a, int(np.median(hh)), 100*np.mean(hh>100), int(np.percentile(hh,90))))

CANDS = [
    ("H1 80~115 S60 V45", (80,60,45), (115,255,255)),
    ("H2 80~115 S65 V50", (80,65,50), (115,255,255)),
    ("H3 80~112 S60 V45", (80,60,45), (112,255,255)),
    ("H4 78~118 S60 V45", (78,60,45), (118,255,255)),
]
print("")
print("%-20s %-10s %-12s %s" % ("区间", "核心召回", "背景误检px", "最大误检块"))
print("-"*70)
agg={}
for f in FRAMES[:8]:
    im = cv2.imread(f); hsv2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    p2, _ = parts(hsv2)
    core = cv2.erode(p2, np.ones((11,11),np.uint8))
    cn = max(int(cv2.countNonZero(core)),1)
    for name, lo, hi in CANDS:
        m = cv2.inRange(hsv2, np.array(lo), np.array(hi))
        rec = cv2.countNonZero(cv2.bitwise_and(m, core))/cn
        fp = cv2.morphologyEx(cv2.bitwise_and(m, cv2.bitwise_not(p2)), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        n, lab, st, ce = cv2.connectedComponentsWithStats(fp, 8)
        big = max([st[i,cv2.CC_STAT_AREA] for i in range(1,n)] or [0])
        agg.setdefault(name,[]).append((rec, cv2.countNonZero(fp), big))
for name,_,_ in CANDS:
    r=agg[name]
    print("%-20s %-10s %-12s %d px" % (name, "%.1f%%"%(100*np.mean([x[0] for x in r])),
          "%.0f"%np.mean([x[1] for x in r]), np.mean([x[2] for x in r])))

print("")
print("=== 每颗真实 ROI 判色（旧 / H2） ===")
old = C.COLOR_RANGES_HSV["green"]
for tag, rng in (("旧 36~85", old), ("H2 80~115", CANDS[1][1:])):
    C.COLOR_RANGES_HSV["green"] = rng
    out=[]
    for (i,x,y,w,h,a) in boxes:
        roi = img[max(0,y):y+h, max(0,x):x+w]
        out.append(CC.classify_color(roi))
    print("  %-10s %s" % (tag, ["%s %.2f"%(a,b) for a,b in out]))
C.COLOR_RANGES_HSV["green"] = old
