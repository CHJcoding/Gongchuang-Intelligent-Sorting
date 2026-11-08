# -*- coding: utf-8 -*-
"""定绿：候选区间在真实帧上的召回/误检 + 生成标注图给人看。"""
import glob, sys
import cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
import color_classifier as CC

FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))
CANDS = [
    ("P1 74~106 S80 V50", (74, 80, 50), (106, 255, 255)),
    ("P2 76~104 S85 V55", (76, 85, 55), (104, 255, 255)),
    ("P3 78~100 S90 V60", (78, 90, 60), (100, 255, 255)),
    ("P4 80~104 S90 V70", (80, 90, 70), (104, 255, 255)),
]
old = C.COLOR_RANGES_HSV["green"]

def comps(hsv, amax=125):
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    m = ((h>=30)&(h<=amax)&(s>=60)&(v>=40)).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m); items=[]
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 400:
            keep[lab==i]=255
            items.append(i)
    return keep, lab, st, items

img = cv2.imread(FRAMES[0]); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
part, lab, st, items = comps(hsv)
parts = []
for i in items:
    x,y,w,h,a = st[i,0],st[i,1],st[i,2],st[i,3],st[i,cv2.CC_STAT_AREA]
    hh = hsv[:,:,0][lab==i]; vv = hsv[:,:,2][lab==i]
    kind = "亮绿" if np.median(hh) < 100 else "暗青"
    ar = max(w,h)/max(min(w,h),1)
    parts.append((x,y,w,h,a,int(np.median(hh)),int(np.median(vv)),kind,ar))

# 背景取样：远离所有物料框的随机小片
mask_any = cv2.dilate(part, np.ones((25,25),np.uint8))
rng = np.random.RandomState(0)
bgs = []
for _ in range(200):
    x = rng.randint(0, 640-48); y = rng.randint(0, 480-48)
    if mask_any[y:y+48, x:x+48].max() > 0: continue
    bgs.append((x,y))
bgs = bgs[:40]
print("背景取样 %d 片（避开物料 25px）" % len(bgs))
print("%-20s %-26s %-16s %s" % ("区间", "亮绿6颗判色", "暗青2颗判色", "背景误判绿"))
print("-"*92)
for name, lo, hi in [("旧 36~85 S40 V40", tuple(map(int,old[0])), tuple(map(int,old[1])))] + CANDS:
    C.COLOR_RANGES_HSV["green"] = (lo, hi)
    bright, dark = [], []
    for (x,y,w,h,a,hmed,vmed,kind,ar) in parts:
        roi = img[max(0,y):y+h, max(0,x):x+w]
        nm, cf = CC.classify_color(roi)
        (bright if kind=="亮绿" else dark).append("%s" % ("绿" if nm=="green" else "-"))
    fpc = 0
    for (x,y) in bgs:
        if CC.classify_color(img[y:y+48, x:x+48])[0] == "green": fpc += 1
    print("%-20s %-26s %-16s %d/%d" % (name, "".join(bright), "".join(dark), fpc, len(bgs)))
C.COLOR_RANGES_HSV["green"] = old

# 标注图：每颗画出框+序号+色相/亮度
vis = img.copy()
for k,(x,y,w,h,a,hmed,vmed,kind,ar) in enumerate(parts):
    col = (0,255,0) if kind=="亮绿" else (255,128,0)
    cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)
    cv2.putText(vis, "%d H%d V%d" % (k+1, hmed, vmed), (x+2, max(14,y-4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
cv2.imwrite("/root/vision_ws/hsv_tune/labeled.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("")
for k,(x,y,w,h,a,hmed,vmed,kind,ar) in enumerate(parts, 1):
    print("  %2d (%3d,%3d) %3dx%3d area=%4d H=%3d V=%3d %s%s"
          % (k,x,y,w,h,a,hmed,vmed,kind, " 细长" if ar>2 else ""))
