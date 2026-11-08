# -*- coding: utf-8 -*-
import glob, sys
import cv2, numpy as np
FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))
img = cv2.imread(FRAMES[0]); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
m = ((h>=30)&(h<=125)&(s>=60)&(v>=40)).astype(np.uint8)*255
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
print("框(x,y,w,h) area  H中位 S中位 V中位  长宽比  判定")
for i in range(1, n):
    a = st[i, cv2.CC_STAT_AREA]
    if a < 400: continue
    sel = (lab == i)
    hh = hsv[:,:,0][sel].astype(int); ss = hsv[:,:,1][sel].astype(int); vv = hsv[:,:,2][sel].astype(int)
    w_, h_ = st[i,2], st[i,3]
    ar = round(max(w_,h_)/max(min(w_,h_),1), 2)
    print("(%3d,%3d,%3dx%3d) %5d  H%3d S%3d V%3d  ar=%.2f  %s"
          % (st[i,0], st[i,1], w_, h_, a, int(np.median(hh)), int(np.median(ss)),
             int(np.median(vv)), ar,
             "细长/背景" if ar > 2.0 else "块状"))
print("")
print("两瓣对比（只看块状 ar<=2）：")
for lo, hi, tag in ((80,100,"绿瓣H80-100"), (100,120,"青瓣H100-120")):
    hs, ss2, vv2 = [], [], []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 400: continue
        w_, h_ = st[i,2], st[i,3]
        if max(w_,h_)/max(min(w_,h_),1) > 2.0: continue
        sel = (lab == i)
        hh = hsv[:,:,0][sel].astype(int)
        if not (lo <= np.median(hh) < hi): continue
        ss2.append(np.median(hsv[:,:,1][sel]))
        vv2.append(np.median(hsv[:,:,2][sel]))
    if ss2:
        print("  %s: %d 颗  S中位=%.0f~%.0f(均%.0f)  V中位=%.0f~%.0f(均%.0f)"
              % (tag, len(ss2), min(ss2), max(ss2), np.mean(ss2), min(vv2), max(vv2), np.mean(vv2)))
