# -*- coding: utf-8 -*-
import glob, sys
import cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C

FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))

def broad_parts(hsv):
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    m = ((h>=30)&(h<=110)&(s>=60)&(v>=40)).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    boxes=[]
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 400:
            keep[lab==i]=255
            boxes.append((st[i,0], st[i,1], st[i,2], st[i,3], st[i,cv2.CC_STAT_AREA]))
    return keep, boxes

img = cv2.imread(FRAMES[0]); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
part, boxes = broad_parts(hsv)
core = cv2.erode(part, np.ones((11,11),np.uint8))
sel = core>0
hh, ss, vv = hsv[:,:,0][sel].astype(np.int16), hsv[:,:,1][sel].astype(np.int16), hsv[:,:,2][sel].astype(np.int16)
print("core px", int(sel.sum()))
print(" H<78: %.1f%%  H>102: %.1f%%  S<60: %.1f%%  V<50: %.1f%%"
      % (100*np.mean(hh<78), 100*np.mean(hh>102), 100*np.mean(ss<60), 100*np.mean(vv<50)))
print(" 联合不满足(78~102,S>=60,V>=50): %.1f%%" % (100*np.mean(~((hh>=78)&(hh<=102)&(ss>=60)&(vv>=50)))))
hist = np.histogram(hh, bins=np.arange(30,116,5))[0]
for b, c in zip(np.arange(30,111,5), hist):
    if c: print("   H %3d-%3d : %5d px (%.1f%%)" % (b, b+4, c, 100.0*c/len(hh)))
print("boxes:", boxes)
