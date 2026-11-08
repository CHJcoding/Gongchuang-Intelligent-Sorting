# -*- coding: utf-8 -*-
"""现场实测：绿色物料 HSV 分割精准化（用实拍帧统计，不靠猜）。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C

FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))
print("帧数 %d" % len(FRAMES))

# 宽窗只用来"找出绿料在哪"，真正阈值由统计决定
BROAD = (30, 100, 60, 40)   # H lo, H hi, S min, V min


def comps(hsv, broad):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= broad[0]) & (h <= broad[1]) & (s >= broad[2]) & (v >= broad[3]))
    m = (m.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 400:
            continue
        out.append((i, area, stats[i], cent[i]))
    return lab, out


allH, allS, allV, areas = [], [], [], []
per_frame = []
for f in FRAMES[:6]:
    img = cv2.imread(f)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab, cs = comps(hsv, BROAD)
    line = []
    for (i, area, st, ce) in cs:
        sel = (lab == i)
        hh = hsv[:, :, 0][sel].astype(np.int16)
        ss = hsv[:, :, 1][sel].astype(np.int16)
        vv = hsv[:, :, 2][sel].astype(np.int16)
        allH += [int(np.percentile(hh, 1)), int(np.percentile(hh, 99))]
        allS.append(int(np.percentile(ss, 1)))
        allV.append(int(np.percentile(vv, 1)))
        areas.append(area)
        line.append("area=%d H%d~%d(p50=%d) S%d~(p50=%d) V%d~(p50=%d)"
                    % (area, int(np.percentile(hh, 1)), int(np.percentile(hh, 99)),
                       int(np.median(hh)), int(np.percentile(ss, 1)), int(np.median(ss)),
                       int(np.percentile(vv, 1)), int(np.median(vv))))
    per_frame.append((os.path.basename(f), len(cs), line))
    print("%-9s 绿块 %d 个" % (os.path.basename(f), len(cs)))
    for x in line:
        print("        " + x)

print("")
print("跨帧/跨物料汇总：H p1 min=%d  H p99 max=%d  S p1 min=%d  V p1 min=%d  面积 %d~%d"
      % (min(allH[0::2]) if allH else -1, max(allH[1::2]) if allH else -1,
         min(allS) if allS else -1, min(allV) if allV else -1,
         min(areas) if areas else -1, max(areas) if areas else -1))
