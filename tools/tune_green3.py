# -*- coding: utf-8 -*-
"""绿色阈值定标 v3：核心召回 + 背景误检定位。"""
import glob
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C

FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))
CANDS = [
    ("B  78~102 S60 V50", (78, 60, 50), (102, 255, 255)),
    ("D  80~100 S70 V55", (80, 70, 55), (100, 255, 255)),
    ("E  76~104 S65 V50", (76, 65, 50), (104, 255, 255)),
    ("F  78~102 S75 V60", (78, 75, 60), (102, 255, 255)),
    ("G  75~105 S60 V50", (75, 60, 50), (105, 255, 255)),
]


def broad_parts(hsv):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= 30) & (h <= 110) & (s >= 60) & (v >= 40)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 400:
            keep[lab == i] = 255
    return keep


print("%-20s %-12s %-14s %s" % ("区间", "核心召回", "背景误检px", "误检块(最大边/面积)"))
print("-" * 80)
agg = {}
for f in FRAMES[:8]:
    img = cv2.imread(f)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    part = broad_parts(hsv)
    core = cv2.erode(part, np.ones((11, 11), np.uint8))
    core_n = max(int(cv2.countNonZero(core)), 1)
    for name, lo, hi in CANDS:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        rec = cv2.countNonZero(cv2.bitwise_and(m, core)) / core_n
        fp = cv2.bitwise_and(m, cv2.bitwise_not(part))
        fp = cv2.morphologyEx(fp, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, ce = cv2.connectedComponentsWithStats(fp, 8)
        big = max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)] or [0])
        agg.setdefault(name, []).append((rec, cv2.countNonZero(fp), big))

for name, _, _ in CANDS:
    r = agg[name]
    print("%-20s %-12s %-14s %s" % (name,
          "%.1f%%" % (100 * np.mean([x[0] for x in r])),
          "%.0f" % np.mean([x[1] for x in r]),
          "%d px" % np.mean([x[2] for x in r])))

# 背景误检到底在哪：用 B 区间看最大误检块的 HSV
img = cv2.imread(FRAMES[0])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
part = broad_parts(hsv)
lo, hi = np.array([78, 60, 50]), np.array([102, 255, 255])
m = cv2.inRange(hsv, lo, hi)
fp = cv2.morphologyEx(cv2.bitwise_and(m, cv2.bitwise_not(part)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, ce = cv2.connectedComponentsWithStats(fp, 8)
order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:5]
print("")
print("B 区间最大背景误检块：")
for i in order:
    a = st[i, cv2.CC_STAT_AREA]
    if a < 20:
        continue
    sel = (lab == i)
    hh = hsv[:, :, 0][sel]; ss = hsv[:, :, 1][sel]; vv = hsv[:, :, 2][sel]
    print("  area=%4d bbox=(%d,%d,%dx%d) HSV中位(%d,%d,%d)"
          % (a, st[i, 0], st[i, 1], st[i, 2], st[i, 3],
             int(np.median(hh)), int(np.median(ss)), int(np.median(vv))))

# 绿料真身 HSV 全量分布（核心区）
core = cv2.erode(part, np.ones((11, 11), np.uint8))
sel = core > 0
hh, ss, vv = hsv[:, :, 0][sel], hsv[:, :, 1][sel], hsv[:, :, 2][sel]
print("")
print("绿料核心区 %d px：H p0.5=%d p50=%d p99.5=%d | S p1=%d p50=%d p99=%d | V p1=%d p50=%d p99=%d"
      % (int(sel.sum()), np.percentile(hh, 0.5), np.median(hh), np.percentile(hh, 99.5),
         np.percentile(ss, 1), np.median(ss), np.percentile(ss, 99),
         np.percentile(vv, 1), np.median(vv), np.percentile(vv, 99)))
