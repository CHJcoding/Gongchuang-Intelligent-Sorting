# -*- coding: utf-8 -*-
"""绿色阈值定标：候选区间 vs 实拍绿料(召回) / 背景(误检)。"""
import glob
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
import color_classifier as CC

FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))
OLD = C.COLOR_RANGES_HSV["green"]
CANDS = {
    "A 76~104 S55 V45": ((76, 55, 45), (104, 255, 255)),
    "B 78~102 S60 V50": ((78, 60, 50), (102, 255, 255)),
    "C 74~106 S50 V40": ((74, 50, 40), (106, 255, 255)),
}


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


def stat(mask, part):
    part_n = int(cv2.countNonZero(part))
    inr = int(cv2.countNonZero(cv2.bitwise_and(mask, part)))
    bg = int(cv2.countNonZero(cv2.bitwise_and(mask, cv2.bitwise_not(part))))
    return inr, part_n, bg


print("旧区间 green = %s" % (OLD,))
print("%-18s %-16s %-14s %s" % ("区间", "绿料召回", "背景误检(px/帧)", "整图占比"))
print("-" * 78)
rows = {}
for f in FRAMES[:8]:
    img = cv2.imread(f)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    part = broad_parts(hsv)
    total = img.shape[0] * img.shape[1]
    for name, (lo, hi) in [("旧 36~85 S40 V40", (tuple(map(int, OLD[0])), tuple(map(int, OLD[1]))))] + list(CANDS.items()):
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        inr, part_n, bg = stat(m, part)
        rows.setdefault(name, []).append((inr / max(part_n, 1), bg, cv2.countNonZero(m) / total))

for name in ["旧 36~85 S40 V40"] + list(CANDS):
    r = rows[name]
    rec = 100 * np.mean([x[0] for x in r])
    bgm = np.mean([x[1] for x in r])
    tot = 100 * np.mean([x[2] for x in r])
    print("%-18s %-16s %-14s %.1f%%" % (name, "%.1f%%" % rec, "%.0f" % bgm, tot))

print("")
print("=== 用真实 ROI 过一遍 classify_color（旧 vs 新） ===")
hsv0 = cv2.cvtColor(cv2.imread(FRAMES[0]), cv2.COLOR_BGR2HSV)
part = broad_parts(hsv0)
n, lab, st, ce = cv2.connectedComponentsWithStats(part, 8)
boxes = [(st[i, 0], st[i, 1], st[i, 2], st[i, 3]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 400]
img = cv2.imread(FRAMES[0])
old_green = C.COLOR_RANGES_HSV["green"]
for tag, rng in (("旧", old_green), ("新", CANDS["B 78~102 S60 V50"])):
    C.COLOR_RANGES_HSV["green"] = rng
    res = []
    for (x, y, w, h) in boxes:
        roi = img[max(0, y):y + h, max(0, x):x + w]
        res.append(CC.classify_color(roi))
    print("  %s: %s" % (tag, ["%s %.2f" % (a, b) for a, b in res]))
    # 背景取样：每个框右移 120px（无料区域附近）不该判成绿
    bgs = []
    for (x, y, w, h) in boxes:
        x2 = min(640 - w, x + 160)
        roi = img[max(0, y):y + h, x2:x2 + w]
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        bgs.append(CC.classify_color(roi)[0])
    print("    背景采样判色: %s" % (bgs,))
C.COLOR_RANGES_HSV["green"] = old_green
