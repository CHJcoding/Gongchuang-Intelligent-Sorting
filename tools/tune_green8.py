# -*- coding: utf-8 -*-
"""12 帧复核 → 通过才写 config；并生成标注图。"""
import glob, sys, shutil, time
import cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
import color_classifier as CC

NEW = ((74, 80, 50), (106, 255, 255))
BOXES = [(389,154,61,61), (176,176,87,76), (309,178,53,54),
         (388,240,66,63), (315,245,90,102), (228,277,74,74)]
DARK = [(325,393,95,87), (104,399,78,81)]
FRAMES = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))

old = C.COLOR_RANGES_HSV["green"]
C.COLOR_RANGES_HSV["green"] = NEW
bad = 0
conf_min = 1.0
for f in FRAMES:
    img = cv2.imread(f)
    for (x,y,w,h) in BOXES:
        nm, cf = CC.classify_color(img[max(0,y):y+h, max(0,x):x+w])
        if nm != "green":
            bad += 1
        else:
            conf_min = min(conf_min, cf)
    for (x,y,w,h) in DARK:
        nm, cf = CC.classify_color(img[max(0,y):y+h, max(0,x):x+w])
        if nm == "green":
            bad += 1
print("12 帧 × 6 颗绿料：失败 %d 次，绿料最低 conf=%.2f" % (bad, conf_min))
print("12 帧 × 2 颗暗青：误判绿 %d 次（要求 0）" % bad)

if bad:
    print("复核不通过，未写入 config")
    C.COLOR_RANGES_HSV["green"] = old
    sys.exit(1)

# 背景 40 片复核
img = cv2.imread(FRAMES[0]); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
allbox = np.zeros(img.shape[:2], np.uint8)
for (x,y,w,h) in BOXES + DARK:
    cv2.rectangle(allbox, (x,y), (x+w,y+h), 255, -1)
allbox = cv2.dilate(allbox, np.ones((25,25),np.uint8))
rng = np.random.RandomState(0); fp = 0; n = 0
while n < 40:
    x = rng.randint(0, 640-48); y = rng.randint(0, 480-48)
    if allbox[y:y+48, x:x+48].max() > 0: continue
    n += 1
    if CC.classify_color(img[y:y+48, x:x+48])[0] == "green": fp += 1
print("背景 40 片误判绿：%d 片" % fp)

# 写入 config
cp = "/root/vision_ws/vision_sorter/config.py"
s = open(cp, encoding="utf-8").read()
OLD_LINE = '    "green":    [(36,  40,  40),  (85,  255, 255)],'
assert s.count(OLD_LINE) == 1, s.count(OLD_LINE)
NEW_LINE = ('    # 2026-10-31 现场实测（640x480，板端相机默认曝光/饱和度）：\n'
            '    # 6 颗绿料实测 H 中位 88~95（p1≈83, p99≈110）、S 108~165、V 120~145。\n'
            '    # 旧区间 36~85 一像素都命中不了（召回 3%），绿料其实一直落在 blue(86~125) 里\n'
            '    # —— 这就是"只放红绿却识别出蓝"的真正原因。\n'
            '    # 上界卡 106 是为了把台面下沿那颗暗青件（H≈108, V 55~75）挡在外面。\n'
            '    "green":    [(74,  80,  50),  (106, 255, 255)],')
shutil.copy2(cp, cp + ".bak." + time.strftime("%m%d_%H%M"))
open(cp, "w", encoding="utf-8").write(s.replace(OLD_LINE, NEW_LINE))
print("config 已写入 green=%s" % (NEW,))

# 标注图（最终区间着色）
vis = img.copy()
m = cv2.inRange(hsv, np.array(NEW[0]), np.array(NEW[1]))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
vis[m > 0] = (0, 255, 0)
for k,(x,y,w,h) in enumerate(BOXES, 1):
    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,255), 1)
    cv2.putText(vis, str(k), (x+2, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
for k,(x,y,w,h) in enumerate(DARK, 1):
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 1)
    cv2.putText(vis, "D%d" % k, (x+2, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
cv2.imwrite("/root/vision_ws/hsv_tune/green_final.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("标注图: /root/vision_ws/hsv_tune/green_final.jpg")
