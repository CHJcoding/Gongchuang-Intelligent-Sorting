# -*- coding: utf-8 -*-
"""只认红/绿之后：其它颜色（含冷白背景、黑夹爪）一律判色失败不下发。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "vision_sorter"),
          "/root/vision_ws", "/root/vision_ws/vision_sorter"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import cv2
import config as C
import color_classifier as CC

FAIL = []
OK = [0]


def check(name, cond, extra=""):
    if cond:
        OK[0] += 1
    else:
        FAIL.append("%s %s" % (name, extra))


def patch(b, g, r, size=(60, 60), noise=6):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:, :] = (b, g, r)
    if noise:
        n = np.random.randint(-noise, noise + 1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def hsv_patch(h, s, v, size=(60, 60)):
    hsv = np.zeros((size[1], size[0], 3), np.uint8)
    hsv[:, :] = (h, s, v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


np.random.seed(0)

# 允许的两种
keep = {
    "红(30,30,200)": ("red", patch(30, 30, 200)),
    "绿(40,190,40)": ("green", patch(40, 190, 40)),
}
# 必须屏蔽的
drop = {
    "黑物料(20,20,20)": patch(20, 20, 20),
    "黑夹爪(45,45,45)": patch(45, 45, 45),
    "中灰(120,120,120)": patch(120, 120, 120),
    "白物料(235,235,235)": patch(235, 235, 235),
    "橙(30,120,220)": patch(30, 120, 220),
    "蓝(200,60,40)": patch(200, 60, 40),
    "黄(30,220,220)": patch(30, 220, 220),
    "粉 H160": hsv_patch(160, 70, 230),
    "冷白背景 H100 S45 V220": hsv_patch(100, 45, 220),
    "偏青绿 H88": hsv_patch(88, 120, 180),
}

print("COLOR_ALLOWED  = %s" % (getattr(C, "COLOR_ALLOWED", None),))
print("COLOR_DISABLED = %s" % (C.COLOR_DISABLED,))
print("%-26s %-10s %s" % ("样本", "判定", "conf"))
print("-" * 50)
for label, (want, img) in keep.items():
    name, conf = CC.classify_color(img.copy())
    print("%-26s %-10s %.3f" % (label, str(name), conf))
    check(label + " 判为 " + want, name == want, "实际 %r" % ((name, conf),))
for label, img in drop.items():
    name, conf = CC.classify_color(img.copy())
    print("%-26s %-10s %.3f" % (label, str(name), conf))
    check(label + " 判色失败", name is None, "实际 %r" % ((name, conf),))

# 混框：冷白背景 + 中间一颗真料（旧逻辑这里会被判成 blue）
for want, hsv_block in (("green", (60, 120, 180)), ("red", (177, 150, 200))):
    roi = hsv_patch(100, 45, 220)
    blk = hsv_patch(*hsv_block, size=(24, 24))
    roi[18:42, 18:42] = blk
    name, conf = CC.classify_color(roi)
    print("%-26s %-10s %.3f" % ("冷白背景 + " + want + " 料", str(name), conf))
    check("冷白背景里的 %s 仍能认出来" % want, name == want, "实际 %r" % ((name, conf),))

# 配置自洽
check("COLOR_ALLOWED 就是 red/green",
      set(getattr(C, "COLOR_ALLOWED", ())) == {"red", "green"},
      str(getattr(C, "COLOR_ALLOWED", None)))
check("屏蔽集合 = 其余 8 色",
      set(C.COLOR_DISABLED) == {"brown", "orange", "yellow", "blue", "purple", "pink", "black", "white"},
      str(C.COLOR_DISABLED))
check("被屏蔽的颜色仍在 COLOR_RANGES_HSV（可一键恢复）",
      all(k in C.COLOR_RANGES_HSV for k in C.COLOR_DISABLED))
check("COLOR_IDS 不变（blue=5 white=9 black=8）",
      (C.COLOR_IDS.get("blue"), C.COLOR_IDS.get("white"), C.COLOR_IDS.get("black")) == (5, 9, 8),
      str(C.COLOR_IDS))

print("")
print("通过 %d 项，失败 %d 项" % (OK[0], len(FAIL)))
for f in FAIL:
    print("  [失败] " + f)
print("全部通过" if not FAIL else "有失败项")
sys.exit(0 if not FAIL else 1)
