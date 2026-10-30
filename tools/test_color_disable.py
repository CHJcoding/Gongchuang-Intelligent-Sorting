# -*- coding: utf-8 -*-
"""黑色被屏蔽后：黑/灰目标判色失败（不下发），真颜色不受影响。"""
import os, sys
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
    if cond: OK[0] += 1
    else: FAIL.append("%s %s" % (name, extra))

def patch(b, g, r, size=(60, 60), noise=6):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:, :] = (b, g, r)
    if noise:
        n = np.random.randint(-noise, noise + 1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img

np.random.seed(0)
cases = {
    "黑物料(20,20,20)":      patch(20, 20, 20),
    "黑夹爪(45,45,45)":      patch(45, 45, 45),
    "暗灰(70,70,70)":        patch(70, 70, 70),
    "中灰(120,120,120)":     patch(120, 120, 120),
    "白物料(235,235,235)":   patch(235, 235, 235),
    "红(30,30,200)":         patch(30, 30, 200),
    "橙(30,120,220)":        patch(30, 120, 220),
    "绿(40,190,40)":         patch(40, 190, 40),
    "蓝(200,60,40)":         patch(200, 60, 40),
    "黄(30,220,220)":        patch(30, 220, 220),
}

print("当前 COLOR_DISABLED = %s" % (C.COLOR_DISABLED,))
print("%-22s %-10s %-6s | %s" % ("样本", "判定", "conf", "若不禁黑色会判成"))
print("-" * 78)
res = {}
for label, img in cases.items():
    name, conf = CC.classify_color(img.copy())
    res[label] = (name, conf)
    old = getattr(C, "COLOR_DISABLED", frozenset())
    C.COLOR_DISABLED = frozenset()
    was = CC.classify_color(img.copy())
    C.COLOR_DISABLED = old
    print("%-22s %-10s %-6.3f | %s conf=%.3f" % (label, str(name), conf, str(was[0]), was[1]))

# 黑色系必须判失败（不下发）
for label in ("黑物料(20,20,20)", "黑夹爪(45,45,45)", "暗灰(70,70,70)", "中灰(120,120,120)"):
    check(label + " 判色失败", res[label][0] is None, "实际 %r" % (res[label],))
# 真颜色不能被误伤
check("白仍在", res["白物料(235,235,235)"][0] == "white", repr(res["白物料(235,235,235)"]))
check("红仍在", res["红(30,30,200)"][0] == "red", repr(res["红(30,30,200)"]))
check("绿仍在", res["绿(40,190,40)"][0] == "green", repr(res["绿(40,190,40)"]))
check("蓝仍在", res["蓝(200,60,40)"][0] == "blue", repr(res["蓝(200,60,40)"]))
check("黄仍在", res["黄(30,220,220)"][0] == "yellow", repr(res["黄(30,220,220)"]))
# 编号不受影响
check("black 的 color_id 仍存在(8)", C.COLOR_IDS.get("black") == 8, str(C.COLOR_IDS.get("black")))
check("black 仍在 COLOR_RANGES_HSV（可一键恢复）", "black" in C.COLOR_RANGES_HSV)

print("")
print("通过 %d 项，失败 %d 项" % (OK[0], len(FAIL)))
for f in FAIL: print("  [失败] " + f)
print("全部通过" if not FAIL else "有失败项")
sys.exit(0 if not FAIL else 1)
