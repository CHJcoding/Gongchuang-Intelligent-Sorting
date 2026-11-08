# -*- coding: utf-8 -*-
"""板端探针：判断「0 检出」是画面里没东西，还是预处理/颜色顺序和训练不一致。

做法：抓一帧 → 用 4 种预处理组合各跑一次 → 打印每种组合的最高分类分数。
    1) nv12(UV)  + letterbox   ← 当前实现
    2) nv21(VU)  + letterbox   ← UV 顺序反了
    3) nv12(UV)  + stretch     ← 官方 TROS 是直接拉伸
    4) nv21(VU)  + stretch
哪组分数明显高，就是哪组对了。同时打印画面颜色统计，判断有没有物料在画面里。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _c in (os.path.join(ROOT, "rdk", "vision_sorter"), os.path.join(ROOT, "vision_sorter")):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)

from hobot_dnn import pyeasy_dnn as dnn  # noqa: E402
import yolov8_bpu as Y  # noqa: E402

BIN = "/root/vision_ws/gc_best_1029.bin"
HSV_RANGES = {
    "red": [((0, 40, 40), (12, 255, 255)), ((160, 40, 40), (180, 255, 255))],
    "green": [((35, 40, 40), (95, 255, 255))],
    "blue": [((95, 40, 40), (135, 255, 255))],
    "pink": [((140, 35, 90), (170, 255, 255))],
    "yellow": [((20, 40, 40), (34, 255, 255))],
}


def nv12_of(bgr, swap_uv=False):
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    ys, cs = h * w, h * w // 4
    y = yuv[:ys].reshape(h, w)
    u = yuv[ys:ys + cs].reshape(h // 2, w // 2)
    v = yuv[ys + cs:ys + 2 * cs].reshape(h // 2, w // 2)
    out = np.empty((h * 3 // 2, w), np.uint8)
    out[:h] = y
    uv = out[h:].reshape(h // 2, w // 2, 2)
    uv[..., 0] = v if swap_uv else u
    uv[..., 1] = u if swap_uv else v
    return out.reshape(1, h * 3 // 2, w, 1)


def color_stats(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = bgr.shape[0] * bgr.shape[1]
    out = {}
    for name, ranges in HSV_RANGES.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        out[name] = (m > 0).sum() / total
    return out


def main():
    cap = cv2.VideoCapture("/dev/cam0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    assert ok, "读帧失败"
    cv2.imwrite("/tmp/probe_frame.jpg", frame)

    b, g, r = [float(frame[..., i].mean()) for i in range(3)]
    print("画面 BGR 均值: B=%.1f G=%.1f R=%.1f  亮度=%.1f" % (b, g, r, (b + g + r) / 3))
    cs = color_stats(frame)
    print("整幅颜色覆盖率: " + "  ".join("%s=%.3f" % (k, v) for k, v in sorted(cs.items(), key=lambda kv: -kv[1])))
    print("（某种颜色 >0.5%% 才说明画面里有物料）")
    print()

    m = dnn.load(BIN)[0]
    print("模型输入:", m.inputs[0].properties.shape)

    variants = [
        ("nv12 letterbox", False, "letterbox"),
        ("nv21 letterbox", True, "letterbox"),
        ("nv12 stretch  ", False, "stretch"),
        ("nv21 stretch  ", True, "stretch"),
    ]
    for tag, swap, mode in variants:
        if mode == "letterbox":
            canvas, ratio, dw, dh = Y.letterbox_image(frame, 640, 640)
        else:
            canvas = cv2.resize(frame, (640, 640))
        inp = nv12_of(canvas, swap_uv=swap)
        outs = m.forward(inp)
        arrs = [np.asarray(o.buffer).reshape(o.buffer.shape) for o in outs]
        # 只统计 cls（每个 stride 一个），转 sigmoid 后取全局最大
        best = -1e9
        best_cls = -1
        per = []
        for k in (0, 2, 4):
            logits = np.asarray(arrs[k], np.float32)[0]
            smax = float(logits.max())
            best = max(best, smax)
            # 每个网格的类别最大 → 全局最大
            gmax = float(logits.max(axis=-1).max())
            best_cls = max(best_cls, gmax)
            per.append(round(gmax, 2))
        print("%s  各尺度 cls 最大 logits=%s  → sigmoid=%.3f"
              % (tag, per, 1.0 / (1.0 + np.exp(-best_cls))))


if __name__ == "__main__":
    main()