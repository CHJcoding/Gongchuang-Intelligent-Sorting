# -*- coding: utf-8 -*-
"""在训练图（模型一定见过的图）上排查：为什么分数这么低。

试 4 种预处理，各自打印：
  - 全局最大 sigmoid 分数及其类别
  - 标注框内（GT）的最大分数
若某种预处理能让 GT 框里分数接近 1，说明就是预处理（BGR/RGB、NV12/NV21）不对。
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
IMG = "/tmp/val_imgs/023.jpg"
LBL = "/tmp/val_lbls/023.txt"


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


def gt_boxes(path, w, h):
    out = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                            (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def run(model, canvas_bgr):
    """返回 (score_map per stride, cls_ids) —— score 已 sigmoid。"""
    inp = nv12_of(canvas_bgr)
    outs = model.forward(inp)
    res = []
    for k in (0, 2, 4):
        logits = np.asarray(outs[k].buffer, np.float32).reshape(-1)
        res.append(logits)
    return res


def main():
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    print("图 %s %dx%d" % (IMG, w, h))
    gts = gt_boxes(LBL, w, h)
    print("GT 框 %d 个，类别: %s" % (len(gts), sorted(set(g[0] for g in gts))))

    model = dnn.load(BIN)[0]
    variants = [
        ("BGR nv12", img, False),
        ("RGB nv12", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), False),
        ("BGR nv21", img, True),
        ("RGB nv21", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), True),
    ]
    for tag, src, swap in variants:
        canvas, ratio, dw, dh = Y.letterbox_image(src, 640, 640)
        if swap:
            inp = nv12_of(canvas, swap_uv=True)
            outs = model.forward(inp)
        else:
            inp = nv12_of(canvas, swap_uv=False)
            outs = model.forward(inp)
        best_all = -1e9
        best_gt = -1e9
        cls_all = -1
        # 每个 stride 的 cls 输出
        for k, stride in ((0, 8), (2, 16), (4, 32)):
            logits = np.asarray(outs[k].buffer, np.float32)[0]     # (h,w,C)
            gh, gw = logits.shape[0], logits.shape[1]
            prob = 1.0 / (1.0 + np.exp(-logits))
            gmax = float(prob.max())
            if gmax > best_all:
                best_all = gmax
                cls_all = int(prob.reshape(-1, prob.shape[-1]).max(axis=-1).argmax())
            # GT 框中心落在哪个网格
            for (c, x1, y1, x2, y2) in gts:
                cx = ((x1 + x2) / 2.0 * ratio + dw) / stride
                cy = ((y1 + y2) / 2.0 * ratio + dh) / stride
                gx, gy = int(min(gw - 1, max(0, cx))), int(min(gh - 1, max(0, cy)))
                v = float(prob[gy, gx].max())
                best_gt = max(best_gt, v)
        print("  %-10s 全局最高 sigmoid=%.3f (cls %d)   GT 中心最高=%.3f"
              % (tag, best_all, cls_all, best_gt))


if __name__ == "__main__":
    main()