# -*- coding: utf-8 -*-
"""两个判据：cls 峰值是否落在物体上；DFL 距离是否贴近真实尺寸。

1) 取 stride8 cls 输出得分最高的 10 个格子，换算成原图像素坐标，和标注框中心比距离。
   若落在物体上 → 只是整体偏置（可修正）；若乱落 → bin 不可用。
2) 在标注框中心格子解 DFL，看 l/t/r/b 像素是否接近真实框的一半。
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
from hobot_dnn import pyeasy_dnn as dnn  # noqa: E402
import yolov8_bpu as Y  # noqa: E402

BIN = "/root/vision_ws/gc_best_1029.bin"
IMG = "/tmp/val_imgs/023.jpg"
LBL = "/tmp/val_lbls/023.txt"
NAMES = ["BQ", "LK", "SLZ", "SMT", "WLZ", "YZ", "YZHU", "cube"]


def load_gt(path, w, h):
    out = []
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                out.append((c, cx * w, cy * h, bw * w, bh * h))
    return out


def main():
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    model = dnn.load(BIN)[0]
    canvas, ratio, dw, dh = Y.letterbox_image(img, 640, 640)
    outs = model.forward(Y.bgr_to_nv12(canvas))
    gts = load_gt(LBL, w, h)

    print("=== 1) stride8 cls 峰值位置 ===")
    cls = np.asarray(outs[0].buffer, np.float32)[0]          # (80,80,8)
    flat = cls.reshape(-1)
    order = np.argsort(flat)[::-1][:10]
    for rank, idx in enumerate(order, 1):
        cell, c = divmod(int(idx), 8)
        gy, gx = divmod(cell, 80)
        px = gx * 8 + 4
        py = (gy * 8 + 4 - dh) / ratio
        if py < 0 or py > h:
            near = -1.0
        else:
            near = min(np.hypot(px - g[1], py - g[2]) for g in gts)
        print("  #%2d 值=%7.3f cls=%-4s 网格(%2d,%2d) → 原图(%.0f,%.0f) 距最近物体中心=%.0fpx"
              % (rank, flat[idx], NAMES[c], gx, gy, px, py, near))

    print("\n=== 2) GT 中心格子的 DFL 距离（像素）===")
    for stride, reg_i in ((8, 1), (16, 3), (32, 5)):
        reg = np.asarray(outs[reg_i].buffer, np.float32)[0]   # (h,w,64)
        gh = reg.shape[0]
        d = Y.dfl_decode(reg[None], stride)                  # (1,h,w,4)
        d = d[0]
        print(" stride%-3d 整体 l/t/r/b 中位数 = %s px"
              % (stride, np.round(np.median(d.reshape(-1, 4), axis=0), 1)))
        for g in gts:
            gx = int(min(gh - 1, max(0, (g[1] * ratio + dw) / stride)))
            gy = int(min(gh - 1, max(0, (g[2] * ratio + dh) / stride)))
            l, t, rr, b = d[gy, gx]
            print("   GT%-5s 真实半宽%.0f 半高%.0f | 解码 l=%.1f t=%.1f r=%.1f b=%.1f 宽=%.1f 高=%.1f"
                  % (NAMES[g[0]], g[3] / 2, g[4] / 2, l, t, rr, b, l + rr, t + b))


if __name__ == "__main__":
    main()