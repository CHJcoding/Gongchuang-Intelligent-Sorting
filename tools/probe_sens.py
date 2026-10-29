# -*- coding: utf-8 -*-
"""两个决定性判据。

A) 输入敏感性：同一形状下喂「训练图」和「反相训练图」，比较每个输出的最大逐元素差异。
   正常模型差异会很大（分类得分可以从 0 变到 1）；若差异只剩 1~2，说明输入几乎没进网络。
B) DFL 合理性：在标注框中心格子解 DFL，看 l/t/r/b 是否接近真实框半径。
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

    def run(bgr):
        canvas, ratio, dw, dh = Y.letterbox_image(bgr, 640, 640)
        outs = model.forward(Y.bgr_to_nv12(canvas))
        return [np.asarray(o.buffer, np.float32) for o in outs], ratio, dw, dh

    a, ratio, dw, dh = run(img)
    b, _, _, _ = run(255 - img)

    print("=== A) 输入敏感性（训练图 vs 反相图，同形状）===")
    for i in range(len(a)):
        d = float(np.max(np.abs(a[i] - b[i])))
        rng = float(a[i].max() - a[i].min())
        print("  输出[%d] shape=%-16s max|diff|=%.3f  自身值域幅度=%.2f  差异占比=%.1f%%"
              % (i, str(a[i].shape), d, rng, 100.0 * d / max(1e-6, rng)))

    print("\n=== B) GT 中心格子的 DFL 解码（单位像素）===")
    gts = load_gt(LBL, w, h)
    for stride, reg_i in ((8, 1), (16, 3), (32, 5)):
        reg = a[reg_i][0]
        gh = reg.shape[0]
        d = Y.dfl_decode(reg, stride)
        print("  stride%-3d 全图中位 l/t/r/b = %s  (值域 l:%.1f..%.1f)"
              % (stride, np.round(np.median(d.reshape(-1, 4), axis=0), 1),
                 float(d[..., 0].min()), float(d[..., 0].max())))
        for g in gts:
            gx = int(min(gh - 1, max(0, (g[1] * ratio + dw) / stride)))
            gy = int(min(gh - 1, max(0, (g[2] * ratio + dh) / stride)))
            l, t, rr, bo = d[gy, gx]
            print("    GT%-5s 真实半宽/半高=%.0f/%.0f | l=%.1f t=%.1f r=%.1f b=%.1f → 宽=%.1f 高=%.1f"
                  % (NAMES[g[0]], g[3] / 2, g[4] / 2, l, t, rr, bo, l + rr, t + bo))


if __name__ == "__main__":
    main()