# -*- coding: utf-8 -*-
"""打印 6 个输出的真实形状与统计，判断哪个是 cls、哪个是 reg。

同时用标注框检验：GT 中心所在网格的 cls 分数高不高；reg 的 DFL 期望距离是否合理。
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
    print("图 %dx%d letterbox ratio=%.4f pad=(%.1f,%.1f)" % (w, h, ratio, dw, dh))
    arrs = []
    for i, o in enumerate(outs):
        a = np.asarray(o.buffer, np.float32)
        arrs.append(a)
        print("输出[%d] shape=%-16s size=%-8d min=%.3f max=%.3f mean=%.3f"
              % (i, str(a.shape), a.size, a.min(), a.max(), a.mean()))

    # 找出 cls 候选：最后一维或第二维等于类别数
    cls_idx = [i for i, a in enumerate(arrs) if a.shape[-1] == len(NAMES) or a.shape[1] == len(NAMES)]
    print("疑似 cls 输出下标:", cls_idx)

    gts = load_gt(LBL, w, h)
    print("\nGT %d 个:" % len(gts))
    for g in gts:
        print("   cls=%s(%d) 中心=(%.0f,%.0f) 尺寸=(%.0fx%.0f)"
              % (NAMES[g[0]] if g[0] < len(NAMES) else "?", g[0], g[1], g[2], g[3], g[4]))

    for i in cls_idx:
        a = arrs[i]
        # 统一成 (1,h,w,C)
        if a.shape[-1] == len(NAMES):
            t = a[0]
        else:
            t = np.transpose(a[0], (1, 2, 0))
        gh, gw = t.shape[0], t.shape[1]
        stride = 640 // gh
        prob = Y.sigmoid(t)
        print("\n--- 输出[%d] 当 cls 看 (h=%d w=%d stride=%d) 原始值域 %.2f..%.2f"
              % (i, gh, gw, stride, t.min(), t.max()))
        print("    sigmoid 全局最大=%.4f  超过0.25的格子数=%d  超过0.5的=%d"
              % (float(prob.max()), int((prob.max(axis=-1) > 0.25).sum()),
                 int((prob.max(axis=-1) > 0.5).sum())))
        for g in gts:
            gx = int(min(gw - 1, max(0, (g[1] * ratio + dw) / stride)))
            gy = int(min(gh - 1, max(0, (g[2] * ratio + dh) / stride)))
            print("    GT%-5s 网格(%2d,%2d) cls分数=%.4f  (该类 %.2f)" %
                  (NAMES[g[0]], gx, gy, float(prob[gy, gx].max()), float(prob[gy, gx, g[0]])))


if __name__ == "__main__":
    main()