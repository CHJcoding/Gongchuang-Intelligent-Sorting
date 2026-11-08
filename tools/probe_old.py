# -*- coding: utf-8 -*-
"""A/B 对照：拿「之前确实能检测出结果」的旧 7 类模型，用完全相同的喂图方式跑一遍。

如果旧模型输出正常（有高分类得分）→ 我们的输入方式没问题，问题在新 bin。
如果旧模型也全是死值 → 问题在 pyeasy_dnn / 我们的 nv12 构造。
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
from hobot_dnn import pyeasy_dnn as dnn  # noqa: E402
import yolov8_bpu as Y  # noqa: E402

IMG = "/tmp/val_imgs/023.jpg"
OLD = "/root/vision_ws/yolov5_best.bin"


def main():
    img = cv2.imread(IMG)
    old = dnn.load(OLD)[0]
    for size in (672, 640):
        canvas, ratio, dw, dh = Y.letterbox_image(img, size, size)
        try:
            outs = old.forward(Y.bgr_to_nv12(canvas))
        except Exception as e:
            print("%d 输入失败: %s: %s" % (size, type(e).__name__, e))
            continue
        print("--- 旧模型 %d 输入：%d 个输出" % (size, len(outs)))
        for i, o in enumerate(outs):
            a = np.asarray(o.buffer, np.float32)
            s = Y.sigmoid(a)
            print("  输出[%d] shape=%-18s min=%8.3f max=%8.3f  最大sigmoid=%.4f"
                  % (i, str(a.shape), a.min(), a.max(), s.max()))


if __name__ == "__main__":
    main()