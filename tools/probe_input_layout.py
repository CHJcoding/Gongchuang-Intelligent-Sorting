# -*- coding: utf-8 -*-
"""排查输入布局：模型到底要什么形状/通道序的输入。

分别喂：
  a) nv12 (1, H*3/2, W, 1)     ← TROS 官方 example 的做法
  b) NHWC RGB (1, H, W, 3)
  c) NCHW RGB (1, 3, H, W)
  d) NCHW BGR (1, 3, H, W)
  e) NHWC BGR (1, H, W, 3)
打印每种的最大分类概率。哪种明显高就是哪种。
"""
import sys

import cv2
import numpy as np

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, "/root/vision_ws/vision_sorter")

from hobot_dnn import pyeasy_dnn as dnn  # noqa: E402
import yolov8_bpu as Y  # noqa: E402

BIN = "/root/vision_ws/gc_best_1029.bin"
IMG = "/tmp/val_imgs/023.jpg"


def as_inputs(img, size=640):
    canvas, ratio, dw, dh = Y.letterbox_image(img, size, size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    out = {}
    out["a nv12 (1,960,640,1)"] = Y.bgr_to_nv12(canvas)
    out["b NHWC RGB"] = rgb.reshape(1, size, size, 3)
    out["c NCHW RGB"] = np.transpose(rgb, (2, 0, 1)).reshape(1, 3, size, size)
    out["d NCHW BGR"] = np.transpose(canvas, (2, 0, 1)).reshape(1, 3, size, size)
    out["e NHWC BGR"] = canvas.reshape(1, size, size, 3)
    return out, ratio, dw, dh


def main():
    img = cv2.imread(IMG)
    print("图 %s %dx%d" % (IMG, img.shape[1], img.shape[0]))
    model = dnn.load(BIN)[0]
    print("模型输入 shape:", model.inputs[0].properties.shape,
          "dtype:", model.inputs[0].properties.dtype,
          "layout:", getattr(model.inputs[0].properties, "layout", None))
    inputs, _, _, _ = as_inputs(img)
    for tag, arr in inputs.items():
        try:
            outs = model.forward(arr)
            logits = np.asarray(outs[0].buffer, np.float32)
            prob = 1.0 / (1.0 + np.exp(-logits))
            print("  %-22s shape=%s -> cls 最大概率=%.4f  最大 logits=%.2f"
                  % (tag, arr.shape, float(prob.max()), float(logits.max())))
        except Exception as e:
            print("  %-22s shape=%s -> 失败: %s: %s"
                  % (tag, arr.shape, type(e).__name__, e))


if __name__ == "__main__":
    main()