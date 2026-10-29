# -*- coding: utf-8 -*-
"""决定性测试：模型的输出到底会不会随输入变化？

对两个模型分别喂 全黑 / 全白 / 随机噪声 三种画布，比较：
  - 输出是否逐字节相同（相同 = 输入根本没进网络，或 bin 坏）
  - 分类最大概率

期望：正常模型三种输入的输出必然不同。
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
from hobot_dnn import pyeasy_dnn as dnn  # noqa: E402
import yolov8_bpu as Y  # noqa: E402

MODELS = [
    ("我们的 gc_best_1029", "/root/vision_ws/gc_best_1029.bin"),
    ("官方 yolov5s_672x672", "/opt/hobot/model/x5/basic/yolov5s_672x672_nv12.bin"),
]


def canvases(size=640):
    black = np.zeros((size, size, 3), np.uint8)
    white = np.full((size, size, 3), 255, np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return [("全黑", black), ("全白", white), ("噪声", noise)]


def main():
    for name, path in MODELS:
        print("=" * 60)
        print(name, path)
        try:
            model = dnn.load(path)[0]
        except Exception as e:
            print("  加载失败:", type(e).__name__, e)
            continue
        prev = None
        first = None
        for tag, img in canvases():
            arr = Y.bgr_to_nv12(img)
            try:
                outs = model.forward(arr)
            except Exception as e:
                print("  %-4s forward 失败: %s: %s" % (tag, type(e).__name__, e))
                continue
            bufs = [np.asarray(o.buffer) for o in outs]
            flat = np.concatenate([b.reshape(-1) for b in bufs])
            prob = 1.0 / (1.0 + np.exp(-np.asarray(outs[0].buffer, np.float32)))
            same = ""
            if first is None:
                first = flat.copy()
            else:
                d = float(np.max(np.abs(flat.astype(np.float64) - first.astype(np.float64))))
                same = "  与全黑差异=%.6g %s" % (d, "（完全一样！）" if d == 0 else "")
            print("  %-4s 输出个数=%d 元素数=%d  cls最大概率=%.4f  输出和=%.4f%s"
                  % (tag, len(outs), flat.size, float(prob.max()), float(flat.sum()), same))


if __name__ == "__main__":
    main()