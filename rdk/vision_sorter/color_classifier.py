# -*- coding: utf-8 -*-
"""
颜色分类：纯 HSV 阈值分割，逐色投票，最高票者胜。

原则（按现场约定）：
  * 只认 HSV 区间，不用任何 RGB/中性色/二次判别的额外逻辑。
  * 阈值全部在 config.COLOR_RANGES_HSV 里改。
  * 红色因色相跨越 0°（H 0~10 与 170~179），用两段区间取并集。
"""
import cv2
import numpy as np
import config as C


def _vote(hsv_roi):
    """对每个 HSV 区间统计命中像素数。"""
    scores = {}
    for name, (lo, hi) in C.COLOR_RANGES_HSV.items():
        m = cv2.inRange(hsv_roi, np.array(lo), np.array(hi))
        if name == "red":
            m2 = cv2.inRange(hsv_roi,
                             np.array(C.RED_HSV_LOWER2[0]),
                             np.array(C.RED_HSV_LOWER2[1]))
            m = cv2.bitwise_or(m, m2)
        scores[name] = int(cv2.countNonZero(m))
    return scores


def classify_color(bgr_roi, verbose=False):
    """
    输入: YOLO 检测到的 BGR ROI 图
    返回: (color_name, confidence 0~1) 或 (None, 0.0) 识别失败
    """
    h, w = bgr_roi.shape[:2]
    if h < 10 or w < 10:
        return None, 0.0

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    total = float(h * w)

    scores = _vote(hsv)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name, top_cnt = ordered[0]
    second_cnt = ordered[1][1] if len(ordered) > 1 else 0

    # 票数占比过低 = 没有像素落进任何区间（阈值不对/物料本就不是彩色阈值内）
    if top_cnt / total < C.MIN_PIXEL_RATIO:
        return None, 0.0

    conf = top_cnt / (top_cnt + second_cnt + 1e-6)
    if verbose:
        print("  votes:", {k: v for k, v in ordered[:3]})
        print("  ratio=%.3f conf=%.3f" % (top_cnt / total, conf))
    return top_name, round(conf, 3)


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("用法: python color_classifier.py <test.jpg> [x y w h]")
    else:
        roi = img
        if len(sys.argv) >= 6:
            x, y, w, h = (int(v) for v in sys.argv[2:6])
            roi = img[max(0, y):y + h, max(0, x):x + w]
        name, conf = classify_color(roi, verbose=True)
        print("color=%s, conf=%.3f" % (name, conf))