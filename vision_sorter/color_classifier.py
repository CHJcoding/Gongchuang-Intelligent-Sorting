# -*- coding: utf-8 -*-
"""
颜色分类：在 YOLO 检测框 ROI 内做 HSV 投票。

设计要点（2026-10-21 板端实测后重写）：
  * 旧版用 (V<230)&(S>40) 做有效掩膜，过曝场景下 53% 像素 V=255 被全部剔除，
    ROI 有效像素只剩 6.8% < MIN_PIXEL_RATIO，导致永远判色失败。
  * 旧版 white/black 区间要求低饱和，却被 S>40 的掩膜排除，永远不可能胜出。
  新版：
    1) 先去高光：高光 = 高 V 且低 S（镜面反射），只按 S 门限筛"彩色像素"，不砍高 V。
    2) 中性色（白/灰/黑）单独判定，不参与色相投票。
    3) 彩色像素足够时才做色相投票；否则按白/黑/失败处理。
"""
import cv2
import numpy as np
import config as C

# 中性色不在色相上投票（它们的 HSV 区间没有色相意义）
NEUTRAL_NAMES = ("black", "white")


def _masks(hsv_roi):
    """返回 uint8 掩膜（0/255）：彩色 / 白灰 / 暗。"""
    h, s, v = cv2.split(hsv_roi)
    chromatic = (s > C.S_LOW_THRES).astype(np.uint8) * 255
    whiteish = ((s <= C.S_LOW_THRES) & (v > C.WHITE_V_MIN)).astype(np.uint8) * 255
    darkish = (v < C.BLACK_V_MAX).astype(np.uint8) * 255
    return chromatic, whiteish, darkish


def _vote_color(hsv_roi, chromatic):
    """对彩色区间统计落在掩膜内的像素数。"""
    scores = {}
    for name, (lo, hi) in C.COLOR_RANGES_HSV.items():
        if name in NEUTRAL_NAMES:
            continue
        m = cv2.inRange(hsv_roi, np.array(lo), np.array(hi))
        if name == "red":
            m2 = cv2.inRange(hsv_roi,
                             np.array(C.RED_HSV_LOWER2[0]),
                             np.array(C.RED_HSV_LOWER2[1]))
            m = cv2.bitwise_or(m, m2)
        m = cv2.bitwise_and(m, chromatic)
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
    chromatic, whiteish, darkish = _masks(hsv)
    total = float(h * w)
    r_chroma = float(np.count_nonzero(chromatic)) / total
    r_white = float(np.count_nonzero(whiteish)) / total
    r_dark = float(np.count_nonzero(darkish)) / total

    if verbose:
        print("  [color] chroma=%.2f white=%.2f dark=%.2f" %
              (r_chroma, r_white, r_dark))

    # ---- 彩色像素不足：按中性色判 ----
    if r_chroma < C.MIN_PIXEL_RATIO:
        if r_white >= C.NEUTRAL_MIN_RATIO:
            return "white", round(r_white, 3)
        if r_dark >= C.NEUTRAL_MIN_RATIO:
            return "black", round(r_dark, 3)
        return None, 0.0

    # ---- 色相投票 ----
    scores = _vote_color(hsv, chromatic)
    sorted_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name, top_cnt = sorted_c[0]
    second_cnt = sorted_c[1][1] if len(sorted_c) > 1 else 0
    if top_cnt == 0:
        return None, 0.0

    conf = top_cnt / (top_cnt + second_cnt + 1e-6)

    # 易混色二次判别：深红 vs 暗橙（用有效彩色像素的 R/(R+G) 均值）
    if top_name == "red" and conf < 0.75:
        b, g, r = cv2.split(bgr_roi)
        mask_bool = chromatic > 0
        if mask_bool.sum() > 20:
            r_m = r[mask_bool].astype(np.float32)
            g_m = g[mask_bool].astype(np.float32)
            ratio = float(np.mean(r_m / (r_m + g_m + 1e-6)))
            if ratio < C.DARK_RED_RATIO_THRES:
                top_name = "orange"

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