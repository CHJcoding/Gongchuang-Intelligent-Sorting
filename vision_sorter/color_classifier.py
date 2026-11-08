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


def _vote(hsv_roi, chiral_mask=None):
    """对每个 HSV 区间统计命中像素数。

    chiral_mask 不为 None 时，只统计"自身是彩色像素"的命中：
    深色台面 / 白纸背景会被 black、white 区间吃进去，把真颜色的票数稀释掉
    （实测绿料 conf 只有 0.60，正好压在判色下限上，会随机判失败）。
    """
    scores = {}
    disabled = getattr(C, "COLOR_DISABLED", frozenset())
    for name, (lo, hi) in C.COLOR_RANGES_HSV.items():
        if name in disabled:
            continue        # 被屏蔽的颜色不参评（如黑色夹爪原地空抓）
        m = cv2.inRange(hsv_roi, np.array(lo), np.array(hi))
        if name == "red":
            m2 = cv2.inRange(hsv_roi,
                             np.array(C.RED_HSV_LOWER2[0]),
                             np.array(C.RED_HSV_LOWER2[1]))
            m = cv2.bitwise_or(m, m2)
        if chiral_mask is not None:
            m = cv2.bitwise_and(m, chiral_mask)
        scores[name] = int(cv2.countNonZero(m))
    return scores


def _chromatic_mask(hsv_roi):
    """彩色像素掩码：S 和 V 都够高，排除深色背景与白色台面。"""
    S = hsv_roi[:, :, 1]
    V = hsv_roi[:, :, 2]
    m = ((S >= C.CHROMATIC_S_MIN) & (V >= C.CHROMATIC_V_MIN))
    return m.astype(np.uint8) * 255


def color_diag(bgr_roi):
    """判色取证：一行说明 ROI 的 HSV 中位数、彩色占比和前两名的票数占比。"""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    total = float(max(hsv.shape[0] * hsv.shape[1], 1))
    chiral = _chromatic_mask(hsv)
    frac = cv2.countNonZero(chiral) / total
    votes = _vote(hsv, chiral if frac >= C.CHROMATIC_MIN_FRAC else None)
    top = sorted(votes.items(), key=lambda kv: -kv[1])[:2]
    return "HSV中位(%d,%d,%d) 彩色%.0f%% 票=%s" % (
        int(np.median(hsv[:, :, 0])), int(np.median(hsv[:, :, 1])),
        int(np.median(hsv[:, :, 2])), frac * 100,
        " ".join("%s:%.0f%%" % (k, 100.0 * v / total) for k, v in top))

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

    # 彩色像素占比够高 -> 只在彩色像素上投票；否则（黑/白/灰物料）退回全像素投票。
    chiral = _chromatic_mask(hsv)
    chiral_frac = cv2.countNonZero(chiral) / total
    if chiral_frac >= C.CHROMATIC_MIN_FRAC:
        scores = _vote(hsv, chiral)
        basis = "chromatic %.0f%%" % (chiral_frac * 100)
    else:
        scores = _vote(hsv)
        basis = "all pixels (chromatic only %.0f%%)" % (chiral_frac * 100)
    if not scores:
        return None, 0.0
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name, top_cnt = ordered[0]
    second_cnt = ordered[1][1] if len(ordered) > 1 else 0

    # 票数占比过低 = 没有像素落进任何区间（阈值不对/物料本就不是彩色阈值内）
    if top_cnt / total < C.MIN_PIXEL_RATIO:
        return None, 0.0

    conf = top_cnt / (top_cnt + second_cnt + 1e-6)
    if verbose:
        print("  basis:", basis)
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