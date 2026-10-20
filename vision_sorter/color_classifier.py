# -*- coding: utf-8 -*-
"""
颜色分类：在YOLO检测框ROI内做HSV投票，自动剔除高光/灰像素。
对深红/暗橙加二次判别。
"""
import cv2
import numpy as np
import config as C


def _valid_mask(hsv_roi):
    """剔除高光和低饱和灰像素，返回有效像素掩膜。"""
    h, s, v = cv2.split(hsv_roi)
    valid = (v < C.V_HIGH_THRES) & (s > C.S_LOW_THRES)
    return valid.astype(np.uint8) * 255


def _vote_color(hsv_roi, valid):
    """对每个颜色区间统计落在mask内的像素数，返回投票结果。"""
    scores = {}
    for name, (lo, hi) in C.COLOR_RANGES_HSV.items():
        m = cv2.inRange(hsv_roi, np.array(lo), np.array(hi))
        if name == "red":
            m2 = cv2.inRange(hsv_roi,
                             np.array(C.RED_HSV_LOWER2[0]),
                             np.array(C.RED_HSV_LOWER2[1]))
            m = cv2.bitwise_or(m, m2)
        m = cv2.bitwise_and(m, valid)
        scores[name] = int(cv2.countNonZero(m))
    return scores


def classify_color(bgr_roi):
    """
    输入: YOLO检测到的BGR ROI图
    返回: (color_name, confidence 0~1) 或 (None, 0.0) 识别失败
    """
    h, w = bgr_roi.shape[:2]
    if h < 10 or w < 10:
        return None, 0.0

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    valid = _valid_mask(hsv)
    valid_pixels = cv2.countNonZero(valid)
    total = h * w
    if valid_pixels / total < C.MIN_PIXEL_RATIO:
        return None, 0.0   # 高光太多或颜色太灰，判色失败

    scores = _vote_color(hsv, valid)
    sorted_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name, top_cnt = sorted_c[0]
    second_cnt = sorted_c[1][1] if len(sorted_c) > 1 else 0

    if top_cnt == 0:
        return None, 0.0

    conf = top_cnt / (top_cnt + second_cnt + 1e-6)

    # 易混色二次判别：深红 vs 暗橙
    if top_name == "red" and conf < 0.75:
        b, g, r = cv2.split(bgr_roi)
        v = valid // 255  # 0/1
        denom = (r.astype(np.float32) + g.astype(np.float32) + 1e-6) * v
        ratio = float(np.sum(r * v) / (np.sum(r + g + 1e-6) * 0 + np.sum(denom) + 1e-6))
        # 上面这个写法有点绕，直接算ROI内有效红像素的R/(R+G)均值
        mask_bool = valid > 0
        if mask_bool.sum() > 20:
            r_m = r[mask_bool].astype(np.float32)
            g_m = g[mask_bool].astype(np.float32)
            ratio = float(np.mean(r_m / (r_m + g_m + 1e-6)))
            if ratio < C.DARK_RED_RATIO_THRES:
                top_name = "orange"   # 偏橙，判成暗橙

    return top_name, round(conf, 3)


if __name__ == "__main__":
    # 简单自测：拍一帧，在画面上随便框个区域看颜色
    import sys
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("用法: python color_classifier.py <test.jpg>")
    else:
        name, conf = classify_color(img)
        print(f"color={name}, conf={conf}")
        cv2.imshow("roi", img)
        cv2.waitKey(0)
