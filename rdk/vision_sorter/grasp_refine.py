# -*- coding: utf-8 -*-
"""抓取点精修：把 YOLO 的松框收紧到物体颜色轮廓上。

为什么需要
----------
YOLO 给的框天生比物体大一圈（回归误差 + 阴影 + stride 8 下采样网格），
框中心跟着一起偏。实测：同一颗绿料，框中心与颜色质心差 3~8px；
640 宽画面约 400mm 视场，1px ≈ 0.27mm，偏 8px 就是 2mm 以上，夹爪会抓偏。
而物料是单色塑料件，**颜色边界比模型回归的边界准得多**，
所以抓取点应该用颜色分割算出来的点，而不是框中心。

做法（全部在 YOLO 框内做，不改变形状分类结果）
--------------------------------------------
1. 按判出来的颜色做 HSV inRange（红色跨 0°，两段取并集）；
2. 开运算去噪点，闭运算填高光/反光打出的孔洞；
3. 取最大连通域（框内其他小色斑一律丢掉），再填内部孔洞；
4. 三种抓取点候选：
     centroid  颜色质心（默认，抗形状不对称）
     rect      最小外接矩形中心（抗边缘毛刺、局部缺色）
     bbox      框中心（也就是不做精修，回退用）
5. 顺带给出：贴合框、最小外接矩形(宽,高,角度)、最小外接圆半径、颜色填充率。

填充率 fill = 轮廓面积 / 框面积。太小时（< GRASP_MIN_FILL）说明框里
几乎没有该颜色，判色本身就可疑，直接回退到框中心并报 ok=False。
"""
import cv2
import numpy as np

import config as C


def color_mask(hsv_img, color_name):
    """按颜色名生成二值掩膜；颜色名不在表里返回 None。红色两段取并集。"""
    rngs = C.COLOR_RANGES_HSV.get(color_name)
    if rngs is None:
        return None
    lo, hi = rngs[0], rngs[-1]
    m = cv2.inRange(hsv_img, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if color_name == "red":
        m2 = cv2.inRange(hsv_img,
                         np.array(C.RED_HSV_LOWER2[0], np.uint8),
                         np.array(C.RED_HSV_LOWER2[1], np.uint8))
        m = cv2.bitwise_or(m, m2)
    return m


def _fallback(bbox, reason):
    x1, y1, x2, y2 = bbox
    return {
        "ok": False,
        "reason": reason,
        "point": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        "bbox": tuple(bbox),
        "tight_bbox": None,
        "centroid": None,
        "rect_center": None,
        "rect_size": None,
        "angle": None,
        "center": None,
        "radius": None,
        "fill": 0.0,
        "mask_px": 0,
        "mode": "bbox",
    }


def refine(frame_bgr, bbox, color_name, point_mode=None, margin=None,
           min_fill=None):
    """在 frame_bgr 上用 color_name 的颜色贴合 bbox 里的物体，返回抓取点。

    参数
    ----
    frame_bgr : np.ndarray | None  原始 BGR 帧（未去畸变，与 bbox 同坐标系）
    bbox      : [x1, y1, x2, y2]   YOLO 框（像素，可越界）
    color_name: 已判定的颜色名（config.COLOR_RANGES_HSV 的键）
    point_mode: "centroid" / "rect" / "bbox"，None 则读 config.GRASP_POINT_MODE
    margin    : 框向外扩的比例（默认 config.GRASP_MASK_MARGIN），
                给"框偏小、物体出框"留余量
    min_fill  : 最小填充率（默认 config.GRASP_MIN_FILL）

    返回 dict：ok / reason / point / centroid / rect_center / tight_bbox /
               rect_size / angle / center / radius / fill / mask_px / mode
    point 永远有效（失败时就是框中心），调用方不必区分成功与否。
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    base = (x1, y1, x2, y2)
    if frame_bgr is None or frame_bgr.size == 0:
        return _fallback(base, "no frame")

    H, W = frame_bgr.shape[:2]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    margin = getattr(C, "GRASP_MASK_MARGIN", 0.12) if margin is None else margin
    min_fill = getattr(C, "GRASP_MIN_FILL", 0.10) if min_fill is None else min_fill
    mx, my = int(round(bw * margin)), int(round(bh * margin))
    X1, Y1 = max(0, int(round(x1)) - mx), max(0, int(round(y1)) - my)
    X2, Y2 = min(W, int(round(x2)) + mx), min(H, int(round(y2)) + my)
    if X2 - X1 < 6 or Y2 - Y1 < 6:
        return _fallback(base, "crop too small")

    crop = frame_bgr[Y1:Y2, X1:X2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = color_mask(hsv, color_name)
    if m is None:
        return _fallback(base, "unknown color %r" % (color_name,))

    # 开运算去掉孤立噪点（反光、地面杂色），闭运算把高光打穿的洞补回来
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _fallback(base, "no contour")
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    box_area = float(max(1.0, bw * bh))
    fill = area / box_area
    if fill < min_fill:
        return _fallback(base, "mask too small (fill=%.3f)" % fill)

    # 填内部孔洞后再求质心，避免高光把质心拉偏
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 255, -1)
    M = cv2.moments(filled)
    if M["m00"] <= 0:
        return _fallback(base, "zero moments")
    cx = X1 + M["m10"] / M["m00"]
    cy = Y1 + M["m01"] / M["m00"]

    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(c)
    # minAreaRect 的 (w,h) 顺序不保证，统一成长边在前，免得下游/日志看晕
    if rw < rh:
        rw, rh = rh, rw
    (ccx, ccy), radius = cv2.minEnclosingCircle(c)
    tx, ty, tw, th = cv2.boundingRect(c)

    mode = (point_mode or getattr(C, "GRASP_POINT_MODE", "centroid")).lower()
    if mode == "rect":
        point = (X1 + float(rcx), Y1 + float(rcy))
    elif mode == "bbox":
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    else:
        mode = "centroid"
        point = (float(cx), float(cy))

    return {
        "ok": True,
        "reason": "",
        "point": point,
        "bbox": base,
        "tight_bbox": (X1 + tx, Y1 + ty, X1 + tx + tw, Y1 + ty + th),
        "centroid": (float(cx), float(cy)),
        "rect_center": (X1 + float(rcx), Y1 + float(rcy)),
        "rect_size": (float(rw), float(rh)),
        "angle": float(ang),
        "center": (X1 + float(ccx), Y1 + float(ccy)),
        "radius": float(radius),
        "fill": float(fill),
        "mask_px": int(np.count_nonzero(filled)),
        "mode": mode,
    }


def describe(res):
    """一行摘要，给日志用。"""
    if not res["ok"]:
        return "精修回退(%s) 用框中心=(%.1f,%.1f)" % (
            res["reason"], res["point"][0], res["point"][1])
    cx, cy = res["point"]
    tc = res["tight_bbox"]
    return ("精修[%s] 抓取点=(%.1f,%.1f) 贴合框=(%d,%d,%d,%d) "
            "填充=%.0f%% 尺寸=%.1fx%.1f 角度=%.0f° 半径=%.1f"
            % (res["mode"], cx, cy, tc[0], tc[1], tc[2], tc[3],
               res["fill"] * 100, res["rect_size"][0], res["rect_size"][1],
               res["angle"], res["radius"]))