# -*- coding: utf-8 -*-
"""抓取点精修：把 YOLO 的松框收紧到物体本身，再取抓取点。

为什么需要
----------
YOLO 给的框天生比物体大一圈（回归误差 + 阴影 + stride 8 下采样网格）。
实测同一盘料：框比实物宽 10~12px、高 11~13px，框中心与颜色质心差 3~11px；
可见视场约 640px / 400mm，1px ≈ 0.27mm，偏 10px 就是 2.7mm，夹爪会抓偏。
而物料是单色塑料件，**颜色边界比模型回归的边界准得多**，
所以抓取点应该从颜色分割里算，而不是用框中心。

做法（全部在 YOLO 框内做，不改变形状分类结果）
--------------------------------------------
1. 先按判出的颜色做全局 HSV inRange（红色跨 0°，两段取并集）——粗掩膜；
2. 若粗掩膜铺满整个框（说明这个颜色区间把背景也框进来了，蓝色最容易这样），
   再用「框内像素的中位 HSV」重建一个收窄的窗口（自适应掩膜）；
3. 两个候选掩膜都做：开运算去噪、闭运算补高光孔洞、取与框重叠最大的连通域、填内部孔洞；
4. 用硬门槛筛掉不可信的掩膜（贴合框/框的宽高比、实心度、贴合框内覆盖率），
   过了门槛的才允许改抓取点，否则**回退框中心并报 ok=False**——绝不静默下发坏点；
5. 抓取点三选一：
     centroid  颜色质心（默认，抗形状不对称）
     rect      最小外接矩形中心（抗边缘毛刺、局部缺色）
     bbox      框中心（等价于不做精修）
6. 顺带给出：贴合框、最小外接矩形(宽,高,角度)、最小外接圆半径、覆盖率、实心度。

不能信的情况（回退）
--------------------
- 掩膜太小（< GRASP_MIN_FILL）：框里几乎没有该颜色，判色本身就可疑；
- 掩膜铺满框甚至糊到背景（ratio > GRASP_RATIO_HI）：这个颜色把背景也框进来了；
- 实心度低（< GRASP_MIN_SOLIDITY）：掩膜是散点/环形，质心没有意义；
- 框内覆盖率低（< GRASP_MIN_COVERAGE）：贴合框里只有一小块是"物体"。
"""
import cv2
import numpy as np

import config as C

# 默认参数（config 里同名项优先；缺项就用这里的值，方便老版本 config 兼容）
_DEFAULTS = {
    "GRASP_MASK_MARGIN": 0.12,     # 裁剪区 = 框外扩比例
    "GRASP_MIN_FILL": 0.10,        # 掩膜面积 / 框面积 下限
    "GRASP_MAX_FILL": 1.60,        # 掩膜面积 / 框面积 上限（超了=糊到背景）
    "GRASP_MIN_SOLIDITY": 0.72,    # 轮廓面积 / 凸包面积 下限
    # 覆盖率下限别定高：三角/四面体的俯视投影本来就只有外接框的一半，
    # 实测绿色三棱柱 0.54 会被 0.55 误杀。真正防"只切到一块碎片"的是下面两项。
    "GRASP_MIN_COVERAGE": 0.40,    # 填充后掩膜 / 贴合框面积 下限
    "GRASP_MAX_CENTER_OFF": 0.35,  # 质心离框中心的距离 / 框短边 上限
    "GRASP_RATIO_LO": 0.45,        # 贴合框宽高 / 框宽高 的合理区间
    "GRASP_RATIO_HI": 1.15,
    "GRASP_POINT_MODE": "centroid",
    "GRASP_REFINE": True,
    "GRASP_ADAPTIVE": True,        # 粗掩膜铺满框时，用框内中位色收窄
    "GRASP_ACHROMATIC_OTSU": True,  # 白/黑物料改用明度 Otsu 兜底分割
    "GRASP_H_TOL": 10,             # 自适应色相半宽
    "GRASP_S_TOL": 70,             # 自适应饱和度半宽
    "GRASP_V_TOL": 70,             # 自适应明度半宽
}


def _cfg(name):
    return getattr(C, name, _DEFAULTS[name])


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


def _adaptive_mask(hsv_crop, coarse, box):
    """用框中心区域的中位 HSV 重建一个收窄的窗口。

    为什么取"框中心区域"而不是整个框：粗掩膜把背景也框进来时（蓝色最常见），
    整框像素的中位数会被背景主导，反而把物体挤出去。而 YOLO 框虽然偏大，
    中心一定落在物体上，取框中心 50% 的patch 统计颜色，背景就压不过物体了。
    返回 None 表示框内颜色像素太少、没法统计。
    """
    bx1, by1, bx2, by2 = box
    bw_, bh_ = max(1, bx2 - bx1), max(1, by2 - by1)
    pw, ph = max(6, int(bw_ * 0.5)), max(6, int(bh_ * 0.5))
    px0 = bx1 + (bw_ - pw) // 2
    py0 = by1 + (bh_ - ph) // 2
    patch = coarse[py0:py0 + ph, px0:px0 + pw]
    px = hsv_crop[py0:py0 + ph, px0:px0 + pw][patch > 0]
    if px.shape[0] < 20:
        # 中心区太小/没色块，退回整框统计
        sub = coarse[by1:by2, bx1:bx2]
        px = hsv_crop[by1:by2, bx1:bx2][sub > 0]
    if px.shape[0] < 20:
        return None
    med = np.median(px, axis=0)
    h = hsv_crop[:, :, 0].astype(np.int16)
    s = hsv_crop[:, :, 1].astype(np.int16)
    v = hsv_crop[:, :, 2].astype(np.int16)
    dh = np.abs(h - int(med[0]))
    dh = np.minimum(dh, 180 - dh)          # 色相是环形的
    m = ((dh <= _cfg("GRASP_H_TOL")) &
         (np.abs(s - int(med[1])) <= _cfg("GRASP_S_TOL")) &
         (np.abs(v - int(med[2])) <= _cfg("GRASP_V_TOL")))
    return (m.astype(np.uint8) * 255)


def _enclosed_components(mask):
    """只保留"不碰画面边缘"的连通域——台面上物料被背景包围，背景一定连边。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape[:2]
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                      # 连边 = 背景/台面
        out[lab == i] = 255
    return out


def _achromatic_mask(hsv_crop, box):
    """白/黑物料的兜底分割：对明度做 Otsu，二分后取"被背景包围"的那一类。

    为什么单靠颜色区间不行：白色物料的 HSV 是 (任意色相, 低饱和, 高亮)，
    而台面/纸面背景**也是**低饱和高亮——颜色区间必然把背景一起框进来，
    实测白色圆柱的掩膜是"围着物体的一圈背景"，实心度只有 0.37，怎么调阈值
    都救不回来。但物体和背景之间总有一条边（明暗差），Otsu 二分正好切在
    那条边上；背景连着画面边缘、物料不连，于是只需要丢掉连边的那一类。

    返回 None 表示这一框里没有可用的二分（例如整框一片均匀）。
    """
    v = cv2.GaussianBlur(hsv_crop[:, :, 2], (5, 5), 0)
    thr, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (v > thr).astype(np.uint8) * 255
    dark = cv2.bitwise_not(bright)
    cands = []
    for name, mk in (("bright", bright), ("dark", dark)):
        enc = _enclosed_components(mk)
        if not np.any(enc):
            continue
        n, lab, stats, cent = cv2.connectedComponentsWithStats(enc, connectivity=8)
        if n < 2:
            continue
        # 取面积最大、且覆盖框中心的那一块
        bx1, by1, bx2, by2 = box
        cx, cy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
        cx = min(max(cx, 0), enc.shape[1] - 1)
        cy = min(max(cy, 0), enc.shape[0] - 1)
        idx = lab[cy, cx]
        if idx == 0:                       # 框中心不在任何块里，取最大块
            idx = 1 + int(np.argmax(stats[1:, 4]))
        if idx == 0:
            continue
        one = np.zeros_like(enc)
        one[lab == idx] = 255
        cands.append((name, one, int(stats[idx, 4])))
    if not cands:
        return None, None
    # 两类都能成块时，取面积更接近框面积的那个（背景类通常大得多）
    bw_, bh_ = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    cands.sort(key=lambda c: abs(c[2] - bw_ * bh_))
    return cands[0][1], cands[0][0]


def _analyse(mask, X1, Y1, box):
    """形态学清理 + 取与框重叠最大的连通域 + 量出各项几何量。"""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    bx1, by1, bx2, by2 = box
    best, best_ov = None, -1
    for c in cnts:
        cm = np.zeros(m.shape, np.uint8)
        cv2.drawContours(cm, [c], -1, 255, -1)
        ov = int(np.count_nonzero(cm[by1:by2, bx1:bx2]))
        if ov > best_ov:
            best, best_ov = c, ov
    if best is None or cv2.contourArea(best) <= 0:
        return None

    # 填内部孔洞：高光/反光打出的洞会把质心拉偏
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, [best], -1, 255, -1)
    M = cv2.moments(filled, binaryImage=True)
    if M["m00"] <= 0:
        return None

    tx, ty, tw, th = cv2.boundingRect(best)
    hull = cv2.convexHull(best)
    hull_area = float(cv2.contourArea(hull))
    area = float(cv2.contourArea(best))
    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(best)
    if rw < rh:                            # 长边在前，日志好读
        rw, rh = rh, rw
    (ccx, ccy), radius = cv2.minEnclosingCircle(best)
    mask_px = int(np.count_nonzero(filled))
    # 框中心是否落在掩膜里、质心离框中心多远：这两项比"覆盖率"更能抓住
    # "只分到物体一侧的一小块"这种坏掩膜（覆盖率对三角形本来就只有 0.5）
    bxc, byc = int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0)
    bxc = min(max(bxc, 0), filled.shape[1] - 1)
    byc = min(max(byc, 0), filled.shape[0] - 1)
    center_covered = bool(filled[byc, bxc] > 0)
    mcx, mcy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    off = ((mcx - bxc) ** 2 + (mcy - byc) ** 2) ** 0.5
    short = max(1.0, min(box[2] - box[0], box[3] - box[1]))
    return {
        "centroid": (X1 + mcx, Y1 + mcy),
        "rect_center": (X1 + float(rcx), Y1 + float(rcy)),
        "circle_center": (X1 + float(ccx), Y1 + float(ccy)),
        "tight_bbox": (X1 + tx, Y1 + ty, X1 + tx + tw, Y1 + ty + th),
        "tw": float(tw), "th": float(th),
        "rect_size": (float(rw), float(rh)),
        "angle": float(ang),
        "radius": float(radius),
        "area": area,
        "mask_px": mask_px,
        "solidity": area / max(1.0, hull_area),
        "coverage": mask_px / max(1.0, float(tw * th)),
        "center_covered": center_covered,
        "centroid_off": off / short,
    }


def _reject_reasons(st, bw, bh, min_fill):
    """返回空列表表示这个掩膜可信；否则列出不可信的原因。"""
    why = []
    fill = st["mask_px"] / max(1.0, bw * bh)
    if fill < min_fill:
        why.append("mask too small (fill=%.2f)" % fill)
    if fill > _cfg("GRASP_MAX_FILL"):
        why.append("mask spills (fill=%.2f)" % fill)
    rw = st["tw"] / max(1.0, bw)
    rh = st["th"] / max(1.0, bh)
    if rw < _cfg("GRASP_RATIO_LO") or rh < _cfg("GRASP_RATIO_LO"):
        why.append("mask too narrow (%.2fx%.2f)" % (rw, rh))
    if rw > _cfg("GRASP_RATIO_HI") or rh > _cfg("GRASP_RATIO_HI"):
        why.append("mask spilled (%.2fx%.2f)" % (rw, rh))
    if st["solidity"] < _cfg("GRASP_MIN_SOLIDITY"):
        why.append("not solid (%.2f)" % st["solidity"])
    if st["coverage"] < _cfg("GRASP_MIN_COVERAGE"):
        why.append("low coverage (%.2f)" % st["coverage"])
    if not st.get("center_covered", True):
        why.append("box center not in mask")
    if st.get("centroid_off", 0.0) > _cfg("GRASP_MAX_CENTER_OFF"):
        why.append("centroid far from box center (%.2f)" % st["centroid_off"])
    return why


def _fallback(bbox, reason, coarse_reasons=None):
    x1, y1, x2, y2 = bbox
    return {
        "ok": False,
        "reason": reason,
        "point": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        "bbox": (x1, y1, x2, y2),
        "tight_bbox": None,
        "centroid": None,
        "rect_center": None,
        "rect_size": None,
        "angle": None,
        "center": None,
        "radius": None,
        "fill": 0.0,
        "mask_px": 0,
        "solidity": 0.0,
        "coverage": 0.0,
        "ratio": None,
        "mode": "bbox",
        "method": None,
        "candidates": coarse_reasons or [],
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

    返回 dict：ok / reason / point / centroid / rect_center / rect_center 等。
    tight_bbox 与 bbox 同格式 (x1, y1, x2, y2)。
    point 永远有效（失败时就是框中心），调用方不必区分成功与否。
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    base = (x1, y1, x2, y2)
    if frame_bgr is None or frame_bgr.size == 0:
        return _fallback(base, "no frame")

    H, W = frame_bgr.shape[:2]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    margin = _cfg("GRASP_MASK_MARGIN") if margin is None else margin
    min_fill = _cfg("GRASP_MIN_FILL") if min_fill is None else min_fill
    mx, my = int(round(bw * margin)), int(round(bh * margin))
    X1, Y1 = max(0, int(round(x1)) - mx), max(0, int(round(y1)) - my)
    X2, Y2 = min(W, int(round(x2)) + mx), min(H, int(round(y2)) + my)
    if X2 - X1 < 6 or Y2 - Y1 < 6:
        return _fallback(base, "crop too small")

    crop = frame_bgr[Y1:Y2, X1:X2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    coarse = color_mask(hsv, color_name)
    if coarse is None:
        return _fallback(base, "unknown color %r" % (color_name,))

    box = (int(round(x1)) - X1, int(round(y1)) - Y1,
           int(round(x2)) - X1, int(round(y2)) - Y1)
    cands = []
    for method, mk in (("color", coarse),
                       ("color_adaptive", _adaptive_mask(hsv, coarse, box)
                        if _cfg("GRASP_ADAPTIVE") else None)):
        if mk is None:
            continue
        st = _analyse(mk, X1, Y1, box)
        if st is None:
            cands.append((method, None, ["no contour"]))
            continue
        cands.append((method, st, _reject_reasons(st, bw, bh, min_fill)))

    # 白/黑物料：颜色区间天生分不开"白物体"和"白台面"，用明度 Otsu 兜底
    if _cfg("GRASP_ACHROMATIC_OTSU") and color_name in ("white", "black"):
        gray_mask, _cls = _achromatic_mask(hsv, box)
        if gray_mask is not None:
            st = _analyse(gray_mask, X1, Y1, box)
            if st is None:
                cands.append(("gray_otsu", None, ["no contour"]))
            else:
                cands.append(("gray_otsu", st,
                              _reject_reasons(st, bw, bh, min_fill)))

    ok_cands = [c for c in cands if c[1] is not None and not c[2]]
    if not ok_cands:
        why = "; ".join("%s:%s" % (m, "/".join(r)) for m, _s, r in cands) or "no contour"
        return _fallback(base, why, [m for m, _s, _r in cands])

    # 优先"贴合框内基本都是物体"的那个；同分取更小的（更贴）
    method, st, _why = max(
        ok_cands, key=lambda c: (round(c[1]["coverage"], 2), -c[1]["tw"] * c[1]["th"]))

    mode = (point_mode or _cfg("GRASP_POINT_MODE")).lower()
    if mode == "rect":
        point = st["rect_center"]
    elif mode == "bbox":
        mode = "bbox"
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    else:
        mode = "centroid"
        point = st["centroid"]

    return {
        "ok": True,
        "reason": "",
        "point": (float(point[0]), float(point[1])),
        "bbox": base,
        "tight_bbox": st["tight_bbox"],
        "centroid": st["centroid"],
        "rect_center": st["rect_center"],
        "rect_size": st["rect_size"],
        "angle": st["angle"],
        "center": st["circle_center"],
        "radius": st["radius"],
        "fill": st["mask_px"] / max(1.0, bw * bh),
        "mask_px": st["mask_px"],
        "solidity": st["solidity"],
        "coverage": st["coverage"],
        "ratio": (st["tw"] / max(1.0, bw), st["th"] / max(1.0, bh)),
        "mode": mode,
        "method": method,
        "candidates": [m for m, _s, _r in cands],
    }


def describe(res):
    """一行摘要，给日志用。"""
    if not res["ok"]:
        return "回退(框中心) 原因=%s" % (res["reason"],)
    cx, cy = res["point"]
    tc = res["tight_bbox"]
    return ("[%s/%s] 抓取点=(%.1f,%.1f) 贴合框=(%d,%d,%d,%d) 覆盖率=%.0f%% "
            "实心=%.2f 尺寸=%.1fx%.1f 角度=%.0f° 半径=%.1f"
            % (res["method"], res["mode"], cx, cy,
               tc[0], tc[1], tc[2], tc[3],
               res["coverage"] * 100, res["solidity"],
               res["rect_size"][0], res["rect_size"][1],
               res["angle"], res["radius"]))