#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取点可视化标注 —— 纯 cv2/numpy，不依赖 ROS，可离线单测。

这个模块只做一件事：把「贴合正框（可旋转）+ 抓取点」画出来。

画三样东西：
  1. **贴合旋转框**：用颜色掩膜算 minAreaRect（斜放物料也贴着边），黄线；
  2. **抓取点**：grasp_refine 给的落点（默认就是旋转框中心），青色圆 + 十字；
  3. **标签**：`形状/颜色 编号=NN`，贴在旋转框上方。

为什么还要画回退框：grasp_refine 算不出可信掩膜时（颜色糊到背景、掩膜只覆盖
半个物体、物体出画），它会**故意回退**到 YOLO 框中心 —— 那种情况必须一眼看出
来，否则现场会误以为是精修失效。回退时画红虚线框 + 标 "回退"。

坐标都是原始 640x480 像素，画完不缩放。
"""

import os
import time

import cv2
import numpy as np

try:
    import config as C
    from color_classifier import classify_color
    import grasp_refine
except Exception:                      # 允许单独 import 做语法/离线检查
    C = None
    classify_color = None
    grasp_refine = None


# 颜色（BGR）
COL_BOX = (0, 215, 255)      # 黄：贴合旋转框
COL_POINT = (255, 255, 0)    # 青：抓取点
COL_FALLBACK = (0, 0, 255)   # 红：回退（不可信）
COL_TEXT = (255, 255, 255)
COL_SHADOW = (0, 0, 0)


def _put(img, text, org, scale=0.45, color=COL_TEXT, bold=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COL_SHADOW, bold + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, bold, cv2.LINE_AA)


def _draw_rotated(img, rect, color, thickness=2):
    """rect = ((cx,cy),(w,h),angle)"""
    pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def _draw_point(img, pt, color=COL_POINT, r=5, thickness=2):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(img, (x, y), r, color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - r - 4, y), (x + r + 4, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - r - 4), (x, y + r + 4), color, 1, cv2.LINE_AA)


def _label(shape, color, gid):
    """一行标签：形状/颜色 编号=N（编号为 None 时只显示前两项）。"""
    base = "%s/%s" % (shape, color)
    if gid is None:
        return base
    return "%s #%d" % (base, int(gid))


def _text_origin(rect, lines, img_shape):
    """把标签放在旋转框上方；贴到画面顶部时翻到框下方。"""
    cx, cy = rect[0]
    w, h = rect[1]
    top = cy - max(w, h) / 2.0
    y = int(max(14, top - 6 - 14 * (lines - 1)))
    if y < 14:                                  # 顶到边了，改放框内底部
        y = int(min(img_shape[0] - 6, cy + max(w, h) / 2.0 + 16))
    x = int(max(2, cx - 60))
    return x, y


def draw_targets(frame, targets, point_mode=None, thickness=2, copy=True):
    """把每个目标画成「贴合旋转框 + 抓取点」。

    参数
    ----
    frame : np.ndarray BGR（会被复制，调用方的原图不动 —— 精修还要用它）
    targets : list[dict]，每项至少 {shape, bbox, conf?}
              bbox = [x1, y1, x2, y2]（原始像素）
    point_mode : None=用 config.GRASP_POINT_MODE；也可 "centroid"/"rect"/"bbox"
    thickness : 线宽

    返回标注后的新图。
    """
    img = frame.copy() if copy else frame
    if not targets:
        return img
    if point_mode is None:
        point_mode = getattr(C, "GRASP_POINT_MODE", "centroid") if C else "centroid"

    for t in targets:
        x1, y1, x2, y2 = [int(round(v)) for v in t["bbox"]]
        shape = t.get("shape", "?")
        color = t.get("color")
        gid = t.get("gid", t.get("material_id"))

        res = None
        if grasp_refine is not None and getattr(C, "GRASP_REFINE", True):
            try:
                if color is None and classify_color is not None:
                    color = classify_color(frame, [x1, y1, x2, y2])[0]
                res = grasp_refine.refine(frame, [x1, y1, x2, y2], color,
                                          point_mode=point_mode)
            except Exception:
                res = None

        ok = bool(res and res.get("ok"))
        if ok:
            rect = (tuple(float(v) for v in res["rect_center"]),
                    tuple(float(v) for v in res["rect_size"]),
                    float(res.get("angle", 0.0)))
            _draw_rotated(img, rect, COL_BOX, thickness)
            pt = res.get("point") or res["rect_center"]
            _draw_point(img, pt, COL_POINT, 5, thickness)
            roi = res.get("mask_px") or 0
            if getattr(C, "ANNOTATE_VERBOSE", False) and res.get("tight_bbox"):
                tx1, ty1, tx2, ty2 = [int(round(v)) for v in res["tight_bbox"]]
                cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 200, 0), 1)
            txt = _label(shape, color, gid)
            if getattr(C, "ANNOTATE_VERBOSE", False):
                w, h = rect[1]
                txt += " %.0fx%.0f %d\xb0" % (w, h, rect[2])
            pos = _text_origin(rect, 1, img.shape)
            _put(img, txt, pos)
            if roi and getattr(C, "ANNOTATE_VERBOSE", False):
                _put(img, "a=%d" % int(roi), (pos[0], pos[1] + 13), 0.35)
        else:
            # 回退：画 YOLO 框（红虚线）+ 说明，别让人以为精修 100% 成功
            _dashed_rect(img, (x1, y1), (x2, y2), COL_FALLBACK, thickness)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            _draw_point(img, (cx, cy), COL_FALLBACK, 4, 1)
            reason = (res or {}).get("reason") or "no-mask"
            _put(img, "%s 回退:%s" % (_label(shape, color, gid), reason),
                 (x1, max(12, y1 - 6)), 0.42, COL_FALLBACK)
    return img


def _dashed_rect(img, p1, p2, color, thickness=2, dash=6):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def save_jpeg(path, img, quality=80):
    """原子写 JPEG：先编码写到 `path + '.tmp'` 再 rename。

    注意用 `cv2.imencode` + `tofile` 而不是 `cv2.imwrite`：临时文件名不带 .jpg 后缀
    （带的话会被看图程序的 `*.jpg` 匹配到半成品），而 imwrite 靠后缀猜编码器，会报
    `could not find a writer for the specified extension`。
    """
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError("JPEG 编码失败")
    buf.tofile(tmp)
    os.replace(tmp, path)
    return path


def save_with_age(path, img, quality=80):
    """写图并返回时间戳（调用方可用它做限频）。"""
    save_jpeg(path, img, quality)
    return time.time()


if __name__ == "__main__":                 # 极简离线自检：造两个方块画一画
    f = np.full((480, 640, 3), 40, np.uint8)
    cv2.rectangle(f, (200, 150), (280, 230), (0, 180, 0), -1)      # 绿块
    ts = [{"shape": "SLZ", "conf": 0.9, "bbox": [195, 145, 285, 235],
           "color": "green", "gid": 33}]
    out = draw_targets(f, ts)
    p = os.path.join(os.getcwd(), "annotate_selftest.jpg")
    save_jpeg(p, out)
    print("saved", p)