# -*- coding: utf-8 -*-
"""把「YOLO 框」和「贴合物料的外接框」画在同一张图上，肉眼对比差多少。

这是回答"框能不能更贴合"的对比工具：同一帧、同一批 YOLO 框，
用现有的颜色掩膜算出三种贴合框，叠在一起看。

三种框的含义：
    红框  YOLO 原始框（网络回归出来的，带量化误差 + 阴影外扩）
    绿框  掩膜的最小外接正矩形 cv2.boundingRect      —— 轴对齐，边长贴住物体
    黄框  掩膜的最小外接旋转矩形 cv2.minAreaRect     —— 带角度，真正贴住斜放的物料
    青点  精修后的抓取点（质心 / 旋转矩形中心，按 config 的 GRASP_POINT_MODE）

用法：
    python tools/tight_box_demo.py --image _scan/frame.jpg --boxes _scan/boxes.json
    python tools/tight_box_demo.py --image x.jpg --out x_boxes.png
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GC = os.path.dirname(_HERE)
for _p in (os.path.join(_GC, "rdk", "vision_sorter"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import grasp_refine as GR                      # noqa: E402
from color_classifier import classify_color    # noqa: E402


def _load_boxes(path):
    """支持两种：服务器吐的 [{type,conf,box:[x,y,w,h]}]，或 [x1,y1,x2,y2]。"""
    data = json.load(io.open(path, encoding="utf-8"))
    out = []
    for it in data:
        if isinstance(it, dict) and "box" in it:
            x, y, w, h = it["box"]
            out.append({"shape": it.get("type", "?"), "conf": it.get("conf", 0.0),
                        "bbox": (int(x), int(y), int(x + w), int(y + h))})
        else:
            x1, y1, x2, y2 = it
            out.append({"shape": "?", "conf": 0.0, "bbox": (int(x1), int(y1),
                                                           int(x2), int(y2))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO 框 vs 贴合框 可视化对比")
    ap.add_argument("--image", default=os.path.join(_GC, "_scan", "frame.jpg"))
    ap.add_argument("--boxes", default=os.path.join(_GC, "_scan", "boxes.json"))
    ap.add_argument("--out", default=os.path.join(_GC, "_scan", "tight_box_demo.jpg"))
    ap.add_argument("--point-mode", default=None,
                    choices=["centroid", "rect", "bbox"])
    ap.add_argument("--scale", type=float, default=2.0, help="放大倍数方便看清")
    ap.add_argument("--style", default="compare", choices=["compare", "annotate"],
                    help="compare=三框对比图；annotate=用生产代码 annotate.py 画"
                         "（与板端预览看到的完全一致）")
    a = ap.parse_args(argv)

    img = cv2.imread(a.image)
    if img is None:
        print("读不到图片:", a.image)
        return 1
    boxes = _load_boxes(a.boxes) if os.path.exists(a.boxes) else []
    if not boxes:
        print("没有框可画（给了 --boxes 但文件空/不存在）")
        return 1

    canvas = img.copy()
    items = []
    print("%-6s %-8s %-22s %-22s %-22s %s"
          % ("形状", "颜色", "YOLO 框", "贴合正框(boundingRect)", "旋转框(minAreaRect)", "抓取点"))
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        roi = img[max(0, y1):y2, max(0, x1):x2]
        color, cconf = (classify_color(roi) if roi.size else (None, 0.0))
        if color is None:
            color = "?"
        items.append({"shape": b["shape"], "conf": b["conf"], "bbox": b["bbox"],
                      "color": color})
        res = GR.refine(img, b["bbox"], color, point_mode=a.point_mode)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)      # 红：YOLO
        ok = bool(res.get("ok"))
        if ok:
            tb = res.get("tight_bbox")
            if tb:
                tx1, ty1, tx2, ty2 = [int(round(v)) for v in tb]
                cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (0, 200, 0), 2)  # 绿
            rc, rs = res.get("rect_center"), res.get("rect_size")
            if rc and rs:
                rot = (tuple(float(v) for v in rc), tuple(float(v) for v in rs),
                       float(res.get("angle", 0.0)))
                pts = cv2.boxPoints(rot).astype(np.int32)
                cv2.polylines(canvas, [pts], True, (0, 220, 255), 2)       # 黄
        gx = res.get("point")
        if gx:
            cv2.drawMarker(canvas, (int(round(gx[0])), int(round(gx[1]))),
                           (255, 255, 0), cv2.MARKER_CROSS, 22, 2)     # 青
        cv2.putText(canvas, "%s/%s %.2f" % (b["shape"], color, b["conf"]),
                    (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 3)
        cv2.putText(canvas, "%s/%s %.2f" % (b["shape"], color, b["conf"]),
                    (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
        print("%-6s %-8s %-22s %-22s %-22s %s"
              % (b["shape"], color, b["bbox"],
                 res.get("tight_bbox") if ok else "回退",
                 (res.get("rect_size"), round(res.get("angle", 0.0), 1)) if ok else "-",
                 tuple(int(round(v)) for v in gx) if gx else "-"))

    if a.style == "annotate":
        import annotate
        canvas = annotate.draw_targets(img, items, point_mode=a.point_mode)
    if a.scale and a.scale > 1:
        canvas = cv2.resize(canvas, None, fx=a.scale, fy=a.scale,
                            interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(a.out, canvas)
    print("\n已保存:", a.out)
    if a.style == "annotate":
        print("生产画法：黄=贴合旋转框  青=抓取点  红虚线=回退(用框中心)")
    else:
        print("红=YOLO框  绿=贴合正框  黄=贴合旋转框  青=抓取点")
    return 0


if __name__ == "__main__":
    sys.exit(main())