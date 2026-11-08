#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定板角点标注 + 选点建议。

用法:
  python3 tools/mark_corners.py --image board.jpg --pick 10 25 39

输出: 角点表(序号/原始像素/棋盘mm) + 标注图 + 一组推荐的三点(最大分散度)。
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_calib as pc  # noqa: E402


def spread_triple(pts):
    """最分散的三点(最大化最短边 + 面积约束)。"""
    best, best_key = None, -1.0
    for a, b, c in itertools.combinations(range(len(pts)), 3):
        d = min(np.linalg.norm(pts[a] - pts[b]),
                np.linalg.norm(pts[b] - pts[c]),
                np.linalg.norm(pts[a] - pts[c]))
        area = 0.5 * abs(np.cross(pts[b] - pts[a], pts[c] - pts[a]))
        key = min(d, area ** 0.5 * 0.5)
        if key > best_key:
            best_key, best = key, (a, b, c)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="标定板角点标注与选点建议")
    ap.add_argument("--image", default="/root/vision_ws/board.jpg")
    ap.add_argument("--cols", type=int, default=pc.DEFAULT_COLS)
    ap.add_argument("--rows", type=int, default=pc.DEFAULT_ROWS)
    ap.add_argument("--square", type=float, default=24.0)
    ap.add_argument("--pick", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args(argv)

    info, pixels, obj, raw = pc.calibrate_from_image(
        args.image, cols=args.cols, rows=args.rows, square_mm=args.square,
        intrinsic_path=pc.find_default_intrinsic(), show=False)
    raw = np.asarray(raw, np.float64).reshape(-1, 2)
    print("[检测] %d 个角点  残差 平均 %.2fmm / 最大 %.2fmm"
          % (info["n_points"], info["mean_residual"], info["max_residual"]))
    print()
    print("%4s %9s %9s %9s %9s" % ("#", "u", "v", "board_x", "board_y"))
    for i, (uv, bm) in enumerate(zip(raw, obj)):
        print("%4d %9.2f %9.2f %9.1f %9.1f" % (i, uv[0], uv[1], bm[0], bm[1]))

    tri = spread_triple(raw)
    print()
    print("[推荐三点] 序号 %d / %d / %d  (最大化最短边)" % tri)
    for i in tri:
        print("    #%-3d 像素 (%.1f, %.1f)  棋盘 (%.0f, %.0f)mm"
              % (i, raw[i][0], raw[i][1], obj[i][0], obj[i][1]))
    d = [np.linalg.norm(raw[tri[0]] - raw[tri[1]]),
         np.linalg.norm(raw[tri[1]] - raw[tri[2]]),
         np.linalg.norm(raw[tri[0]] - raw[tri[2]])]
    area = 0.5 * abs(np.cross(raw[tri[1]] - raw[tri[0]], raw[tri[2]] - raw[tri[0]]))
    print("    像素边长 %.1f / %.1f / %.1f px, 面积 %.0f px^2"
          % (d[0], d[1], d[2], area))

    print()
    print("[备选组合] 面积>2500px^2, 按最短边排序")
    cand = []
    for a, b, c in itertools.combinations(range(len(raw)), 3):
        dd = min(np.linalg.norm(raw[a] - raw[b]),
                 np.linalg.norm(raw[b] - raw[c]),
                 np.linalg.norm(raw[a] - raw[c]))
        ar = 0.5 * abs(np.cross(raw[b] - raw[a], raw[c] - raw[a]))
        if ar > 2500.0:
            cand.append((dd, (a, b, c)))
    cand.sort(reverse=True)
    for dd, (a, b, c) in cand[:args.top]:
        print("    %2d / %2d / %2d   最短边 %.1fpx" % (a, b, c, dd))

    img = pc.imread_unicode(args.image)
    vis = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                     interpolation=cv2.INTER_LINEAR)
    for i, (u, v) in enumerate(raw):
        c = (int(round(u * args.scale)), int(round(v * args.scale)))
        cv2.circle(vis, c, 3, (0, 0, 255), -1)
        if i % 3 == 0 or i in args.pick or i in tri:
            cv2.putText(vis, str(i), (c[0] + 4, c[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(vis, str(i), (c[0] + 4, c[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)]
    for k, i in enumerate(args.pick):
        u, v = raw[i]
        c = (int(round(u * args.scale)), int(round(v * args.scale)))
        col = colors[k % len(colors)]
        cv2.drawMarker(vis, c, col, cv2.MARKER_CROSS, 34, 2)
        cv2.putText(vis, "#%d (%.0f,%.0f)" % (i, u, v),
                    (c[0] + 10, c[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(vis, "#%d (%.0f,%.0f)" % (i, u, v),
                    (c[0] + 10, c[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1)
    pts = np.array([raw[tri[0]], raw[tri[1]], raw[tri[2]]], np.float64) * args.scale
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [poly], True, (255, 255, 0), 1)
    out = args.out or os.path.splitext(args.image)[0] + "_marked.jpg"
    cv2.imwrite(out, vis)
    print()
    print("[图] %s  (%.1fx, %dx%d)" % (out, args.scale, vis.shape[1], vis.shape[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())