#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用「三点龙门真值」反过来找它是棋盘上哪三个角点。

标定板挪位置后，人按旧图点号量出来的 (X,Y) 常对不上号。这个工具不猜序号：
直接遍历 54 个角点的所有三点组合，做相似变换拟合（允许镜像、比例自由），
按残差排序，谁最像就报谁 —— 顺便给出比例(应≈1.00 mm/mm)和旋转角。

用法:
  python3 tools/match_anchors.py --image board.jpg --xy 15.1 162 --xy 187 127 --xy 71.7 75
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_calib as pc  # noqa: E402


def similarity(src, dst):
    """2D 相似变换 src->dst（复数最小二乘），返回 (比例, 旋转角, 镜像, 残差mm)。"""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    S, D = src - cs, dst - cd
    z = S[:, 0] + 1j * S[:, 1]
    best = None
    for mirror in (False, True):
        w = D[:, 0] + 1j * D[:, 1]
        if mirror:
            w = np.conj(w)
        den = float(np.sum(np.abs(w) ** 2))
        if den <= 1e-12:
            continue
        rot = complex(np.sum(z * w) / den)
        resid = rot * w - z
        r = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
        if best is None or r < best[3]:
            best = (abs(rot), float(np.degrees(np.angle(rot))), mirror, r)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="三点真值反查角点序号")
    ap.add_argument("--image", default="/root/vision_ws/board.jpg")
    ap.add_argument("--cols", type=int, default=pc.DEFAULT_COLS)
    ap.add_argument("--rows", type=int, default=pc.DEFAULT_ROWS)
    ap.add_argument("--square", type=float, default=24.0)
    ap.add_argument("--xy", nargs=2, type=float, action="append", required=True)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    info, pixels, obj, raw = pc.calibrate_from_image(
        args.image, cols=args.cols, rows=args.rows, square_mm=args.square,
        intrinsic_path=pc.find_default_intrinsic(), show=False)
    obj = np.asarray(obj, np.float64).reshape(-1, 2)
    raw = np.asarray(raw, np.float64).reshape(-1, 2)
    target = np.asarray(args.xy, np.float64)
    print("[检测] %d 角点 | 整板单应残差 平均 %.2fmm 最大 %.2fmm"
          % (info["n_points"], info["mean_residual"], info["max_residual"]))
    print("[真值] %s mm" % np.round(target, 2).tolist())
    dt = [np.linalg.norm(target[i] - target[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
    print("[真值] 三点边长 %.1f / %.1f / %.1f mm" % tuple(dt))

    cand = []
    for tri in itertools.combinations(range(len(obj)), 3):
        scale, ang, mirror, r = similarity(target, obj[list(tri)])
        cand.append((r, scale, ang, mirror, tri))
    cand.sort(key=lambda c: c[0])
    print()
    print("%-16s %8s %7s %7s %4s" % ("角点组合", "残差mm", "比例", "旋转", "镜像"))
    for r, scale, ang, mirror, tri in cand[:args.top]:
        print("%-16s %8.2f %7.3f %7.1f %4s"
              % ("/".join(str(t) for t in tri), r, scale, ang,
                 "是" if mirror else "否"))
    print()
    r, scale, ang, mirror, tri = cand[0]
    print("[最佳] 角点 %s  残差 %.2fmm 比例 %.3f 旋转 %.1f° 镜像 %s"
          % ("/".join(str(t) for t in tri), r, scale, ang,
             "是" if mirror else "否"))
    for k, i in enumerate(tri):
        print("   #%-3d 棋盘 (%.0f, %.0f)mm  像素 (%.1f, %.1f)  真值 (%s)"
              % (i, obj[i][0], obj[i][1], raw[i][0], raw[i][1],
                 np.round(target[k], 1).tolist()))
    print()
    print("[提示] 比例应接近 1.000（棋盘 mm 与龙门 mm 同量纲）；残差 <3mm 才算对上号。")
    return 0


if __name__ == "__main__":
    sys.exit(main())