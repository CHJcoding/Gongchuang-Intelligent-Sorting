# -*- coding: utf-8 -*-
"""在棋盘照片上把 9x6 内角点全部编号标出来，并突出指定锚点（手眼 STEP 用）。

用途：现场核对「#10 到底是哪一个角」，避免把夹爪对到相邻的角上。

  python3 tools/mark_board.py --image /root/vision_ws/board.jpg \
          --out /root/vision_ws/board_marked.png --corners 10 16 37 43

只依赖 cv2 + numpy，不碰 ROS、不占相机。
"""
import argparse
import os
import sys

import cv2
import numpy as np

STEP_COLORS = [(0, 0, 230), (0, 140, 255), (200, 0, 200), (0, 180, 0)]


def detect_corners(grey, cols, rows):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, pts = cv2.findChessboardCorners(grey, (cols, rows), flags)
    if not ok or pts is None:
        return None
    return cv2.cornerSubPix(grey, pts, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                             30, 0.001))


def text(img, s, org, scale, color, thick=2):
    """带黑描边的文字，浅色背景上也读得清。"""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def dashed_line(img, p1, p2, color, thick=2, dash=12):
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-6:
        return
    n = max(2, int(dist / dash))
    for i in range(n):
        if i % 2:
            continue
        a = p1 + (p2 - p1) * (i / float(n))
        b = p1 + (p2 - p1) * (min(i + 1, n) / float(n))
        cv2.line(img, tuple(np.round(a).astype(int)),
                 tuple(np.round(b).astype(int)), color, thick, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description="标注棋盘内角点编号 + 手眼锚点")
    ap.add_argument("--image", default="/root/vision_ws/board.jpg")
    ap.add_argument("--out", default="")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=24.0, help="格子边长 mm（只用于算料盘坐标）")
    ap.add_argument("--corners", type=int, nargs="*", default=[10, 16, 37, 43],
                    help="要突出的锚点编号（按 STEP 顺序）")
    ap.add_argument("--scale", type=float, default=2.0, help="放大倍数（便于读数）")
    ap.add_argument("--csv", default="", help="把 54 个角点写成 csv")
    args = ap.parse_args(argv)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print("[mark] 读不到图：%s" % args.image)
        return 2
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = detect_corners(grey, args.cols, args.rows)
    if pts is None:
        print("[mark] 没检出 %dx%d 内角点" % (args.cols, args.rows))
        return 3
    uv = pts.reshape(-1, 2)
    n = len(uv)

    out = args.out or os.path.splitext(args.image)[0] + "_marked.png"
    s = max(1.0, float(args.scale))
    canvas = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    # ---- 全部角点：小圈 + 编号 ----
    for i, (u, v) in enumerate(uv):
        c = (int(round(u * s)), int(round(v * s)))
        cv2.circle(canvas, c, int(3 * s), (255, 160, 0), 1, cv2.LINE_AA)
        text(canvas, str(i), (c[0] + int(4 * s), c[1] - int(4 * s)),
             0.34 * s, (240, 240, 240), 1)

    # ---- 锚点：大方框 + STEP 标签 ----
    picked = [c for c in args.corners if 0 <= c < n]
    for k, idx in enumerate(picked):
        u, v = uv[idx]
        c = (int(round(u * s)), int(round(v * s)))
        col = STEP_COLORS[k % len(STEP_COLORS)]
        cv2.circle(canvas, c, int(4 * s), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, c, int(9 * s), col, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, c, col, cv2.MARKER_CROSS, int(22 * s), 2)
        text(canvas, "S%d=#%d" % (k + 1, idx),
             (c[0] + int(12 * s), c[1] + int(11 * s)), 0.55 * s, col, 2)
    for a, b in zip(picked, picked[1:]):
        pa = tuple(np.round(uv[a] * s).astype(int))
        pb = tuple(np.round(uv[b] * s).astype(int))
        dashed_line(canvas, pa, pb, (255, 255, 255), 2, int(14 * s))

    # ---- 顶部信息条 ----
    info = "raw px  (no undistort)   %dx%d grid   square=%.1fmm   %.3f mm/px   N=%d" % (
        args.cols, args.rows, args.square, args.square / 34.0 * 0.676 / 0.676, n)
    # 真实 mm/px 用锚点间距估（若锚点够）
    mmpx = None
    if len(picked) >= 2:
        a, b = picked[0], picked[1]
        dx = (b % args.cols) - (a % args.cols)
        dy = (b // args.cols) - (a // args.cols)
        span_cells = (dx * dx + dy * dy) ** 0.5
        px = np.linalg.norm(uv[b] - uv[a])
        if span_cells > 0:
            mmpx = (span_cells * args.square) / px
            info = "raw px (no undistort)  %dx%d grid  square=%.1fmm  measured %.3f mm/px  N=%d" % (
                args.cols, args.rows, args.square, mmpx, n)
    bar_h = int(30 * s)
    canvas = cv2.copyMakeBorder(canvas, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(32, 32, 32))
    text(canvas, info, (int(8 * s), int(21 * s)), 0.55 * s, (0, 255, 255), 2)

    ok = cv2.imwrite(out, canvas)
    print("[mark] 检出 %d 个角点，写出 %s (%s)" % (n, out, "ok" if ok else "失败"))
    if mmpx:
        print("[mark] 实测 %.4f mm/px（用 S1/S2 两点估）" % mmpx)
    print("[mark] STEP 顺序：")
    for k, idx in enumerate(picked):
        u, v = uv[idx]
        r, c = divmod(idx, args.cols)
        print("       S%d  #%-2d  pixel=(%7.2f,%7.2f)  料盘=(%5.1f,%5.1f)mm  (r%d,c%d)"
              % (k + 1, idx, u, v, c * args.square, r * args.square, r, c))

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("idx,u,v,x_mm,y_mm\n")
            for i, (u, v) in enumerate(uv):
                r, c = divmod(i, args.cols)
                fh.write("%d,%.3f,%.3f,%.2f,%.2f\n"
                         % (i, u, v, c * args.square, r * args.square))
        print("[mark] 角点表 -> %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())