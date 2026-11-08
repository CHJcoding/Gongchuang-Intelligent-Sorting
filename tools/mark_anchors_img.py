#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在标定板照片上标出锚点/探针点（带龙门坐标）。

用法:
  python3 tools/mark_anchors_img.py --image board.jpg \
      --anchor 43 15.1 162 --anchor 28 187 127 --anchor 14 71.7 75 \
      --probe 0 205.4 47.5 --probe 49 103.1 187.8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_calib as pc  # noqa: E402

import cv2
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
]


def load_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), True
            except Exception:
                continue
    return ImageFont.load_default(), False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/root/vision_ws/board.jpg")
    ap.add_argument("--anchor", nargs=3, type=float, action="append", default=[])
    ap.add_argument("--probe", nargs=3, type=float, action="append", default=[])
    ap.add_argument("--out", default="/root/vision_ws/board_anchors_marked.jpg")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--square", type=float, default=24.0)
    args = ap.parse_args(argv)

    info, pixels, obj, raw = pc.calibrate_from_image(
        args.image, cols=pc.DEFAULT_COLS, rows=pc.DEFAULT_ROWS,
        square_mm=args.square, intrinsic_path=pc.find_default_intrinsic(),
        show=False)
    raw = np.asarray(raw, np.float64).reshape(-1, 2)
    obj = np.asarray(obj, np.float64).reshape(-1, 2)

    img = pc.imread_unicode(args.image)
    vis = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                     interpolation=cv2.INTER_LINEAR)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(vis)
    dr = ImageDraw.Draw(im)
    font_l, cjk_l = load_font(int(15 * args.scale / 2))
    font_s, cjk_s = load_font(int(11 * args.scale / 2))
    S = args.scale

    anchors = [(int(a[0]), a[1], a[2]) for a in args.anchor]
    probes = [(int(p[0]), p[1], p[2]) for p in args.probe]
    pick_ids = set([a[0] for a in anchors] + [p[0] for p in probes])

    for i, (u, v) in enumerate(raw):
        c = (u * S, v * S)
        r = 3 if i in pick_ids else 2
        dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(255, 0, 0))
        if i % 2 == 0 or i in pick_ids:
            dr.text((c[0] + 5, c[1] - 12), str(i), font=font_s,
                    fill=(255, 230, 0), stroke_width=2, stroke_fill=(0, 0, 0))

    if len(anchors) >= 2:
        pts = [(raw[i][0] * S, raw[i][1] * S) for i, _, _ in anchors]
        dr.polygon(pts, outline=(0, 200, 255))

    for k, (idx, X, Y) in enumerate(anchors):
        u, v = raw[idx] * S
        col = [(0, 255, 0), (255, 140, 0), (255, 0, 255)][k % 3]
        dr.line([(u - 20, v), (u + 20, v)], fill=col, width=2)
        dr.line([(u, v - 20), (u, v + 20)], fill=col, width=2)
        txt = ('锚点 #%d  龙门(%g, %g)' % (idx, X, Y)) if cjk_l else \
              ('ANCHOR #%d  (%g, %g)' % (idx, X, Y))
        dr.text((u + 12, v + 8), txt, font=font_l, fill=col,
                stroke_width=3, stroke_fill=(0, 0, 0))

    for k, (idx, X, Y) in enumerate(probes):
        u, v = raw[idx] * S
        col = (0, 255, 255)
        dr.ellipse([u - 14, v - 14, u + 14, v + 14], outline=col, width=2)
        txt = ('验收 #%d  应到(%g, %g)' % (idx, X, Y)) if cjk_l else \
              ('PROBE #%d  (%g, %g)' % (idx, X, Y))
        dr.text((u + 12, v + 8), txt, font=font_l, fill=col,
                stroke_width=3, stroke_fill=(0, 0, 0))

    out = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, out)
    print('[图] %s (%dx%d) 中文字体 %s' % (args.out, out.shape[1], out.shape[0],
                                          "已用" if cjk_l else "缺失-用英文"))
    for idx, X, Y in anchors:
        print('  锚点 #%-2d 像素 (%.1f, %.1f) 龙门 (%g, %g)'
              % (idx, raw[idx][0], raw[idx][1], X, Y))
    for idx, X, Y in probes:
        print('  验收 #%-2d 像素 (%.1f, %.1f) 应到 (%g, %g)'
              % (idx, raw[idx][0], raw[idx][1], X, Y))
    return 0


if __name__ == "__main__":
    sys.exit(main())