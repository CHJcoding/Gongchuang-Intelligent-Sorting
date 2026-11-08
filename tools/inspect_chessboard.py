#!/usr/bin/env python3
"""Read-only checks for a hand-eye checkerboard photo.

This is a diagnostic helper, not a replacement for planar_calib.py. It reports
the image bounds and large four-sided black regions so a clipped or rotated
board can be identified before spending time on a full calibration.
"""

import argparse
import json

import cv2
import numpy as np


def parse_shape(value):
    cols, rows = value.lower().split("x", 1)
    return int(cols), int(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--classic", action="store_true",
                        help="also try the classic detector on normal/inverted images")
    args = parser.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("cannot read image: %s" % args.image)

    shapes = args.shape or [(9, 6), (8, 5), (11, 8)]
    checks = []
    for shape in shapes:
        ok, corners = cv2.findChessboardCornersSB(gray, shape, flags=0)
        checks.append({
            "detector": "SB",
            "shape": list(shape),
            "found": bool(ok),
            "count": 0 if corners is None else int(len(corners)),
        })
    if args.classic:
        equalized = cv2.equalizeHist(gray)
        variants = (
            ("classic", gray),
            ("classic-equalized", equalized),
            ("classic-inverted", 255 - gray),
        )
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        for shape in shapes:
            for name, image in variants:
                ok, corners = cv2.findChessboardCorners(image, shape, flags=flags)
                checks.append({
                    "detector": name,
                    "shape": list(shape),
                    "found": bool(ok),
                    "count": 0 if corners is None else int(len(corners)),
                })

    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 150.0 < area < 5000.0:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not 0.35 < w / max(h, 1) < 2.8:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        quads.append({
            "center": [round(cx, 1), round(cy, 1)],
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": round(area, 1),
        })

    quads.sort(key=lambda item: (round(item["center"][1] / 12.0),
                                 item["center"][0]))
    result = {
        "image": args.image,
        "size": [int(gray.shape[1]), int(gray.shape[0])],
        "chessboard_checks": checks,
        "black_quad_count": len(quads),
        "black_quads": quads,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("image=%s size=%dx%d" % (
        args.image, gray.shape[1], gray.shape[0]))
    for check in checks:
        print("chessboard %-16s %dx%d found=%s count=%d" % (
            check["detector"], check["shape"][0], check["shape"][1],
            check["found"], check["count"]))
    print("four-sided black regions: %d" % len(quads))
    for index, quad in enumerate(quads):
        print("%3d center=(%6.1f,%6.1f) bbox=%s area=%.1f" % (
            index, quad["center"][0], quad["center"][1],
            tuple(quad["bbox"]), quad["area"]))


if __name__ == "__main__":
    main()
