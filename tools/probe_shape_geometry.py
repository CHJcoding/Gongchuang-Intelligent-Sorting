#!/usr/bin/env python3

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = "/root/vision_ws"
sys.path.insert(0, os.path.join(ROOT, "vision_sorter"))

import config as C
import grasp_refine
from color_classifier import classify_color


CLASSES = {"cylinder": 5, "cube": 7}


def load_boxes(path, w, h):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            if cls not in CLASSES.values():
                continue
            cx, cy, bw, bh = map(float, p[1:5])
            boxes.append((cls, [
                (cx - bw / 2) * w, (cy - bh / 2) * h,
                (cx + bw / 2) * w, (cy + bh / 2) * h,
            ]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="/tmp/val_imgs")
    ap.add_argument("--labels-dir", default="/tmp/val_lbls")
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.images_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(args.images_dir, name))
        if image is None:
            continue
        h, w = image.shape[:2]
        stem = os.path.splitext(name)[0]
        for cls, bbox in load_boxes(
                os.path.join(args.labels_dir, stem + ".txt"), w, h):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            color, cconf = classify_color(crop)
            if color is None:
                continue
            gp = grasp_refine.refine(image, bbox, color)
            if not gp["ok"]:
                continue
            rw, rh = gp["rect_size"]
            rect_area = max(1.0, rw * rh)
            rect_fill = gp["mask_px"] / rect_area
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            rows.append((cls, name, color, cconf, gp["coverage"],
                         gp["solidity"], rect_fill, aspect))

    for cls, label in ((5, "cylinder"), (7, "cube")):
        data = [r for r in rows if r[0] == cls]
        print("\n%s count=%d" % (label, len(data)))
        for r in data:
            print("  %s color=%s cconf=%.2f coverage=%.3f solidity=%.3f "
                  "rect_fill=%.3f aspect=%.3f"
                  % (r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
        if data:
            arr = np.asarray([[r[4], r[5], r[6], r[7]] for r in data])
            print("  median coverage/solidity/rect_fill/aspect: %s"
                  % np.round(np.median(arr, axis=0), 3))
            print("  rect_fill p10/p50/p90: %s"
                  % np.round(np.percentile(arr[:, 2], [10, 50, 90]), 3))


if __name__ == "__main__":
    main()
