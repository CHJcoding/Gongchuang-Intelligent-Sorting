# -*- coding: utf-8 -*-
"""验证 Python 解码是否正确：拿训练集的图 + 标注（YOLO txt）跑一遍，比对框。

如果解码/坐标映射正确，检测框和标注框应该高度重合（IoU 高、中心差几个像素）。
如果 IoU 普遍很低或整体偏移，说明 letterbox/DFL/stride 有一处错了。

板端用法：
    python3 tools/validate_decode.py --images-dir /tmp/val_imgs --labels-dir /tmp/val_lbls \
        [--conf 0.05] [--bin ...] [--labels ...] [--limit 5]
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _c in (os.path.join(ROOT, "rdk", "vision_sorter"), os.path.join(ROOT, "vision_sorter")):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)

from yolov8_bpu import Yolov8Bpu  # noqa: E402


def load_gt(label_path, w, h):
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="/tmp/val_imgs")
    ap.add_argument("--labels-dir", default="/tmp/val_lbls")
    ap.add_argument("--bin", default="/root/vision_ws/gc_best_1029.bin")
    ap.add_argument("--labels", default="/root/vision_ws/gc_1029_labels.list")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    det = Yolov8Bpu(args.bin, names, conf_thres=args.conf, iou_thres=args.iou)

    files = sorted(fn for fn in os.listdir(args.images_dir)
                   if fn.lower().endswith((".jpg", ".png", ".jpeg")))
    if args.limit:
        files = files[:args.limit]

    n_gt = n_hit = 0
    ious, score_hits = [], []
    for fn in files:
        img = cv2.imread(os.path.join(args.images_dir, fn))
        if img is None:
            print("读图失败", fn)
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(fn)[0]
        gt = load_gt(os.path.join(args.labels_dir, stem + ".txt"), w, h)
        dets = det.detect(img)
        line = ["%-12s GT=%2d DET=%2d" % (fn, len(gt), len(dets))]
        for (c, gx1, gy1, gx2, gy2) in gt:
            best_iou, best_s, best_c = 0.0, 0.0, -1
            for d in dets:
                v = iou((gx1, gy1, gx2, gy2), (d.x1, d.y1, d.x2, d.y2))
                if v > best_iou:
                    best_iou, best_s, best_c = v, d.score, d.cls_id
            n_gt += 1
            if best_iou >= 0.5 and best_c == c:
                n_hit += 1
                ious.append(best_iou)
                score_hits.append(best_s)
            else:
                line.append("  miss cls=%d(%s) bestIoU=%.2f bestCls=%s s=%.2f"
                            % (c, names[c], best_iou, names[best_c] if best_c >= 0 else "-", best_s))
        print("\n".join(line))

    print("\n=== 汇总 ===")
    print("标注框 %d 个，命中（IoU>=0.5 且类别对）%d 个 → 召回 %.1f%%"
          % (n_gt, n_hit, 100.0 * n_hit / max(1, n_gt)))
    if ious:
        print("命中框 IoU: 均值 %.3f 最小 %.3f" % (float(np.mean(ious)), float(np.min(ious))))
        print("命中框 分数: 均值 %.3f" % float(np.mean(score_hits)))


if __name__ == "__main__":
    main()