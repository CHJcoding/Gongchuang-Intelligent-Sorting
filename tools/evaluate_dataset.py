#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the deployed YOLO/HSV pipeline against the board capture dataset.

The capture folders use the short training labels while config.SHAPE_NAMES uses
the model labels.  This tool runs on the RDK X5 because hobot_dnn is required.
It never opens the serial port or publishes sorting commands.
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, "vision_sorter"))

import config as C  # noqa: E402
from color_classifier import classify_color, color_diag  # noqa: E402
from grasp_refine import refine  # noqa: E402
from yolov8_bpu import Yolov8Bpu  # noqa: E402


LABEL_TO_SHAPE = {
    "LK": "hollow_cube",
    "SMT": "tetrahedron",
    "WLZ": "pentagonal_prism",
    "SLZ": "triangular_prism",
    "YZHU": "cylinder",
    "YZ": "cone",
    "BQ": "hemisphere",
    "CUBE": "cube",
}


def evenly_sample(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return paths
    step = len(paths) / float(limit)
    return [paths[min(len(paths) - 1, int(i * step))] for i in range(limit)]


def roi_for(frame, det):
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(round(det.x1))))
    y1 = max(0, min(h - 1, int(round(det.y1))))
    x2 = max(x1 + 1, min(w, int(round(det.x2))))
    y2 = max(y1 + 1, min(h, int(round(det.y2))))
    return frame[y1:y2, x1:x2], [x1, y1, x2, y2]


def summarize(rows, thresholds):
    result = {}
    for threshold in thresholds:
        by_class = collections.defaultdict(lambda: [0, 0])
        color_votes = collections.Counter()
        color_ok = 0
        refine_ok = 0
        detected = 0
        fp = 0
        negatives = 0
        for row in rows:
            truth = row["truth_shape"]
            top = row["top"]
            color = row["color"]
            if truth is None:
                negatives += 1
                if top is not None and top["score"] >= threshold:
                    fp += 1
                continue
            acc = by_class[truth]
            acc[0] += 1
            if top is not None and top["score"] >= threshold:
                detected += 1
                if top["shape"] == truth:
                    acc[1] += 1
                if color is not None:
                    color_votes[color] += 1
                    color_ok += 1
                if row["refine_ok"]:
                    refine_ok += 1
        classes = {}
        total = 0
        hits = 0
        for name, (n, ok) in sorted(by_class.items()):
            classes[name] = {"n": n, "hits": ok,
                             "acc": round(ok / float(n), 4) if n else None}
            total += n
            hits += ok
        result[str(threshold)] = {
            "shape_hits": hits,
            "shape_total": total,
            "shape_acc": round(hits / float(total), 4) if total else None,
            "detected": detected,
            "color_ok": color_ok,
            "color_votes": dict(color_votes),
            "refine_ok": refine_ok,
            "negative_images": negatives,
            "negative_false_positives": fp,
            "negative_fp_rate": round(fp / float(negatives), 4) if negatives else None,
            "classes": classes,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(
        WS, "dataset", "20261031_1126", "raw"))
    parser.add_argument("--limit", type=int, default=25,
                        help="images per class; 0 means all")
    parser.add_argument("--thresholds", default="0.25,0.40,0.60,0.80")
    parser.add_argument("--json-out", default=os.path.join(
        WS, "dataset_eval.json"))
    args = parser.parse_args(argv)

    thresholds = sorted({float(v) for v in args.thresholds.split(",") if v})
    detector = Yolov8Bpu(C.YOLO_BIN_MODEL, C.SHAPE_NAMES,
                         conf_thres=0.01,
                         iou_thres=C.YOLO_IOU_THRES,
                         verbose=True)
    rows = []
    folders = sorted([
        d for d in os.listdir(args.dataset)
        if os.path.isdir(os.path.join(args.dataset, d))
    ])
    for folder in folders:
        truth = LABEL_TO_SHAPE.get(folder)
        if truth is None and folder != "negative":
            continue
        paths = sorted([
            os.path.join(args.dataset, folder, name)
            for name in os.listdir(os.path.join(args.dataset, folder))
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        paths = evenly_sample(paths, args.limit)
        for path in paths:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            dets = detector.detect(frame, conf_thres=0.01)
            dets.sort(key=lambda d: -d.score)
            top = None
            color = None
            color_conf = 0.0
            refine_ok = False
            color_text = ""
            if dets:
                d = dets[0]
                top = {"shape": d.name, "score": round(float(d.score), 4),
                       "bbox": [round(float(d.x1), 1), round(float(d.y1), 1),
                                round(float(d.x2), 1), round(float(d.y2), 1)]}
                roi, bbox = roi_for(frame, d)
                if roi.size:
                    color, color_conf = classify_color(roi)
                    color_text = color_diag(roi)
                    if color:
                        gp = refine(frame, bbox, color)
                        refine_ok = bool(gp.get("ok"))
            rows.append({
                "file": os.path.relpath(path, args.dataset),
                "folder": folder,
                "truth_shape": truth,
                "n_det": len(dets),
                "top": top,
                "color": color,
                "color_conf": color_conf,
                "color_diag": color_text,
                "refine_ok": refine_ok,
            })
            print("%-42s truth=%-18s top=%-18s score=%-6s color=%s" % (
                os.path.relpath(path, args.dataset), truth or "negative",
                top["shape"] if top else "-",
                "%.3f" % top["score"] if top else "-",
                color or "-"))

    summary = {
        "dataset": args.dataset,
        "images": len(rows),
        "limit_per_class": args.limit,
        "shape_names": C.SHAPE_NAMES,
        "active_colors": sorted(getattr(C, "COLOR_ALLOWED", [])),
        "thresholds": summarize(rows, thresholds),
        "rows": rows,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    for threshold in thresholds:
        s = summary["thresholds"][str(threshold)]
        print("threshold %.2f  shape %d/%d (%.1f%%)  color %d  refine %d"
              "  negative FP %d/%d (%.1f%%)" % (
                  threshold, s["shape_hits"], s["shape_total"],
                  100.0 * (s["shape_acc"] or 0.0), s["color_ok"],
                  s["refine_ok"], s["negative_false_positives"],
                  s["negative_images"],
                  100.0 * (s["negative_fp_rate"] or 0.0)))
        for name, v in s["classes"].items():
            print("  %-18s %3d/%3d  %.1f%%" % (
                name, v["hits"], v["n"], 100.0 * (v["acc"] or 0.0)))
    print("json:", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
