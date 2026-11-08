#!/usr/bin/env python3
"""Evaluate one YOLOv8 BPU bin on class-named capture folders."""

import argparse
import collections
import json
import os

import cv2

from verify_model_deploy import load_labels


LABEL_TO_SHAPE = {
    "LK": "hollow_cube",
    "SMT": "tetrahedron",
    "WLZ": "pentagonal_prism",
    "SLZ": "triangular_prism",
    "YZHU": "cone",
    "YZ": "cylinder",
    "BQ": "hemisphere",
    "CUBE": "cube",
}


def sample(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return paths
    step = len(paths) / float(limit)
    return [paths[min(len(paths) - 1, int(i * step))] for i in range(limit)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--thresholds", default="0.25,0.45,0.60")
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    from yolov8_bpu import Yolov8Bpu

    detector = Yolov8Bpu(
        args.bin,
        labels,
        conf_thres=0.01,
        iou_thres=0.45,
        verbose=True,
    )
    rows = []
    for folder in sorted(os.listdir(args.dataset)):
        folder_path = os.path.join(args.dataset, folder)
        if not os.path.isdir(folder_path):
            continue
        truth = LABEL_TO_SHAPE.get(folder)
        if truth is None and folder != "negative":
            continue
        paths = sorted(
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for path in sample(paths, args.limit):
            image = cv2.imread(path)
            if image is None:
                continue
            detections = detector.detect(image, conf_thres=0.01)
            detections.sort(key=lambda item: -item.score)
            top = detections[0] if detections else None
            rows.append(
                {
                    "folder": folder,
                    "truth": truth,
                    "file": os.path.basename(path),
                    "top": None
                    if top is None
                    else {
                        "name": top.name,
                        "class_id": top.cls_id,
                        "score": float(top.score),
                    },
                    "n_det": len(detections),
                }
            )
            print(
                "%-8s truth=%-18s top=%-18s score=%s"
                % (
                    folder,
                    truth or "negative",
                    top.name if top else "-",
                    "%.3f" % top.score if top else "-",
                )
            )

    report = {
        "bin": args.bin,
        "dataset": args.dataset,
        "images": len(rows),
        "thresholds": {},
        "rows": rows,
    }
    for value in sorted(float(item) for item in args.thresholds.split(",") if item):
        by_class = collections.defaultdict(lambda: [0, 0])
        negative_total = 0
        negative_fp = 0
        for row in rows:
            top = row["top"]
            if row["truth"] is None:
                negative_total += 1
                if top is not None and top["score"] >= value:
                    negative_fp += 1
                continue
            by_class[row["truth"]][1] += 1
            if (
                top is not None
                and top["score"] >= value
                and top["name"] == row["truth"]
            ):
                by_class[row["truth"]][0] += 1

        classes = {
            name: {
                "hits": counts[0],
                "total": counts[1],
                "accuracy": counts[0] / float(counts[1]) if counts[1] else None,
            }
            for name, counts in sorted(by_class.items())
        }
        hits = sum(item["hits"] for item in classes.values())
        total = sum(item["total"] for item in classes.values())
        summary = {
            "hits": hits,
            "total": total,
            "accuracy": hits / float(total) if total else None,
            "negative_fp": negative_fp,
            "negative_total": negative_total,
            "classes": classes,
        }
        report["thresholds"][str(value)] = summary
        print(
            "SUMMARY conf=%.2f %d/%d (%.1f%%) negative_fp=%d/%d"
            % (
                value,
                hits,
                total,
                100.0 * hits / max(1, total),
                negative_fp,
                negative_total,
            )
        )
        for name, item in classes.items():
            print(
                "  %-18s %3d/%3d %.1f%%"
                % (
                    name,
                    item["hits"],
                    item["total"],
                    100.0 * item["accuracy"] if item["accuracy"] is not None else 0.0,
                )
            )

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
