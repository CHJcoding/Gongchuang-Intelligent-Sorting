#!/usr/bin/env python3
"""Compare YOLOv8 BPU bins on the same set of camera frames."""

import argparse
import collections
import glob
import json
import os

import cv2

from verify_model_deploy import load_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", nargs="+", required=True)
    parser.add_argument("--images", required=True, help="image glob")
    parser.add_argument("--labels", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit("no images matched: %s" % args.images)
    labels = load_labels(args.labels)

    from yolov8_bpu import Yolov8Bpu

    report = {"images": paths, "conf": args.conf, "models": {}}
    for bin_path in args.bins:
        detector = Yolov8Bpu(
            bin_path,
            labels,
            conf_thres=args.conf,
            iou_thres=args.iou,
            verbose=True,
        )
        rows = []
        sequences = collections.Counter()
        scores = []
        class_counts = collections.Counter()
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                raise SystemExit("cannot read image: %s" % path)
            detections = sorted(detector.detect(image), key=lambda d: d.cx)
            sequence = tuple(item.name for item in detections)
            sequences[sequence] += 1
            scores.extend(item.score for item in detections)
            class_counts.update(item.name for item in detections)
            rows.append(
                {
                    "image": os.path.basename(path),
                    "sequence": list(sequence),
                    "detections": [
                        {
                            "class_id": item.cls_id,
                            "name": item.name,
                            "score": float(item.score),
                            "center": [float(item.cx), float(item.cy)],
                        }
                        for item in detections
                    ],
                }
            )
            print(
                "%-18s %-45s n=%d"
                % (os.path.basename(bin_path), os.path.basename(path), len(detections))
            )

        top_sequence, top_count = sequences.most_common(1)[0]
        model_report = {
            "bin": bin_path,
            "frames": len(paths),
            "detections": sum(class_counts.values()),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "class_counts": dict(class_counts),
            "top_sequence": list(top_sequence),
            "top_sequence_count": top_count,
            "sequence_stability": top_count / float(len(paths)),
            "unique_sequences": len(sequences),
            "rows": rows,
        }
        report["models"][bin_path] = model_report
        print(
            "SUMMARY %s frames=%d detections=%d mean=%.3f "
            "top_sequence=%s stability=%.3f unique=%d"
            % (
                os.path.basename(bin_path),
                len(paths),
                sum(class_counts.values()),
                model_report["mean_score"],
                ",".join(top_sequence) or "EMPTY",
                model_report["sequence_stability"],
                len(sequences),
            )
        )

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
