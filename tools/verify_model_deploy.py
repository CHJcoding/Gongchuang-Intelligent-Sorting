#!/usr/bin/env python3
"""Verify an RDK X5 .bin with model_info and one real inference."""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for candidate in (
    os.path.join(ROOT, "vision_sorter"),
    os.path.join(ROOT, "rdk", "vision_sorter"),
):
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)

import config as C  # noqa: E402
from yolov8_bpu import Yolov8Bpu  # noqa: E402


def prop_value(tensor, name, default=None):
    try:
        return getattr(tensor.properties, name)
    except (AttributeError, RuntimeError):
        return default


def shape_of(tensor):
    shape = prop_value(tensor, "shape")
    return tuple(int(v) for v in shape) if shape is not None else None


def text_of(value):
    return "" if value is None else str(value)


def expected_shapes(num_classes):
    return (
        (1, 80, 80, num_classes),
        (1, 80, 80, 64),
        (1, 40, 40, num_classes),
        (1, 40, 40, 64),
        (1, 20, 20, num_classes),
        (1, 20, 20, 64),
    )


def check_model_info(model_path, report, issues):
    executable = shutil.which("hrt_model_exec")
    if executable is None:
        issues.append("hrt_model_exec was not found")
        return
    command = [executable, "model_info", "--model_file", model_path]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    report["model_info"] = {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout,
    }
    print("---- hrt_model_exec model_info ----")
    print(result.stdout)
    if result.returncode != 0:
        issues.append("hrt_model_exec returned %d" % result.returncode)
    forbidden = (
        "HB_DNN_TENSOR_TYPE_S32",
        "HB_DNN_LAYOUT_NONE",
        "get_tensor_hw failed",
    )
    for marker in forbidden:
        if marker in result.stdout:
            issues.append("model_info contains forbidden marker: %s" % marker)


def open_camera(device):
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError("cannot open camera %s" % device)
    try:
        for _ in range(8):
            capture.read()
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError("camera read failed: %s" % device)
    return frame


def load_labels(path):
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        if labels:
            return labels
    return list(C.YOLO_CLASS_NAMES)


def main():
    global cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=os.getenv("GC_YOLO_BIN_MODEL", C.YOLO_BIN_MODEL))
    parser.add_argument("--labels", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--camera", default="/dev/cam0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", default="/tmp/model_deploy_verify.jpg")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--skip-model-info", action="store_true")
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    try:
        import cv2 as cv2_module
    except ImportError:
        print("ERROR: install opencv-python first", file=sys.stderr)
        return 2
    cv2 = cv2_module

    issues = []
    report = {"status": "FAIL", "bin": args.bin}
    if not os.path.isfile(args.bin):
        print("ERROR: model not found:", args.bin, file=sys.stderr)
        return 2

    if not args.skip_model_info:
        check_model_info(args.bin, report, issues)

    labels = load_labels(args.labels)
    report["labels"] = labels
    if len(labels) != C.YOLO_NUM_CLASSES:
        issues.append(
            "label count %d does not match configured %d"
            % (len(labels), C.YOLO_NUM_CLASSES)
        )

    detector = Yolov8Bpu(
        args.bin,
        labels,
        conf_thres=args.conf,
        iou_thres=args.iou,
        verbose=True,
    )
    model = detector.model
    input_shape = shape_of(model.inputs[0])
    input_dtype = text_of(prop_value(model.inputs[0], "dtype"))
    input_layout = text_of(prop_value(model.inputs[0], "layout"))
    report["input"] = {
        "shape": input_shape,
        "dtype": input_dtype,
        "layout": input_layout,
    }
    if input_shape != (1, 3, 640, 640):
        issues.append("runtime input shape is %s, expected (1, 3, 640, 640)" % (input_shape,))

    expected = expected_shapes(len(labels))
    runtime_outputs = []
    for index, tensor in enumerate(model.outputs):
        runtime_outputs.append(
            {
                "shape": shape_of(tensor),
                "dtype": text_of(prop_value(tensor, "dtype")),
                "layout": text_of(prop_value(tensor, "layout")),
            }
        )
    report["runtime_outputs"] = runtime_outputs
    if len(runtime_outputs) != 6:
        issues.append("runtime output count is %d, expected 6" % len(runtime_outputs))
    for index, item in enumerate(runtime_outputs):
        if index >= len(expected):
            break
        if item["shape"] != expected[index]:
            issues.append(
                "output[%d] shape is %s, expected %s"
                % (index, item["shape"], expected[index])
            )
        if "S32" in item["dtype"].upper():
            issues.append("output[%d] is S32" % index)
        if "NONE" in item["layout"].upper():
            issues.append("output[%d] layout is NONE" % index)
        if item["layout"] and "NHWC" not in item["layout"].upper():
            if "NCHW" not in item["layout"].upper() or item["shape"] != expected[index]:
                issues.append(
                    "output[%d] layout/shape mismatch: %s %s"
                    % (index, item["layout"], item["shape"])
                )

    frame = cv2.imread(args.image) if args.image else open_camera(args.camera)
    if frame is None:
        print("ERROR: cannot read image:", args.image, file=sys.stderr)
        return 2
    report["image_shape"] = list(frame.shape)

    arrays, ratio, pad_x, pad_y = detector.raw_forward(frame)
    report["forward"] = {
        "shape": [list(item.shape) for item in arrays],
        "dtype": [str(item.dtype) for item in arrays],
        "letterbox": [float(ratio), float(pad_x), float(pad_y)],
    }
    if len(arrays) != 6:
        issues.append("forward returned %d tensors, expected 6" % len(arrays))
    for index, array in enumerate(arrays):
        if index >= len(expected):
            break
        if tuple(array.shape) != expected[index]:
            issues.append(
                "forward output[%d] shape is %s, expected %s"
                % (index, tuple(array.shape), expected[index])
            )
        if array.dtype != np.float32:
            issues.append("forward output[%d] dtype is %s" % (index, array.dtype))

    detections = detector.detect(frame)
    report["detections"] = []
    for item in detections:
        record = {
            "class_id": int(item.cls_id),
            "name": item.name,
            "score": float(item.score),
            "box": [float(item.x1), float(item.y1), float(item.x2), float(item.y2)],
            "center": [float(item.cx), float(item.cy)],
        }
        report["detections"].append(record)
        print(
            "#%d %-18s %.3f box=(%.1f,%.1f)-(%.1f,%.1f) center=(%.1f,%.1f)"
            % (
                item.cls_id,
                item.name,
                item.score,
                item.x1,
                item.y1,
                item.x2,
                item.y2,
                item.cx,
                item.cy,
            )
        )
        if item.cls_id < 0 or item.cls_id >= len(labels):
            issues.append("detection class id %d is out of range" % item.cls_id)
        if abs(item.cx) < 1e-6 and abs(item.cy) < 1e-6:
            issues.append("detection center collapsed to (0,0)")
    if not detections and not args.allow_empty:
        issues.append("no objects detected; use a real/training image or --allow-empty")

    annotated = frame.copy()
    for item in detections:
        p1 = (int(round(item.x1)), int(round(item.y1)))
        p2 = (int(round(item.x2)), int(round(item.y2)))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "%s %.2f" % (item.name, item.score),
            (p1[0], max(16, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(args.save, annotated)
    report["annotated_image"] = args.save

    report["issues"] = issues
    report["status"] = "PASS" if not issues else "FAIL"
    print("RESULT:", report["status"])
    for issue in issues:
        print(" -", issue)
    print("Annotated image:", args.save)

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
