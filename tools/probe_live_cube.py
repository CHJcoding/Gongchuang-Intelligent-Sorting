#!/usr/bin/env python3

import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

ROOT = "/root/vision_ws"
sys.path.insert(0, os.path.join(ROOT, "vision_sorter"))

import config as C
import camera_model
import grasp_refine
from color_classifier import classify_color
from yolov8_bpu import Yolov8Bpu, sigmoid


class GrabOne(Node):
    def __init__(self):
        super().__init__("probe_live_cube")
        self.frame = None
        self.create_subscription(
            CompressedImage, C.IMG_TOPIC, self.on_image, 10)

    def on_image(self, msg):
        data = bytes(msg.data)
        frame = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_COLOR)
        if frame is not None:
            self.frame = frame


def main():
    rclpy.init()
    node = GrabOne()
    deadline = time.time() + 8.0
    while time.time() < deadline and node.frame is None:
        rclpy.spin_once(node, timeout_sec=0.2)
    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()

    if frame is None:
        raise SystemExit("没有收到 /image")
    path = os.path.join(ROOT, "live_cube_probe.jpg")
    cv2.imwrite(path, frame)
    print("frame", path, frame.shape)

    detector = Yolov8Bpu(
        C.YOLO_BIN_MODEL, C.YOLO_CLASS_NAMES,
        conf_thres=0.001, iou_thres=0.5, verbose=True)
    detections = detector.detect(frame, conf_thres=0.001, iou_thres=0.5)
    print("detections", len(detections))
    for d in sorted(detections, key=lambda x: -x.score):
        level, why = camera_model.check_point(d.cx, d.cy)
        valid = True
        margin = getattr(C, "DET_BORDER_MARGIN_PX", 6)
        if d.x1 < margin or d.y1 < margin or \
                d.x2 >= C.IMAGE_WIDTH - margin or d.y2 >= C.IMAGE_HEIGHT - margin:
            valid = False
        if d.w < getattr(C, "DET_MIN_BOX_PX", 12) or \
                d.h < getattr(C, "DET_MIN_BOX_PX", 12):
            valid = False
        if level != "ok":
            valid = False
        print("%-18s score=%.4f bbox=(%.0f,%.0f)-(%.0f,%.0f) "
              "center=(%.0f,%.0f) valid=%s level=%s %s"
              % (d.name, d.score, d.x1, d.y1, d.x2, d.y2,
                 d.cx, d.cy, valid, level, why))
        if d.score >= C.YOLO_CONF_THRES:
            shape_name, color = C.parse_yolo_class(d.name)
            cconf = d.score
            gp = (grasp_refine.refine(
                frame, [d.x1, d.y1, d.x2, d.y2], color)
                if color and getattr(C, "GRASP_REFINE", False) else None)
            if gp and gp["ok"]:
                fb = [int(round(v)) for v in gp["tight_bbox"]]
                fcx, fcy = (fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0
                flevel, fwhy = camera_model.check_point(fcx, fcy)
                fvalid = (fb[2] - fb[0] >= getattr(C, "DET_MIN_BOX_PX", 12)
                          and fb[3] - fb[1] >= getattr(C, "DET_MIN_BOX_PX", 12)
                          and fb[0] >= margin and fb[1] >= margin
                          and fb[2] < C.IMAGE_WIDTH - margin
                          and fb[3] < C.IMAGE_HEIGHT - margin
                          and flevel == "ok")
                print("  FIT color=%s cconf=%.3f bbox=%s valid=%s level=%s %s"
                      % (color, cconf, fb, fvalid, flevel, fwhy))
            else:
                print("  FIT failed color=%s reason=%s"
                      % (color, None if gp is None else gp["reason"]))

    arrs, ratio, dw, dh = detector.raw_forward(frame)
    print("class max probabilities")
    for scale_i in range(3):
        cls = np.asarray(arrs[scale_i * 2], dtype=np.float32)[0]
        logits = cls.min() < 0.0 or cls.max() > 1.0
        probs = sigmoid(cls) if logits else cls
        maxima = probs.reshape(-1, probs.shape[-1]).max(axis=0)
        ranked = sorted(
            zip(C.YOLO_CLASS_NAMES, maxima.tolist()), key=lambda x: -x[1])
        print("scale%d logits=%s: %s" % (
            scale_i, logits,
            " ".join("%s=%.4f" % (n, v) for n, v in ranked)))

    if detections:
        d = max(detections, key=lambda x: x.score)
        cls = np.asarray(arrs[0], dtype=np.float32)[0]
        logits = cls.min() < 0.0 or cls.max() > 1.0
        probs = sigmoid(cls) if logits else cls
        lx = int(round((d.cx + dw) / ratio / 8.0))
        ly = int(round((d.cy + dh) / ratio / 8.0))
        lx = max(0, min(probs.shape[1] - 1, lx))
        ly = max(0, min(probs.shape[0] - 1, ly))
        ranked = sorted(
            zip(C.SHAPE_NAMES, probs[ly, lx].tolist()),
            key=lambda x: -x[1])
        print("top target center grid=(%d,%d): %s" % (
            lx, ly, " ".join("%s=%.4f" % (n, v) for n, v in ranked)))
        x1, y1, x2, y2 = [int(round(v)) for v in
                          (d.x1, d.y1, d.x2, d.y2)]
        roi = frame[max(0, y1):min(frame.shape[0], y2),
                    max(0, x1):min(frame.shape[1], x2)]
        color, cconf = classify_color(roi)
        gp = grasp_refine.refine(
            frame, [d.x1, d.y1, d.x2, d.y2], color) if color else None
        if gp and gp["ok"]:
            rw, rh = gp["rect_size"]
            rect_fill = gp["mask_px"] / max(1.0, rw * rh)
            aspect = max(rw, rh) / max(1.0, min(rw, rh))
            print("top target geometry color=%s cconf=%.3f coverage=%.3f "
                  "solidity=%.3f rect_fill=%.3f aspect=%.3f radius=%.2f"
                  % (color, cconf, gp["coverage"], gp["solidity"],
                     rect_fill, aspect, gp["radius"]))
        else:
            print("top target geometry failed color=%s reason=%s"
                  % (color, None if gp is None else gp["reason"]))


if __name__ == "__main__":
    main()
