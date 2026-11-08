#!/usr/bin/env python3
"""Capture one frame whenever a colored object's pose changes enough."""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def color_mask(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "red":
        return cv2.inRange(hsv, (0, 40, 40), (12, 255, 255)) | cv2.inRange(
            hsv, (160, 40, 40), (179, 255, 255)
        )
    return cv2.inRange(hsv, (35, 40, 40), (105, 255, 255))


def pose(mask):
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    if area < 100:
        return None
    x, y, width, height = [int(v) for v in stats[index, :4]]
    points = cv2.findNonZero((mask == index).astype(np.uint8))
    if points is None:
        return None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(points)
    return {
        "center": (float(cx), float(cy)),
        "bbox": (x, y, width, height),
        "rect": (float(rw), float(rh), float(angle % 90.0)),
        "area": area,
    }


def is_distinct(candidate, saved, min_shift, min_angle, min_size_change):
    cx, cy = candidate["center"]
    rw, rh, angle = candidate["rect"]
    for other in saved:
        ox, oy = other["center"]
        orw, orh, oangle = other["rect"]
        shift = math.hypot(cx - ox, cy - oy)
        angle_delta = min(abs(angle - oangle), 90.0 - abs(angle - oangle))
        size_delta = max(
            abs(rw - orw) / max(1.0, orw),
            abs(rh - orh) / max(1.0, orh),
        )
        if shift < min_shift and angle_delta < min_angle and size_delta < min_size_change:
            return False
    return True


class PoseCapture(Node):
    def __init__(self, args):
        super().__init__("gc_pose_capture")
        self.args = args
        self.saved = []
        self.last_check = 0.0
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(CompressedImage, args.topic, self.on_image, qos)

    def on_image(self, msg):
        if len(self.saved) >= self.args.count:
            return
        now = time.time()
        if now - self.last_check < 0.2:
            return
        self.last_check = now
        payload = bytes(msg.data)
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        current = pose(color_mask(image, self.args.color))
        if current is None:
            return
        if self.saved and not is_distinct(
            current,
            self.saved,
            self.args.min_shift,
            self.args.min_angle,
            self.args.min_size_change,
        ):
            return
        index = len(self.saved) + 1
        filename = "%s_%s_pose_%03d.jpg" % (
            self.args.color, self.args.session, index
        )
        path = os.path.join(self.args.output, filename)
        with open(path, "wb") as handle:
            handle.write(payload)
        record = {
            "file": filename,
            "bytes": len(payload),
            "center": list(current["center"]),
            "bbox": list(current["bbox"]),
            "rect": list(current["rect"]),
            "area": current["area"],
            "stamp": float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9,
        }
        self.saved.append(record)
        self.get_logger().info(
            "saved %d/%d %s center=(%.1f,%.1f) rect=(%.1f,%.1f,%.1f)"
            % (
                index,
                self.args.count,
                filename,
                current["center"][0],
                current["center"][1],
                current["rect"][0],
                current["rect"][1],
                current["rect"][2],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--color", choices=("red", "green"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--min-shift", type=float, default=20.0)
    parser.add_argument("--min-angle", type=float, default=12.0)
    parser.add_argument("--min-size-change", type=float, default=0.12)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    rclpy.init()
    node = PoseCapture(args)
    deadline = time.time() + args.timeout
    try:
        while len(node.saved) < args.count and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        with open(os.path.join(args.output, "capture_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(node.saved, handle, ensure_ascii=False, indent=2)
        node.destroy_node()
        rclpy.shutdown()
    print("captured %d/%d distinct %s poses" % (len(node.saved), args.count, args.color))


if __name__ == "__main__":
    main()
