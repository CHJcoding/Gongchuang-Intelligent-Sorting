#!/usr/bin/env python3
"""Capture raw /image JPEG frames for a color/layout dataset session."""

import argparse
import hashlib
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class CaptureNode(Node):
    def __init__(self, args):
        super().__init__("gc_dataset_capture")
        self.args = args
        self.frames = []
        self.last_keep = 0.0
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(CompressedImage, args.topic, self.on_image, qos)

    @staticmethod
    def red_fraction(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0, 40, 40), np.uint8),
                           np.array((12, 255, 255), np.uint8))
        mask |= cv2.inRange(hsv, np.array((160, 40, 40), np.uint8),
                            np.array((179, 255, 255), np.uint8))
        return float(np.count_nonzero(mask)) / float(mask.size)

    @staticmethod
    def lap_var(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def on_image(self, msg):
        if len(self.frames) >= self.args.count:
            return
        now = time.time()
        if now - self.last_keep < self.args.interval:
            return
        payload = bytes(msg.data)
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        self.last_keep = now
        index = len(self.frames) + 1
        filename = "%s_%03d.jpg" % (self.args.prefix, index)
        path = os.path.join(self.args.output, filename)
        with open(path, "wb") as handle:
            handle.write(payload)
        record = {
            "file": filename,
            "bytes": len(payload),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "red_fraction": round(self.red_fraction(image), 6),
            "laplacian": round(self.lap_var(image), 3),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "stamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
        }
        self.frames.append(record)
        self.get_logger().info(
            "capture %03d red=%.4f lap=%.1f bytes=%d"
            % (index, record["red_fraction"], record["laplacian"], len(payload))
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.8)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    rclpy.init()
    node = CaptureNode(args)
    deadline = time.time() + max(20.0, args.count * args.interval + 10.0)
    try:
        while len(node.frames) < args.count and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        meta_path = os.path.join(args.output, "capture_meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(node.frames, handle, ensure_ascii=False, indent=2)
        node.destroy_node()
        rclpy.shutdown()
    print("captured %d/%d frames to %s" % (len(node.frames), args.count, args.output))


if __name__ == "__main__":
    main()
