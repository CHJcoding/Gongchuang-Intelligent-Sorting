#!/usr/bin/env python3
"""Save exactly one raw CompressedImage frame from /image."""

import argparse
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class RawCapture(Node):
    def __init__(self, topic):
        super().__init__("gc_raw_frame_capture")
        self.payload = None
        self.stamp = None
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(CompressedImage, topic, self.callback, qos)

    def callback(self, msg):
        if self.payload is None:
            self.payload = bytes(msg.data)
            self.stamp = float(msg.header.stamp.sec) + (
                float(msg.header.stamp.nanosec) * 1e-9
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    args.output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rclpy.init()
    node = RawCapture(args.topic)
    deadline = time.time() + args.timeout
    try:
        while node.payload is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.payload is None:
            raise SystemExit("timeout waiting for %s" % args.topic)
        with open(args.output, "wb") as handle:
            handle.write(node.payload)
        print(
            "saved %s bytes=%d stamp=%.9f"
            % (args.output, len(node.payload), node.stamp)
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
