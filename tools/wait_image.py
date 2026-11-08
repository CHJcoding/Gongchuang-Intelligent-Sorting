#!/usr/bin/env python3
"""等 /image 真的在出帧就返回 0（比 ros2 topic hz 快得多，不启动 ros2 daemon）。

用法: python3 wait_image.py [超时秒=4] [最少帧数=3]
"""
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    need = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rclpy.init()
    node = Node("wait_image_probe")
    got = []
    node.create_subscription(
        CompressedImage, "/image", lambda m: got.append(1), 10)
    t0 = time.time()
    while time.time() - t0 < timeout and len(got) < need:
        rclpy.spin_once(node, timeout_sec=0.2)
    ok = len(got) >= need
    print("frames=%d in %.2fs" % (len(got), time.time() - t0))
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
