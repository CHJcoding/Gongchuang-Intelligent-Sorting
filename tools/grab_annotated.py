# -*- coding: utf-8 -*-
"""抓一帧 /image_annotated 存到磁盘，并统计"画上去的颜色"像素数。

用途：确认旋转贴合框真的画进了推流图里（不是只在离线脚本里能画）。
  - 绿框 (0,210,0)  -> 旋转贴合框
  - 红点 (0,0,255)  -> 抓取点
  - 黄框 (0,230,255)-> 原始 YOLO 框
"""
import sys
import time

import numpy as np
from sensor_msgs.msg import CompressedImage
import rclpy
from rclpy.node import Node
import cv2


class Grab(Node):
    def __init__(self, topic, out):
        super().__init__("grab_annotated")
        self.out = out
        self.saved = 0
        self.create_subscription(CompressedImage, topic, self.cb, 10)

    def cb(self, msg):
        if self.saved >= 2:
            return
        buf = np.frombuffer(bytes(msg.data), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
        self.saved += 1
        if self.saved == 2:            # 第 2 帧再存，确保画图逻辑已跑过
            cv2.imwrite(self.out, img)
            b, g, r = (img[:, :, 0].astype(np.int16), img[:, :, 1].astype(np.int16),
                       img[:, :, 2].astype(np.int16))
            green = int(((g - r > 90) & (g - b > 90)).sum())
            yellow = int(((r > 120) & (g > 120) & (b < 110)).sum())
            red = int(((r > 120) & (g < 90) & (b < 90)).sum())
            print("帧 %dx%d 已存 %s" % (img.shape[1], img.shape[0], self.out))
            print("绿(贴合框)像素=%d  黄(原框)像素=%d  红(抓取点)像素=%d"
                  % (green, yellow, red))
            print("判定：%s" % ("✓ 框已画进推流图"
                                if green > 200 else "✗ 没看到绿色贴合框"))
            raise SystemExit(0)


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/image_annotated"
    out = sys.argv[2] if len(sys.argv) > 2 else "/root/vision_ws/annotated_now.jpg"
    rclpy.init()
    node = Grab(topic, out)
    t0 = time.time()
    try:
        while time.time() - t0 < 20:
            rclpy.spin_once(node, timeout_sec=0.5)
    except SystemExit:
        pass
    if node.saved == 0:
        print("20 秒内没收到 %s 的任何一帧" % topic)
        print("判定：✗ 业务节点没有发布调试图（看 sort 日志有没有报错）")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if node.saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())