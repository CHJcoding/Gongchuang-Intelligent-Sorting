# -*- coding: utf-8 -*-
"""临时：发布假的检测 + 图像，用来验证 sort_vision_node 的抓取流控（无需相机）。"""
import cv2
import rclpy
from rclpy.node import Node

from ai_msgs.msg import PerceptionTargets, Target, Roi
from sensor_msgs.msg import CompressedImage

IMG = "/root/vision_ws/t08_0.jpg"
BOX = (278, 189, 82, 78)      # 该区域在 t08_0.jpg 上是绿色圆锥
CONF = 0.99


class FakeDet(Node):
    def __init__(self):
        super().__init__("fake_det")
        self.pub_det = self.create_publisher(
            PerceptionTargets, "/hobot_dnn_detection", 10)
        self.pub_img = self.create_publisher(
            CompressedImage, "/image", 10)
        img = cv2.imread(IMG)
        if img is None:
            raise SystemExit("读不到 %s" % IMG)
        ok, buf = cv2.imencode(".jpg", img)
        self.data = buf.tobytes()
        self.h, self.w = img.shape[:2]
        self.create_timer(0.1, self.tick)
        self.get_logger().info("假检测发布器启动 %dx%d" % (self.w, self.h))

    def tick(self):
        stamp = self.get_clock().now().to_msg()

        ci = CompressedImage()
        ci.header.stamp = stamp
        ci.format = "jpeg"
        ci.data = self.data
        self.pub_img.publish(ci)

        t = PerceptionTargets()
        t.header.stamp = stamp
        tgt = Target()
        tgt.type = "YZHU"
        roi = Roi()
        roi.type = "body"
        roi.rect.x_offset = BOX[0]
        roi.rect.y_offset = BOX[1]
        roi.rect.width = BOX[2]
        roi.rect.height = BOX[3]
        roi.confidence = CONF
        tgt.rois = [roi]
        t.targets = [tgt]
        self.pub_det.publish(t)


def main():
    rclpy.init()
    n = FakeDet()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()