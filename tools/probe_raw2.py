# -*- coding: utf-8 -*-
import time, cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def sharp(img, tag):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    roi = g[int(h * 0.25):int(h * 0.85), int(w * 0.25):int(w * 0.85)]
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    print("%-32s %dx%d 拉普拉斯方差=%8.1f V中位=%3d"
          % (tag, img.shape[1], img.shape[0], lap,
             int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]))))
    return lap


class Grab(Node):
    def __init__(self):
        super().__init__("probe_raw2")
        self.br = CvBridge()
        self.last = None
        self.n = 0
        q = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/image", self.cb, q)
        self.create_subscription(Image, "/image_annotated", self.cb2, q)

    def cb(self, msg):
        try:
            self.last = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.n += 1
        except Exception as e:
            print("convert err:", e)

    def cb2(self, msg):
        pass


rclpy.init()
node = Grab()
t0 = time.time()
while time.time() - t0 < 8.0:
    rclpy.spin_once(node, timeout_sec=0.2)
print("收到 /image 帧数:", node.n)
if node.last is not None:
    cv2.imwrite("/root/vision_ws/hsv_tune/ros_raw.png", node.last)
    sharp(node.last, "ROS /image 原始帧 (best_effort)")
node.destroy_node()
rclpy.shutdown()
