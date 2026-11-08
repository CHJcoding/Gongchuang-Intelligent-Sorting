# -*- coding: utf-8 -*-
import sys, cv2, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def sharp(img, tag):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    roi = g[int(h * 0.25):int(h * 0.85), int(w * 0.25):int(w * 0.85)]
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    ten = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    print("%-30s %dx%d 拉普拉斯方差=%8.1f Tenengrad=%6.2f V中位=%3d"
          % (tag, img.shape[1], img.shape[0], lap, ten,
             int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]))))
    return lap


class Grab(Node):
    def __init__(self):
        super().__init__("probe_raw_frame")
        self.br = CvBridge()
        self.last = None
        self.n = 0
        self.create_subscription(Image, "/image", self.cb, 10)

    def cb(self, msg):
        try:
            self.last = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.n += 1
        except Exception as e:
            print("convert err:", e)


rclpy.init()
node = Grab()
import time
t0 = time.time()
while time.time() - t0 < 8.0:
    rclpy.spin_once(node, timeout_sec=0.2)
raw = node.last
print("收到 /image 帧数:", node.n)
if raw is None:
    print("没收到 /image")
else:
    cv2.imwrite("/root/vision_ws/hsv_tune/ros_raw.png", raw)
    sharp(raw, "ROS /image 原始帧 (ros_raw.png)")

pv = cv2.imread("/root/vision_ws/hsv_tune/../hsv_tune/none.jpg")
import os
if os.path.exists("/tmp/pv_last.jpg"):
    im = cv2.imread("/tmp/pv_last.jpg")
    if im is not None:
        sharp(im, "预览 /last.jpg")
node.destroy_node()
rclpy.shutdown()
