#!/usr/bin/env python3
"""用建议区间在当前现场帧上做分割可视化：原图 / 红掩膜 / 绿掩膜 拼成一张图。"""
import sys, time
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

RED = [((0, 54, 114), (179, 182, 255))]
GREEN = [((77, 35, 41), (111, 129, 226))]

frames = []
def cb(m):
    frames.append(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR))

rclpy.init()
n = Node('mask_vis')
n.create_subscription(CompressedImage, '/image', cb,
    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
t0 = time.time(); last = 0
while time.time() - t0 < 20 and sum(1 for _ in range(1)) :
    rclpy.spin_once(n, timeout_sec=0.2)
    if frames:
        break
bgr = frames[-1]; hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def overlay(ranges):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    out = bgr.copy(); out[m > 0] = (0, 255, 255)
    return out, int((m > 0).sum())

ro, nr = overlay(RED)
go, ng = overlay(GREEN)
mont = np.hstack([bgr, ro, go])
cv2.imwrite('/tmp/mask_vis.jpg', mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('red px=%d green px=%d size=%s' % (nr, ng, mont.shape))
rclpy.shutdown()
