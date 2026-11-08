#!/usr/bin/env python3
"""统计建议区间在当前帧上的连通域构成，判断掩膜是否"一块料"还是散在背景。"""
import numpy as np, cv2, rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

RED = [((0, 54, 114), (179, 182, 255))]
GREEN = [((77, 35, 41), (111, 129, 226))]
frames = []
def cb(m):
    frames.append(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR))
rclpy.init()
n = Node('mask_stat')
n.create_subscription(CompressedImage, '/image', cb,
    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
t0 = time.time()
while time.time() - t0 < 15 and not frames:
    rclpy.spin_once(n, timeout_sec=0.2)
bgr = frames[-1]; hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
print('frame', bgr.shape)

for name, ranges in (('red', RED), ('green', GREEN)):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, lo, hi)
    raw = int((m > 0).sum())
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ncc, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    comps = sorted(range(1, ncc), key=lambda i: -stats[i, 4])[:6]
    print('--- %s total=%d open=%d comps=%d' % (name, raw, int((m > 0).sum()), ncc - 1))
    for i in comps:
        x, y, w, h, a = stats[i]
        sel = (lab == i)
        hh = hsv[:, :, 0][sel]; ss = hsv[:, :, 1][sel]; vv = hsv[:, :, 2][sel]
        print('   area=%6d bbox=(%3d,%3d,%3d,%3d) fill=%.2f H=%.0f S=%.0f V=%.0f' % (
            a, x, y, w, h, a / float(max(1, w * h)), np.median(hh), np.median(ss), np.median(vv)))
rclpy.shutdown()
