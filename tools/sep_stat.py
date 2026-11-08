#!/usr/bin/env python3
"""红/绿料 vs 背景的可分性分析：红按跨 0 度拆两段，绿对比主块与背景块的 S/V 分布。"""
import numpy as np, cv2, rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

frames = []
def cb(m):
    frames.append(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR))
rclpy.init(); n = Node('sep'); n.create_subscription(CompressedImage, '/image', cb,
    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
t0 = time.time()
while time.time() - t0 < 15 and not frames:
    rclpy.spin_once(n, timeout_sec=0.2)
bgr = frames[-1]; hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)

def q(a, name):
    if len(a) == 0:
        print('   %s: none' % name); return
    print('   %-10s n=%6d H %3d/%3d/%3d  S %3d/%3d/%3d  V %3d/%3d/%3d' % (name, len(a),
          np.percentile(a[0], 5), np.median(a[0]), np.percentile(a[0], 95),
          np.percentile(a[1], 5), np.median(a[1]), np.percentile(a[1], 95),
          np.percentile(a[2], 5), np.median(a[2]), np.percentile(a[2], 95)))

# 红：粗筛后按 H 拆成 0 侧与 179 侧
red = (S >= 70) & (V >= 90) & ((H <= 14) | (H >= 160))
print('== red (S>=70) ==')
q((H[red], S[red], V[red]), 'red all')
for lo, hi, nm in ((0, 14, 'h0-14'), (160, 179, 'h160+'), (15, 159, 'h15-159')):
    m = red & (H >= lo) & (H <= hi)
    q((H[m], S[m], V[m]), nm)

# 绿：粗筛，再对比主块与其余块
green = ((H >= 70) & (H <= 115)) & (S >= 40) & (V >= 40)
m8 = green.astype(np.uint8) * 255
ncc, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
order = sorted(range(1, ncc), key=lambda i: -stats[i, 4])
print('== green (S>=40) ==')
q((H[green], S[green], V[green]), 'green all')
if order:
    i = order[0]
    m = lab == i
    q((H[m], S[m], V[m]), 'main comp')
    others = green & ~m
    q((H[others], S[others], V[others]), 'others')
    print('   main bbox', stats[i][:4])
    for thr in (55, 60, 65, 70):
        keep = m & (S >= thr)
        # 背景用 others 在 S>=thr 下的数量
        bg = others & (S >= thr)
        print('   S>=%d : main 保留 %.0f%%  背景仍留 %d px' % (thr, 100.0 * keep.sum() / m.sum(), bg.sum()))
rclpy.shutdown()
