#!/usr/bin/env python3
"""在「当前现场光照 + 相机参数」下量红/绿料的 HSV 分布，输出可直接粘进
vision_sorter/config.py 的 COLOR_RANGES_HSV 片段。

用法：
    source /opt/tros/humble/setup.bash && python3 /root/vision_ws/tools/hsv_probe.py [帧数]

每 1s 抓一帧，取画面里最大的红块 / 绿块，按像素统计 H/S/V 的 p1/p50/p99，
再把多帧合并给出建议区间；同时统计「用建议区间去分割整幅画面时，落在目标框
以外的像素数」——这个数越大说明区间越脏，会把背景/台面吃进来。
"""
import sys, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# 粗筛区间（只为找到候选块，最终建议区间由统计给出）
SEEDS = {
    'red':   [((0, 40, 40), (14, 255, 255)), ((163, 40, 40), (180, 255, 255))],
    'green': [((45, 40, 40), (115, 255, 255))],
}
MIN_AREA = 400


def biggest_blob(hsv, ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(c)
    sub = hsv[y:y + h, x:x + w]
    m = np.zeros(sub.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(sub, lo, hi)
    px = sub[m > 0]
    return dict(bbox=(x, y, w, h), n=int(len(px)), hsv=px)


def pct(px, i, q):
    return float(np.percentile(px[:, i], q)) if len(px) else float('nan')


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rclpy.init()
    node = Node('hsv_probe')
    frames = []
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(CompressedImage, '/image', lambda m: frames.append(
        cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)), qos)

    acc = {k: [] for k in SEEDS}
    t0 = time.time(); last = 0.0
    while time.time() - t0 < n_frames * 1.2 + 10 and len(acc['red']) + len(acc['green']) < n_frames * len(SEEDS):
        rclpy.spin_once(node, timeout_sec=0.2)
        if not frames or time.time() - last < 1.0:
            continue
        last = time.time()
        bgr = frames[-1]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        line = []
        for name, ranges in SEEDS.items():
            b = biggest_blob(hsv, ranges)
            if b is None:
                line.append('%s=None' % name)
                continue
            acc[name].append(b)
            line.append('%s bbox=%s px=%d H=%.0f/%.0f/%.0f S=%.0f/%.0f/%.0f V=%.0f/%.0f/%.0f' % (
                name, b['bbox'], b['n'],
                pct(b['hsv'], 0, 1), pct(b['hsv'], 0, 50), pct(b['hsv'], 0, 99),
                pct(b['hsv'], 1, 1), pct(b['hsv'], 1, 50), pct(b['hsv'], 1, 99),
                pct(b['hsv'], 2, 1), pct(b['hsv'], 2, 50), pct(b['hsv'], 2, 99)))
        print(' | '.join(line))

    print('\n================ 建议区间 ================')
    for name, blobs in acc.items():
        if not blobs:
            print('# %s: 画面里没找到候选块' % name)
            continue
        px = np.vstack([b['hsv'] for b in blobs])
        lo = (max(0, int(np.percentile(px[:, 0], 1)) - 3),
              max(35, int(np.percentile(px[:, 1], 2)) - 10),
              max(40, int(np.percentile(px[:, 2], 2)) - 15))
        hi = (min(179, int(np.percentile(px[:, 0], 99)) + 3),
              min(255, int(np.percentile(px[:, 1], 99)) + 10),
              min(255, int(np.percentile(px[:, 2], 99)) + 15))
        # 脏度：用建议区间分割最后一帧，统计落在所有目标框之外的像素
        last_bgr = frames[-1]
        lhsv = cv2.cvtColor(last_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(lhsv, lo, hi)
        inner = np.zeros_like(m)
        for b in blobs[-2:]:
            x, y, w, h = b['bbox']
            inner[y:y + h, x:x + w] = 255
        leak = int(((m > 0) & (inner == 0)).sum())
        print('    "%s": [(%d, %d, %d), (%d, %d, %d)],   # 命中 %d px，框外漏入 %d px'
              % (name, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], len(px), leak))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
