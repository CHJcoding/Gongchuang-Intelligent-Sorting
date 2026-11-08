#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓 /image 帧 + 统计检测消息（兼容 Target.rois 与 Target.tracks 两种版本）。"""
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ai_msgs.msg import PerceptionTargets

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else 'v2'


def iter_rois(target):
    rois = getattr(target, 'rois', None)
    if rois:
        return list(rois)
    tracks = getattr(target, 'tracks', None)
    out = []
    if tracks:
        for tr in tracks:
            out.extend(list(getattr(tr, 'rois', []) or []))
    return out


class Cap(Node):
    def __init__(self):
        super().__init__('cap3')
        self.n = 0
        self.empty = 0
        self.nonempty = 0
        self.samples = []
        self.create_subscription(CompressedImage, '/image', self.on_img, 10)
        self.create_subscription(PerceptionTargets, '/hobot_dnn_detection',
                                 self.on_det, 10)

    def on_img(self, m):
        if self.n >= N_IMG:
            return
        fn = '/root/vision_ws/%s_%d.jpg' % (TAG, self.n)
        with open(fn, 'wb') as f:
            f.write(bytes(m.data))
        print('IMG %d %s fmt=%s bytes=%d' % (self.n, fn, m.format, len(m.data)))
        self.n += 1

    def on_det(self, m):
        desc = []
        for t in m.targets:
            for roi in iter_rois(t):
                desc.append('%s@%.2f(%d,%d,%d,%d)' % (
                    t.type, roi.confidence, roi.rect.x_offset,
                    roi.rect.y_offset, roi.rect.width, roi.rect.height))
        if not desc:
            self.empty += 1
        else:
            self.nonempty += 1
            if len(self.samples) < 6:
                self.samples.append(' | '.join(desc))


def main():
    rclpy.init()
    node = Cap()
    deadline = time.time() + 45
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.n >= N_IMG and (node.empty + node.nonempty) >= 60:
            break
    total = node.empty + node.nonempty
    print('DET_STATS total=%d empty=%d nonempty=%d' % (total, node.empty, node.nonempty))
    for s in node.samples:
        print('HIT %s' % s)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()