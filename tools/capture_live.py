#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""订阅 /image (CompressedImage) 抓帧，并统计检测消息。"""
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ai_msgs.msg import PerceptionTargets

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 3


class Cap(Node):
    def __init__(self):
        super().__init__('cap2')
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
        ext = '.jpg'
        fn = '/root/vision_ws/live_%d%s' % (self.n, ext)
        if m.format and 'jpeg' not in m.format.lower() and 'jpg' not in m.format.lower():
            ext = '.' + (m.format or 'bin').replace('/', '_')
            fn = '/root/vision_ws/live_%d%s' % (self.n, ext)
        with open(fn, 'wb') as f:
            f.write(bytes(m.data))
        print('IMG %d saved %s fmt=%s bytes=%d' % (self.n, fn, m.format, len(m.data)))
        self.n += 1

    def on_det(self, m):
        n = sum(len(t.tracks) for t in m.targets)
        if n == 0:
            self.empty += 1
        else:
            self.nonempty += 1
            desc = []
            for t in m.targets:
                for tr in t.tracks:
                    for roi in tr.rois:
                        desc.append('%s@%.2f(%d,%d,%d,%d)' % (
                            t.type, roi.confidence, roi.rect.x_offset,
                            roi.rect.y_offset, roi.rect.width, roi.rect.height))
            if len(self.samples) < 8:
                self.samples.append(' | '.join(desc))


def main():
    rclpy.init()
    node = Cap()
    deadline = time.time() + 40
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.n >= N_IMG and (node.empty + node.nonempty) >= 40:
            break
    print('检测消息: 空=%d 非空=%d' % (node.empty, node.nonempty))
    for s in node.samples:
        print('  检出:', s)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()