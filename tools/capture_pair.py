#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓一帧 + 收集检测框，写 JSON 供本地对比。"""
import json
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ai_msgs.msg import PerceptionTargets

N_DET = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def iter_rois(target):
    rois = getattr(target, 'rois', None)
    if rois:
        return list(rois)
    out = []
    for tr in (getattr(target, 'tracks', None) or []):
        out.extend(list(getattr(tr, 'rois', []) or []))
    return out


class Cap(Node):
    def __init__(self):
        super().__init__('cap_pair')
        self.img_saved = False
        self.msgs = []
        self.create_subscription(CompressedImage, '/image', self.on_img, 10)
        self.create_subscription(PerceptionTargets, '/hobot_dnn_detection',
                                 self.on_det, 10)

    def on_img(self, m):
        if self.img_saved:
            return
        with open('/root/vision_ws/pair.jpg', 'wb') as f:
            f.write(bytes(m.data))
        print('IMG saved pair.jpg bytes=%d' % len(m.data))
        self.img_saved = True

    def on_det(self, m):
        if len(self.msgs) >= N_DET:
            return
        rec = []
        for t in m.targets:
            for roi in iter_rois(t):
                rec.append({
                    'type': t.type,
                    'conf': float(roi.confidence),
                    'x': int(roi.rect.x_offset),
                    'y': int(roi.rect.y_offset),
                    'w': int(roi.rect.width),
                    'h': int(roi.rect.height),
                })
        self.msgs.append(rec)


def main():
    rclpy.init()
    node = Cap()
    deadline = time.time() + 60
    while time.time() < deadline and len(node.msgs) < N_DET:
        rclpy.spin_once(node, timeout_sec=0.1)
    with open('/root/vision_ws/pair_det.json', 'w') as f:
        json.dump(node.msgs, f, ensure_ascii=False)
    print('DET collected=%d' % len(node.msgs))
    print('last_msg_boxes=%d' % (len(node.msgs[-1]) if node.msgs else -1))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()