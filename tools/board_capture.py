#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同时抓 /image 一帧 和 一条 /hobot_dnn_detection，用于核对检测结果的坐标空间。"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ai_msgs.msg import PerceptionTargets

OUT_IMG = '/root/vision_ws/scene.jpg'
OUT_JSON = '/root/vision_ws/det.json'


class Capture(Node):
    def __init__(self):
        super().__init__('capture_node')
        self.img_done = False
        self.det_done = False
        self.create_subscription(Image, '/image', self.on_img, 10)
        self.create_subscription(PerceptionTargets, 'hobot_dnn_detection',
                                 self.on_det, 10)

    def on_img(self, m):
        if self.img_done:
            return
        data = bytes(m.data)
        try:
            with open(OUT_IMG, 'wb') as f:
                f.write(data)
        except Exception as e:
            print('write img failed:', e)
        print('IMAGE: w=%d h=%d encoding=%r step=%d bytes=%d' % (
            m.width, m.height, m.encoding, m.step, len(data)))
        self.img_done = True

    def on_det(self, m):
        if self.det_done:
            return
        out = {'stamp': None, 'targets': []}
        try:
            out['stamp'] = '%d.%09d' % (m.header.stamp.sec, m.header.stamp.nanosec)
            out['frame_id'] = m.header.frame_id
        except Exception:
            pass
        for t in m.targets:
            for tr in t.tracks:
                for roi in tr.rois:
                    out['targets'].append({
                        'type': t.type,
                        'track_id': int(tr.id) if hasattr(tr, 'id') else -1,
                        'conf': round(float(roi.confidence), 4),
                        'x': int(roi.rect.x_offset),
                        'y': int(roi.rect.y_offset),
                        'w': int(roi.rect.width),
                        'h': int(roi.rect.height),
                    })
        with open(OUT_JSON, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print('DET: %s' % json.dumps(out, ensure_ascii=False))
        self.det_done = True


def main():
    rclpy.init()
    node = Capture()
    deadline = time.time() + 40
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.img_done and node.det_done:
            break
    print('done: img=%s det=%s' % (node.img_done, node.det_done))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()