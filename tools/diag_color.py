# -*- coding: utf-8 -*-
"""诊断：为什么绿色/红色物料会被判成蓝色（只读，不影响节点）。"""
import json, sys, os, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "vision_sorter"),
          "/root/vision_ws", "/root/vision_ws/vision_sorter"):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as C
import color_classifier as CC


class Diag(Node):
    def __init__(self):
        super().__init__("diag_color")
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.frame = None
        self.seen = set()
        self.rows = []
        self.create_subscription(CompressedImage, C.IMG_TOPIC, self.on_img, qos)
        self.create_subscription(String, "/sort/debug", self.on_dbg, 10)

    def on_img(self, msg):
        self.frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

    def on_dbg(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        top = d.get("top")
        if not top or self.frame is None:
            return
        key = tuple(top.get("bbox") or ())
        if not key or key in self.seen:
            return
        f = self.frame
        H, W = f.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in top["bbox"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return
        self.seen.add(key)
        roi = f[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        chiral = CC._chromatic_mask(hsv)
        frac = cv2.countNonZero(chiral) / float(roi.shape[0] * roi.shape[1])
        votes = CC._vote(hsv)
        name, conf = CC.classify_color(roi.copy())
        total = float(roi.shape[0] * roi.shape[1])
        ordered = sorted(votes.items(), key=lambda kv: -kv[1])[:4]
        med_h = int(np.median(hsv[chiral > 0][:, 0])) if frac > 0.02 else -1
        med_s = int(np.median(hsv[:, :, 1]))
        med_v = int(np.median(hsv[:, :, 2]))
        self.rows.append(1)
        print("[#%d] bbox=[%d,%d,%d,%d] yolo=%s conf=%.2f | 判色=%s conf=%.2f | 彩色=%.0f%% h=%d s=%d v=%d | %s"
              % (len(self.rows), x1, y1, x2, y2, top.get("shape"),
                 float(top.get("conf") or 0), name, conf, frac * 100,
                 med_h, med_s, med_v,
                 " ".join("%s:%d(%.0f%%)" % (k, v, 100.0 * v / total) for k, v in ordered)))


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    rclpy.init()
    n = Diag()
    t0 = time.time()
    while time.time() - t0 < secs:
        rclpy.spin_once(n, timeout_sec=0.2)
    print("=== 共 %d 个目标 ===" % len(n.rows))
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
