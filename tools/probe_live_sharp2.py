# -*- coding: utf-8 -*-
"""量测真实链路 /image 的清晰度（det 节点解码后送进检测的那一帧），并落盘样张。"""
import time, json, cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

class P(Node):
    def __init__(self):
        super().__init__("sharp_probe")
        q = QoSProfile(depth=5)
        q.reliability = ReliabilityPolicy.BEST_EFFORT
        q.history = HistoryPolicy.KEEP_LAST
        self.frames = []
        self.create_subscription(CompressedImage, "/image", self.cb, q)
    def cb(self, m):
        if len(self.frames) < 6:
            a = np.frombuffer(m.data, np.uint8)
            img = cv2.imdecode(a, cv2.IMREAD_COLOR)
            if img is not None:
                self.frames.append((img, len(m.data)))

rclpy.init()
n = P()
t0 = time.time()
while len(n.frames) < 6 and time.time() - t0 < 12:
    rclpy.spin_once(n, timeout_sec=0.3)
for i, (img, nbytes) in enumerate(n.frames):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lv = float(cv2.Laplacian(g, cv2.CV_32F).var())
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gn = float(np.percentile(np.hypot(gx, gy), 99.0) / (float(g.std()) or 1.0))
    print(json.dumps({"i": i, "sz": [img.shape[1], img.shape[0]], "jpeg_bytes": nbytes,
                      "lap": round(lv, 1), "grad99": round(gn, 2), "mean": round(float(g.mean()), 1)}), flush=True)
    if i == 2:
        cv2.imwrite("/root/vision_ws/tools/probe_out/live_image.png", img)
        cv2.imwrite("/root/vision_ws/tools/probe_out/live_crop.png", img[180:340, 260:520])
n.destroy_node(); rclpy.shutdown()
print("DONE")
