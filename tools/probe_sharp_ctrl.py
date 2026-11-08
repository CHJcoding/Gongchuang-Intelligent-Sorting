# -*- coding: utf-8 -*-
import time, cv2, numpy as np, subprocess
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


def sharp(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    roi = g[int(h * 0.25):int(h * 0.85), int(w * 0.25):int(w * 0.85)]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


class Grab(Node):
    def __init__(self):
        super().__init__("probe_sharp_ctrl")
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.frames = []
        self.create_subscription(CompressedImage, "/image", self.cb, q)

    def cb(self, msg):
        im = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            self.frames.append(im)
            if len(self.frames) > 90:
                self.frames.pop(0)


rclpy.init()
node = Grab()


def collect(node, sec=2.5):
    node.frames.clear()
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(node, timeout_sec=0.1)
    fr = list(node.frames)
    if not fr:
        return None, None
    vals = [sharp(f) for f in fr]
    return fr[-1], float(np.median(vals))


def setctl(name, val):
    subprocess.run(["v4l2-ctl", "-d", "/dev/cam0", "-c", "%s=%d" % (name, val)],
                   capture_output=True)


im, base = collect(node, 2.5)
print("当前 resolution=%s  清晰度中位=%.1f" % (None if im is None else im.shape[:2], base or -1))
for val in (6, 0):
    setctl("sharpness", val)
    time.sleep(1.2)
    im2, v = collect(node, 2.5)
    print("设 sharpness=%d -> 清晰度中位=%.1f" % (val, v if v else -1))
setctl("sharpness", 3)
time.sleep(1.0)
im3, v3 = collect(node, 2.0)
print("恢复 sharpness=3 -> 清晰度中位=%.1f" % (v3 if v3 else -1))
if im3 is not None:
    cv2.imwrite("/root/vision_ws/hsv_tune/live_s3.jpg", im3, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
node.destroy_node()
rclpy.shutdown()
