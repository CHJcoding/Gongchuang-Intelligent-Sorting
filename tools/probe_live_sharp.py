# -*- coding: utf-8 -*-
import sys, time, cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
from color_classifier import classify_color
from yolov8_bpu import Yolov8Bpu


def sharp(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    roi = g[int(h * 0.25):int(h * 0.85), int(w * 0.25):int(w * 0.85)]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


class Grab(Node):
    def __init__(self):
        super().__init__("probe_sharp_live")
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.frames = []
        self.create_subscription(CompressedImage, "/image", self.cb, q)

    def cb(self, msg):
        im = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            self.frames.append(im)
            if len(self.frames) > 60:
                self.frames.pop(0)


rclpy.init()
node = Grab()
t0 = time.time()
while time.time() - t0 < 6.0:
    rclpy.spin_once(node, timeout_sec=0.2)
frames = node.frames[-30:] if len(node.frames) >= 30 else node.frames
node.destroy_node()
rclpy.shutdown()
print("拿到帧数:", len(node.frames))

ref = cv2.imread("/root/vision_ws/hsv_tune/f00.jpg")
print("f00(早上那张，材料齐) 清晰度=%.1f V中位=%d" % (sharp(ref), int(np.median(cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)[:, :, 2]))))
if frames:
    last = frames[-1]
    cv2.imwrite("/root/vision_ws/hsv_tune/live_raw2.jpg", last, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    vals = [sharp(f) for f in frames]
    print("现在实时 清晰度 %.1f~%.1f 中位=%.1f   V中位=%d"
          % (min(vals), max(vals), float(np.median(vals)),
             int(np.median(cv2.cvtColor(last, cv2.COLOR_BGR2HSV)[:, :, 2]))))
    # 和早上的图比画面是否换了（帧差）
    if ref is not None:
        d = cv2.absdiff(cv2.resize(last, (640, 480)), ref)
        print("与 f00 的平均像素差=%.1f（>25 基本就是机位/物料全变了）" % float(d.mean()))

det = Yolov8Bpu(getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin"),
                C.SHAPE_NAMES, conf_thres=0.25,
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=False)
print("== 连测 %d 帧，看类别是否稳定 ==" % min(5, len(frames)))
for i, f in enumerate(frames[:: max(1, len(frames) // 5)][:5], 1):
    ds = sorted(det.detect(f), key=lambda d: -float(d.score))
    print("第%d帧: %s" % (i, ", ".join("%s %.2f" % (d.name, d.score) for d in ds) or "无检出"))
