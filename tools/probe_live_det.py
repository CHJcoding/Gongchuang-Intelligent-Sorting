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
import camera_model, comm
from hand_eye import pixel_to_robot


def xy_mm(cx, cy):
    try:
        u, v = camera_model.undistort_point(cx, cy)
        x, y, _ = pixel_to_robot(u, v)
        return (round(float(x), 1), round(float(y), 1))
    except Exception:
        return None


def in_ws(p):
    if p is None:
        return None
    return (C.WORKSPACE_X_MIN <= p[0] <= C.WORKSPACE_X_MAX) and (C.WORKSPACE_Y_MIN <= p[1] <= C.WORKSPACE_Y_MAX)


class Grab(Node):
    def __init__(self):
        super().__init__("probe_live_det")
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.last = None
        self.n = 0
        self.create_subscription(CompressedImage, "/image", self.cb, q)

    def cb(self, msg):
        arr = np.frombuffer(bytes(msg.data), np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is not None:
            self.last = im
            self.n += 1


rclpy.init()
node = Grab()
t0 = time.time()
while time.time() - t0 < 8.0:
    rclpy.spin_once(node, timeout_sec=0.2)
frame = node.last
print("收到 /image(CompressedImage) 帧数:", node.n)
node.destroy_node()
rclpy.shutdown()
if frame is None:
    raise SystemExit("拿不到实时帧")
cv2.imwrite("/root/vision_ws/hsv_tune/live_raw.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

det = Yolov8Bpu(getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin"),
                C.SHAPE_NAMES, conf_thres=0.25,
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=False)
dets = sorted(det.detect(frame), key=lambda d: -float(d.score))
canvas = frame.copy()
print("== 实时帧检出 %d 个（conf_thres=0.25 全列，>=%.2f 才会下发）==" % (len(dets), C.YOLO_CONF_THRES))
for i, d in enumerate(dets, 1):
    x1, y1, x2, y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
    roi = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    cn, cc = (None, 0.0)
    if roi.size:
        cn, cc = classify_color(roi)
    p = xy_mm((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    ok = float(d.score) >= C.YOLO_CONF_THRES
    col = (0, 220, 0) if ok else (0, 165, 255)
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
    cv2.putText(canvas, "#%d %s %.2f" % (i, d.name, d.score), (int(x1) + 2, max(12, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    print("#%d %-16s %.3f  %s color=%-5s conf=%.2f  mm=%s  限位=%s  会下发=%s"
          % (i, d.name, d.score, "OK " if ok else "低 ", str(cn), cc, p, in_ws(p), ok and cn is not None and in_ws(p)))
cv2.imwrite("/root/vision_ws/hsv_tune/live_det.jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print("标注图: /root/vision_ws/hsv_tune/live_det.jpg")
