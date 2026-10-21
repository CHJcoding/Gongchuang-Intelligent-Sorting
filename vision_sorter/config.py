# -*- coding: utf-8 -*-
"""
全局配置：RDK X5 板端版本。现场只改这一个文件。
"""

# ============ 相机 (C100 USB, Linux V4L2) ============
CAM_INDEX = 0              # /dev/video0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

# ============ YOLOv5s 检测 (地平线 BPU .bin 模型) ============
# 基线（2026-10-21 实读核实）：best.pt = myYOLOv5s，7 类，训练 imgsz=640；
# 实际导出 ONNX/bin 为 672×672 + NHWC raw 三尺度（36 = 3×(5+7)）。
# 决策 D2：暂用板上已跑通的 672，不重导出；现场精度不足再回 640。
YOLO_BIN_MODEL = "weights/yolov5_best.bin"   # 板上实际文件名以板端为准
YOLO_IMGSZ = 672            # 必须与 .bin 输入尺寸一致，否则维度报错
YOLO_NUM_CLASSES = 7
YOLO_CONF_THRES = 0.55
YOLO_IOU_THRES = 0.45
# 7 类形状：顺序即 class_id，必须与 best.pt 的 model.names 完全一致
SHAPE_NAMES = ["BQ", "LK", "SLZ", "SMT", "WLZ", "YZ", "YZHU"]
SHAPE_NAMES_CN = {
    "BQ": "半球", "LK": "镂空正方体", "SLZ": "三棱柱", "SMT": "正四面体",
    "WLZ": "五棱柱", "YZ": "圆柱", "YZHU": "圆锥",
}

# ============ HSV 颜色阈值 ============
# 用 color_tuner.py 在现场补光下调好后填这里。
# (lower_HSV, upper_HSV), H:0-179, S:0-255, V:0-255
COLOR_RANGES_HSV = {
    "red":      [(0,   90,  50),  (10,  255, 255)],
    "orange":   [(11,  90,  50),  (22,  255, 255)],
    "yellow":   [(23,  90,  50),  (35,  255, 255)],
    "green":    [(36,  70,  40),  (85,  255, 255)],
    "cyan":     [(86,  70,  40),  (100, 255, 255)],
    "blue":     [(101, 70,  40),  (125, 255, 255)],
    "purple":   [(126, 70,  40),  (155, 255, 255)],
    "magenta":  [(156, 70,  40),  (169, 255, 255)],
    "black":    [(0,   0,   0),   (179, 60,  60)],
    "white":    [(0,   0,   200), (179, 30,  255)],
}
RED_HSV_LOWER2 = [(170, 90, 50), (179, 255, 255)]

# 反光/灰像素剔除
V_HIGH_THRES = 230
S_LOW_THRES = 40
MIN_PIXEL_RATIO = 0.15
# 深红 vs 暗橙 二次判别阈值
DARK_RED_RATIO_THRES = 0.62

# ============ 手眼标定 ============
# [x_mm, y_mm] = M @ [u, v, 1]，标定后填这里
HAND_EYE_M = [
    [0.038, 0.000, -24.5],
    [0.000, -0.038, 410.0],
]
Z_OFFSET = 0.0

# ============ 串口 (Linux) ============
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
FRAME_HEAD = [0xAA, 0x55]
FRAME_TAIL = 0x0D
ACK_BYTE = 0x06
RETRY_TIMES = 3
TIMEOUT_S = 0.5

# ============ 状态机 ============
GRAB_DONE_WAIT_S = 3.0
CAMERA_WARMUP_S = 1.0
