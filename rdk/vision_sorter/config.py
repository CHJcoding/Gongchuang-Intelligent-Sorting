# -*- coding: utf-8 -*-
"""
全局配置：RDK X5 板端版本。现场只改这一个文件。
"""

# ============ 相机 (C100 USB, Linux V4L2) ============
# 板端实测（10-21）：hobot_usb_cam 实际出图 640x480，MJPG。原配置写的 1280x720 与现场不符。
CAM_INDEX = 0              # /dev/video0
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# ============ YOLOv5s 检测 (地平线 BPU .bin 模型) ============
# 基线（2026-10-21 实读核实）：best.pt = myYOLOv5s，7 类，训练 imgsz=640；
# 实际导出 ONNX/bin 为 672×672 + NHWC raw 三尺度（36 = 3×(5+7)）。
# 决策 D2：暂用板上已跑通的 672，不重导出；现场精度不足再回 640。
YOLO_BIN_MODEL = "weights/yolov5_best.bin"   # 板上实际文件名以板端为准
YOLO_IMGSZ = 672            # 必须与 .bin 输入尺寸一致，否则维度报错
YOLO_NUM_CLASSES = 7
# 板端实测（10-21）：0.25 时一帧能吐 45 个重叠误检；0.50 -> 2~4 个；0.80 -> 稳定 1 个。
# 现场只有圆锥一种物料，统一用 0.80。
YOLO_CONF_THRES = 0.80
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

# 红色色相跨 0°，第二段区间（与 COLOR_RANGES_HSV["red"] 取并集）
RED_HSV_LOWER2 = [(170, 90, 50), (179, 255, 255)]

# 票数占比下限：最高票颜色占 ROI 面积低于此值则判失败（阈值不对时兜底）
MIN_PIXEL_RATIO = 0.15

# ============ 手眼标定 ============
# [x_mm, y_mm] = M @ [u, v, 1]，标定后填这里
HAND_EYE_M = [
    [0.038, 0.000, -24.5],
    [0.000, -0.038, 410.0],
]
Z_OFFSET = 0.0

# ============ ROS2 话题（D1 架构：订阅官方 dnn_node_example）============
# 板端实测：/hobot_dnn_detection 类型 ai_msgs/PerceptionTargets，Target 字段是 rois
DET_TOPIC = "/hobot_dnn_detection"
IMG_TOPIC = "/image"                 # sensor_msgs/CompressedImage (jpeg)
RESULT_TOPIC = "/vision/result"      # 与 chassis_driver.py 的订阅名对齐

# ============ 分拣触发 ============
AUTO_TRIGGER = True        # True: 检测到就抓（冷却 SORT_COOLDOWN_S）；False: 等 /sort/start
SORT_COOLDOWN_S = 2.0      # 两次下发的最小间隔
SAME_TARGET_LOCK_PX = 45   # 已下发物料在多少像素内视为"同一个"，不重复抓
SAME_TARGET_LOCK_S = 20.0  # 锁定最长保持时间，超时后允许再抓
COLOR_MIN_CONF = 0.60      # HSV 颜色投票置信度下限，低于则本帧放弃
SERIAL_SIM = False         # True: 不连下位机，只打印指令（无 STM32 时调试用）

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
