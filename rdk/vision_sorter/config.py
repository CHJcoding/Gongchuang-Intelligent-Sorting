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
# 2026-10-24：删除 magenta（现场没有品红物料）。注意 color_id 用下面的
# COLOR_IDS 显式固定，不随本字典增删而漂移，否则 STM32 侧编号会错位。
COLOR_RANGES_HSV = {
    "red":      [(0,   90,  50),  (10,  255, 255)],
    "orange":   [(11,  90,  50),  (22,  255, 255)],
    "yellow":   [(23,  90,  50),  (35,  255, 255)],
    "green":    [(36,  70,  40),  (85,  255, 255)],
    "cyan":     [(86,  70,  40),  (100, 255, 255)],
    "blue":     [(101, 70,  40),  (125, 255, 255)],
    "purple":   [(126, 70,  40),  (155, 255, 255)],
    "black":    [(0,   0,   0),   (179, 60,  60)],
    "white":    [(0,   0,   200), (179, 30,  255)],
}

# 下位机 color_id 映射（发到 STM32 的字节值）——与字典顺序解耦，固定不变。
# 7 号原为 magenta，已废弃并【保留不再使用】，避免 black/white 编号前移导致
# STM32 侧颜色错位。若将来要复用品红，直接用 7 号即可。
COLOR_IDS = {
    "red": 0, "orange": 1, "yellow": 2, "green": 3, "cyan": 4,
    "blue": 5, "purple": 6,
    # 7 = 预留（原 magenta，已删除）
    "black": 8, "white": 9,
}

# 红色色相跨 0°，第二段区间（与 COLOR_RANGES_HSV["red"] 取并集）
RED_HSV_LOWER2 = [(170, 90, 50), (179, 255, 255)]

# 票数占比下限：最高票颜色占 ROI 面积低于此值则判失败（阈值不对时兜底）
MIN_PIXEL_RATIO = 0.15

# ============ 手眼标定 ============
# [x_mm, y_mm] = M @ [u, v, 1]
#
# 矩阵来源优先级（见 hand_eye.py）：
#   1. 环境变量 GC_HAND_EYE_JSON
#   2. 本文件的 HAND_EYE_JSON（非空时）
#   3. vision_sorter/hand_eye.json（tools/planar_calib.py 默认输出到仓库根）
#   4. 当前工作目录 hand_eye.json
#   5. /root/vision_ws/hand_eye.json（板端约定位置）
#   6. 以上都没有 -> 用下面的 HAND_EYE_M 占位值，hand_eye.CALIBRATED=False
#
# ⚠ 下面这组是【占位值】，按相机高度猜的，不是实测：
#   - 比例 0.038 mm/px（约 26 px/mm），y 轴取负（画面向下 -> 坐标减小）
#   - 只能用来打通链路，标定前别拿它驱动龙门实动！
# 标定方法见 rdk/vision_sorter/BOARD_README.md 第九节。
HAND_EYE_M = [
    [0.038, 0.000, -24.5],
    [0.000, -0.038, 410.0],
]
HAND_EYE_M_IS_PLACEHOLDER = True   # 标定后由 hand_eye.json 覆盖，本行不用改

# 想固定 JSON 路径就填绝对路径；留空则按上面的优先级自动找
HAND_EYE_JSON = ""       # 例："/root/vision_ws/hand_eye.json"

Z_OFFSET = 0.0

# ============ 相机内参 / 去畸变（见 camera_model.py）============
# C100 是广角镜头，实测畸变：画面中心 0.5px，内圈 6~30px，四角 130~200px。
# 手眼矩阵是仿射的，吸收不了这种非线性畸变，所以【像素→毫米之前】必须先去畸变，
# 否则边缘物料会偏十几毫米。
#
# 内参文件来源顺序：
#   1. 环境变量 GC_CAMERA_INTRINSIC_JSON
#   2. 本文件的 CAMERA_INTRINSIC_JSON（非空时）
#   3. vision_sorter/camera_intrinsic.json
#   4. 当前工作目录 camera_intrinsic.json
#   5. /root/vision_ws/camera_intrinsic.json（板端约定位置）
# 生成方法：python tools/camera_intrinsic_calib.py（棋盘格拍 15 张以上，
# 要覆盖画面四角，否则模型在边缘会外推失真）
#
# ⚠ 这条开关必须和手眼标定时的状态一致：
#    planar_calib.py 找到内参就自动去畸变后再检角点（hand_eye.json 里记
#    "undistorted": true/false）。节点启动会比对，不一致直接 ERROR。
UNDISTORT_ENABLE = True      # 找到 camera_intrinsic.json 才真正生效
CAMERA_INTRINSIC_JSON = ""   # 想固定路径填绝对路径；留空自动找

# 图像尺寸（去畸变、越界保护、几何校验用）
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

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
ACK_BYTE = 0x06            # 帧级 ACK（"收到帧了"）—— 本方案默认不要求，见 SEND_WAIT_ACK
RETRY_TIMES = 3            # 仅 SEND_WAIT_ACK=True 时生效
TIMEOUT_S = 0.5            # 仅 SEND_WAIT_ACK=True 时生效

# ============ 抓取完成判定（一发一抓的流控）============
# 单向发送时 RDK 不知道 STM32 抓完没有，必须选一种"解锁"依据：
#   "byte"  : STM32 抓取动作结束时回 1 字节 GRAB_DONE_BYTE
#             —— 这是【动作级完成信号】，不是帧级 ACK；不需要重试、不需要握手。
#   "timer" : 纯开环，按 GRAB_CYCLE_S 定时解锁，STM32 一行代码都不用加。
GRAB_DONE_MODE = "timer"
GRAB_CYCLE_S = 6.0         # timer 模式：一次完整抓取循环时长（现场实测后调准）
GRAB_DONE_BYTE = 0x06      # byte 模式：完成字节（与 ACK_BYTE 同值也可，语义不同）
GRAB_TIMEOUT_S = 12.0      # 兜底：超时强制解锁，防止永久卡死
BLANK_AFTER_DONE_S = 0.8   # 抓完后再屏蔽这么久（等机械臂收回、离开视野）
GRAB_BLANK_DETECT = True   # 抓取期间忽略检测：机械臂进入视野会被误检成物料
SEND_WAIT_ACK = False      # False = 发完就走，不等帧级 ACK（你现在的方案）

# ---- 抓取结果状态：2 字节回收协议（已实现，但先不启用）----
# 1 字节协议只表示"抓完了"；2 字节协议还能带回"抓成功没有"：
#     下行不需要改，上行变成：0x06 + status
#         status: 0x00=成功  0x01=空抓/失败  0x02=急停  0x03=其它错误
# 前提：机械臂得有限位开关/夹持检测/真空压力开关之类，否则拿不到成功与否。
# 启用方法：把 GRAB_STATUS_ENABLE 改成 True，并把 GRAB_DONE_MODE 改成 "byte"。
GRAB_STATUS_ENABLE = False        # ← 现在注释着不启用
# GRAB_STATUS_ENABLE = True       # ← 等 STM32 固件真的发 2 字节了，换成这一行
GRAB_STATUS_OK = 0x00
GRAB_STATUS_FAIL = 0x01
GRAB_STATUS_ESTOP = 0x02
GRAB_STATUS_OTHER = 0x03
GRAB_STATUS_NAMES = {
    GRAB_STATUS_OK: "成功",
    GRAB_STATUS_FAIL: "空抓/失败",
    GRAB_STATUS_ESTOP: "急停",
    GRAB_STATUS_OTHER: "其它错误",
}

# ============ 夹爪坐标回传（方案B：STM32 -> RDK，已实现，默认不启用）============
# 用途：手眼标定要"真值"。RDK 侧只有像素，坐标真值只在 STM32 手里。
# 让 STM32 周期性（或按请求）把当前夹爪坐标回上来，标定就不用靠人眼抄数。
#
# 上行帧（6 字节，与下行同一套 CRC）：
#   [0]   0x07        帧头
#   [1:3] x_int16 LE  当前 X(mm)，有符号
#   [3:5] y_int16 LE  当前 Y(mm)，有符号
#   [5]   crc8        校验：对前 5 字节（0x07 + x + y）算，poly 0x07
#
# STM32 侧注意事项：
#   - 和已有的 0x06 完成字节走同一条串口，交替发出即可，RDK 侧统一解析
#   - 建议 20Hz 以上；RDK 侧只保留最新一帧，丢帧无所谓
#   - 坐标必须是【和下行帧同一个坐标系】，否则标定结果会整体平移
#
# ⚠ 固件没做之前必须保持 False：否则 RDK 收不到 0x07，标定工具会一直等超时。
POS_ENABLE = False                # ← 现在注释着不启用
# POS_ENABLE = True               # ← 等 STM32 固件会发 0x07 帧了，换成这一行
POS_FRAME_HEAD = 0x07             # 坐标帧帧头
POS_FRAME_LEN = 6                 # 坐标帧总长度
POS_TOPIC = "/gantry/pos"         # POS_ENABLE=True 时发布的调试话题（JSON 字符串）

# ============ 状态机 ============
GRAB_DONE_WAIT_S = 3.0
CAMERA_WARMUP_S = 1.0
