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

# ============ 物料编号（与 STM32 侧 10x10 表对齐，2026-10-26）============
# STM32 侧是一张 10 行 x 10 列的表，物料编号 = 形状*10 + 颜色：
#
#          红0  橙1  黄2  绿3  棕4  蓝5  紫6  品7  黑8  白9
#   0 长方体     0    1    2    3    4    5    6    7    8    9
#   1 镂空正方体 10   11   12   13   14   15   16   17   18   19
#   2 球        20   21   22   23   24   25   26   27   28   29
#   3 三棱柱    30   31   32   33   34   35   36   37   38   39
#   4 圆锥      40   41   42   43   44   45   46   47   48   49
#   5 正方体    50   51   52   53   54   55   56   57   58   59
#   6 圆柱      60   61   62   63   64   65   66   67   68   69
#   7 五棱柱    70   71   72   73   74   75   76   77   78   79
#   8 六棱柱    80   81   82   83   84   85   86   87   88   89
#   9 三棱锥    90   91   92   93   94   95   96   97   98   99
#
# 我们的检测模型只有 7 类，缺 0(长方体) / 5(正方体) / 8(六棱柱)；
# BQ 训练时叫"半球"，在 32 的表里落到 2(球)。下面的映射就是"模型类 -> 32 形状号"。
SHAPE_IDS = {
    "LK": 1,     # 镂空正方体
    "BQ": 2,     # 半球 -> 球
    "SLZ": 3,    # 三棱柱
    "YZHU": 4,   # 圆锥
    "YZ": 6,     # 圆柱
    "WLZ": 7,    # 五棱柱
    "SMT": 9,    # 正四面体 -> 三棱锥
}
# 模型里没有、但 32 的表里有的形状（识别不到就会把它们判成别的类，需谨慎）
SHAPE_IDS_MISSING = {0: "长方体", 5: "正方体", 8: "六棱柱"}

# ============ HSV 颜色阈值 ============
# 用 color_tuner.py 在现场补光下调好后填这里。
# (lower_HSV, upper_HSV), H:0-179, S:0-255, V:0-255
# 2026-10-24：删除 magenta（现场没有品红物料）。注意 color_id 用下面的
# COLOR_IDS 显式固定，不随本字典增删而漂移，否则 STM32 侧编号会错位。
COLOR_RANGES_HSV = {
    # 2026-10-26 板端实测重定：实物是"粉彩/浅色"塑料件，不是鲜艳色块。
    # 相机 brightness=50 时实测物体 S 中位数只有 30~35，全部落在彩色区
    # (S>=70) 和白色区 (S<=30) 之间的【空档】里，于是被判成 判色失败/白色。
    # 现在 brightness 由 cam_watchdog.sh 改成 32、saturation 改成 128，
    # 典型 S 升到 60~146；再把彩色下限从 70 降到 40 兜住浅色件。
    # 另外补掉两处空档：紫色上限 155->168（粉紫落在 156~169）、白 V 200->170。
    "red":      [(0,   40,  50),  (10,  255, 255)],
    # 2026-10-26：STM32 的颜色表里 4 号是"棕"，不是"青"，所以把 cyan 段让给 brown。
    # 棕色 = 暗橙：色相跟橙一样(11~22)，只是明度低（V<=165），所以靠 V 分开：
    #   orange 抬到 V>=166，brown 收 V 40~165。现场有真橙色件时再按实测校准。
    "brown":    [(5,   70,  40),  (25,  255, 165)],
    "orange":   [(11,  40,  166), (22,  255, 255)],
    "yellow":   [(23,  40,  50),  (35,  255, 255)],
    "green":    [(36,  40,  40),  (85,  255, 255)],
    "blue":     [(86,  40,  40),  (125, 255, 255)],
    "purple":   [(126, 40,  40),  (152, 255, 255)],
    # 粉色：2026-10-26 现场实物实测 H=156 S=65 V=230（浅粉塑料件）。
    # 色相带 153~168 与 purple(<=152) / red 第二段(169~179) 严格不重叠，
    # 靠 V>=90 把"深紫"和"暗红"排除在外，靠 S>=35 兜住浅色。
    "pink":     [(153, 35,  90),  (168, 255, 255)],
    "black":    [(0,   0,   0),   (179, 60,  60)],
    "white":    [(0,   0,   170), (179, 35,  255)],
}

# 下位机 color_id 映射（发到 STM32 的字节值）——与字典顺序解耦，固定不变。
# 顺序完全照 STM32 的 10 列表头：0红 1橙 2黄 3绿 4棕 5蓝 6紫 7品 8黑 9白。
# 7 号（品红/粉）2026-10-26 现场实测 H=156 → 就是粉色物料。
COLOR_IDS = {
    "red": 0, "orange": 1, "yellow": 2, "green": 3, "brown": 4,
    "blue": 5, "purple": 6, "pink": 7,
    "black": 8, "white": 9,
}
COLOR_NAMES_CN = {
    "red": "红", "orange": "橙", "yellow": "黄", "green": "绿", "brown": "棕",
    "blue": "蓝", "purple": "紫", "pink": "品红", "black": "黑", "white": "白",
}

# 红色色相跨 0°，第二段区间（与 COLOR_RANGES_HSV["red"] 取并集）
RED_HSV_LOWER2 = [(169, 40, 50), (179, 255, 255)]

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

# 彩色像素掩码（2026-10-26 实测加）：S、V 都够高才算"有颜色"。
# 深色台面会被 black 区间吃进去、白纸背景被 white 吃进去，把真颜色的票数稀释
# 到 conf≈0.60（正好压在 COLOR_MIN_CONF 上，随机判色失败）。彩色像素占比超过
# CHROMATIC_MIN_FRAC 时只在彩色像素上投票；低于它（黑件/白件/灰件）仍用全像素。
CHROMATIC_S_MIN = 40
CHROMATIC_V_MIN = 50
CHROMATIC_MIN_FRAC = 0.20

# ============ 抓取点精修（2026-10-26 加）============
# 下发给 STM32 的 (x,y) 取哪个点：
#   "centroid" 颜色轮廓质心（默认）：抗形状不对称，物料的几何重心就在这儿
#   "rect"     最小外接矩形中心：抗边缘毛刺、局部缺色
#   "bbox"     YOLO 框中心（等于不做精修）
# 关闭：GRASP_REFINE = False（整段逻辑跳过，行为回到老版本"框中心"）
GRASP_REFINE = True
GRASP_POINT_MODE = "centroid"
GRASP_MASK_MARGIN = 0.12    # 框向外扩的比例，给"框偏小、物料出框"留余量
GRASP_MIN_FILL = 0.10       # 轮廓面积/框面积 低于此值视为判色可疑，回退框中心
# 掩膜可信度门槛（不满足就回退框中心，绝不静默下发坏点）
GRASP_MAX_FILL = 1.60       # 轮廓面积/框面积 高于此值 = 糊到背景上了
GRASP_MIN_SOLIDITY = 0.72   # 轮廓面积/凸包面积 低于此值 = 掩膜是散点/环形
# 覆盖率别定太高：三角形/四面体俯视投影本来就只有外接框的一半（实测绿色
# 三棱柱 0.54，定 0.55 会把它误杀）。防"只切到碎片"靠下面两项。
GRASP_MIN_COVERAGE = 0.40   # 掩膜/贴合框面积 低于此值 = 贴合框里只有一小块是物体
GRASP_MAX_CENTER_OFF = 0.35 # 质心离框中心距离/框短边 高于此值 = 掩膜偏到一侧了
GRASP_RATIO_LO = 0.45       # 贴合框宽(高)/框宽(高) 的合理区间
GRASP_RATIO_HI = 1.15
# 自适应收窄：全局颜色区间把背景也框进来时（蓝色最常发生），
# 用框内像素的中位 HSV 重建一个更窄的窗口，把背景挤出去
GRASP_ADAPTIVE = True
GRASP_ACHROMATIC_OTSU = True  # 白/黑物料：白物体 vs 白台面，颜色区间分不开，用明度 Otsu 兜底
GRASP_H_TOL = 10            # 色相半宽（0~180 制，10 约等于 ±20°）
GRASP_S_TOL = 70            # 饱和度半宽
GRASP_V_TOL = 70            # 明度半宽
SERIAL_SIM = False         # True: 不连下位机，只打印指令（无 STM32 时调试用）
SCAN_ALL = False           # True: 每秒把所有目标+判色结果打一行 [扫描]（台架联调用，只影响日志）

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

# ============ 安全策略（上线前必须确认）============
# 安全闸：手眼没标定时【禁止下发坐标】。
#
# 为什么需要：没标定时算出来的坐标来自 config.HAND_EYE_M 的占位值，是假的。
# 以前只是打 ERROR 日志然后照发不误 —— 龙门装好以后，一旦忘了标定就直接乱走，
# 这种事故代价远大于少抓一颗料。
#
# 想在没有 hand_eye.json 的情况下演示下发（台架调试）：
#   python3 sort_vision_node.py --ros-args -p allow_uncalibrated:=true
SAFETY_REQUIRE_CALIB = True
ALLOW_UNCALIBRATED = False     # 等价于命令行 -p allow_uncalibrated:=true，改这里也行

# 串口打不开时的策略。
#   False（生产推荐）：进"串口故障"状态，拒绝下发并每 5 秒重试，日志明确报错
#   True （台架方便）  ：静默进模拟模式，照常打印指令
# 注意：True 的问题是"看起来在抓、其实根本没发给 STM32"，现场很容易误判。
SERIAL_SIM_ON_FAIL = False
# 串口掉了以后多久重试打开一次（插拔 USB 转串口后能自愈）
SERIAL_REOPEN_S = 5.0
