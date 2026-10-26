# -*- coding: utf-8 -*-
"""
视觉分拣节点（D1 架构：不自跑 BPU，订阅官方 dnn_node_example 的检测结果）。

数据流：
    hobot_usb_cam --/hbmem_img--> dnn_node_example
                                     --/hobot_dnn_detection--> 本节点
    hobot_usb_cam --/image(CompressedImage)--> 本节点（取同帧 HSV 颜色）
    本节点 --/vision/result(JSON)--> 底盘/机械臂节点
    本节点 --串口(AA55+CRC8)--> STM32

关键事实（2026-10-21 板端实测）：
  * /hobot_dnn_detection 类型 ai_msgs/PerceptionTargets，Target 字段是 rois（没有 tracks）
  * ROI 坐标就是原图 640x480 像素坐标，无需任何缩放（实测误差 1~3px）
  * 板端 dnn_node 前处理是拉伸（非 letterbox），精度有损，后续建议重建 640x480 bin
"""
import json
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from ai_msgs.msg import PerceptionTargets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from color_classifier import classify_color
import hand_eye
import camera_model
import safety
from hand_eye import pixel_to_robot


# ============================ 串口工作线程 ============================
class SerialWorker(threading.Thread):
    """串口发送放到独立线程，绝不阻塞 ROS 回调。"""

    def __init__(self, comm, node):
        super().__init__(daemon=True)
        self.comm = comm
        self.node = node
        self.q = queue.Queue(maxsize=8)
        self.running = True
        self.stop_event = threading.Event()

    def submit(self, shape, color, x_mm, y_mm):
        try:
            self.q.put_nowait((shape, color, x_mm, y_mm))
            return True
        except queue.Full:
            self.node.get_logger().warn("串口队列满，丢弃本次指令")
            return False

    def run(self):
        while self.running:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            shape, color, x_mm, y_mm = item
            t0 = time.time()
            ok = self.comm.send_grab(shape, color, x_mm, y_mm)
            dt = (time.time() - t0) * 1000
            if ok:
                self.node.get_logger().info(
                    "串口已发出 %.1fms | %s/%s (%d,%d) | 等待抓取完成(%s模式)" %
                    (dt, shape, color, round(x_mm), round(y_mm),
                     getattr(self.node, "grab_done_mode",
                             getattr(C, "GRAB_DONE_MODE", "timer"))))
            else:
                self.node.get_logger().error(
                    "串口发送失败 %.1fms | %s/%s" % (dt, shape, color))
                self.node.on_send_failed()

    def stop(self):
        self.running = False
        self.stop_event.wait(0.5)


# ============================ 主节点 ============================
class SortVisionNode(Node):
    def __init__(self):
        super().__init__("sort_vision_node")

        # ---- 参数 ----
        self.declare_parameter("auto_trigger", bool(getattr(C, "AUTO_TRIGGER", True)))
        self.declare_parameter("serial_sim", bool(getattr(C, "SERIAL_SIM", False)))
        # 安全闸：台架演示时可以显式绕过（默认不绕）
        self.declare_parameter("allow_uncalibrated",
                               bool(getattr(C, "ALLOW_UNCALIBRATED", False)))
        # 抓取窗口时长：现场实测完可以不改代码，直接命令行覆盖
        self.declare_parameter("grab_cycle_s",
                               float(getattr(C, "GRAB_CYCLE_S", 6.0)))
        self.declare_parameter("grab_timeout_s",
                               float(getattr(C, "GRAB_TIMEOUT_S", 12.0)))
        self.declare_parameter("grab_done_mode",
                               str(getattr(C, "GRAB_DONE_MODE", "timer")))
        self.auto_trigger = bool(self.get_parameter("auto_trigger").value)
        self.serial_sim = bool(self.get_parameter("serial_sim").value)
        self.allow_uncalibrated = bool(
            self.get_parameter("allow_uncalibrated").value)
        self.grab_cycle_s = float(self.get_parameter("grab_cycle_s").value)
        self.grab_timeout_s = float(self.get_parameter("grab_timeout_s").value)
        self.grab_done_mode = str(
            self.get_parameter("grab_done_mode").value).lower()
        self.require_calib = bool(getattr(C, "SAFETY_REQUIRE_CALIB", True))

        # ---- 帧缓存（按时间戳配对，避免用到过期画面）----
        self.frames = deque(maxlen=20)
        self.frame_lock = threading.Lock()

        # ---- 状态 ----
        self.armed = self.auto_trigger     # 是否允许触发一次分拣
        self.busy = False                  # 是否处于一次抓取窗口（期间屏蔽检测）
        self.grab_start = 0.0              # 本次抓取开始时刻
        self.blank_until = 0.0             # 检测屏蔽截止时刻（抓取中 + 收臂期）
        self.last_fire_t = 0.0
        self.lock_center = None            # 已下发目标的位置，防重复抓同一物料
        self.lock_time = 0.0
        self.stat_det = 0                  # 收到检测消息数
        self.stat_hit = 0                  # 含有效目标的消息数
        self.stat_fire = 0                 # 实际下发次数
        self.stat_blank = 0                # 因抓取窗口被屏蔽的检测消息数
        self.stat_color_fail = 0
        self.stat_blocked = 0              # 被安全闸拦下的次数
        self.stat_send_fail = 0            # 串口写失败次数
        self.stat_edge = 0                 # 目标落在去畸变覆盖区外的次数
        self.last_reopen_t = 0.0           # 上次尝试重连串口的时刻

        # ---- 发布 ----
        self.pub_result = self.create_publisher(String, C.RESULT_TOPIC, 10)
        self.pub_state = self.create_publisher(String, "/sort/state", 10)
        self.pub_debug = self.create_publisher(String, "/sort/debug", 10)
        # 夹爪坐标回显（POS_ENABLE=True 才建话题，避免没开时挂个空话题）
        self.pub_pos = None
        if getattr(C, "POS_ENABLE", False):
            self.pub_pos = self.create_publisher(
                String, getattr(C, "POS_TOPIC", "/gantry/pos"), 10)

        # ---- 订阅 ----
        img_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, C.IMG_TOPIC,
                                 self.on_image, img_qos)
        self.create_subscription(PerceptionTargets, C.DET_TOPIC,
                                 self.on_detection, 10)
        self.create_subscription(Bool, "/sort/start", self.on_start, 10)
        self.create_subscription(Bool, "/sort/reset", self.on_reset, 10)

        # 标定热加载：标定完不用重启节点
        self.create_service(Trigger, "/sort/reload_calib", self.on_reload_calib)

        # ---- 串口 ----
        self.comm = None
        self.serial_worker = None
        self.serial_ok = bool(self.serial_sim)     # 模拟模式视为"可用"
        self._init_serial()

        # ---- 定时器：状态心跳 + 统计 ----
        self.create_timer(5.0, self.on_timer)
        # 抓取窗口看门狗：判定"抓完了"并解除屏蔽（10Hz）
        self.create_timer(0.1, self.on_grab_watchdog)
        self.get_logger().info(
            "sort_vision_node 启动 | det=%s img=%s result=%s | auto=%s sim=%s | "
            "完成判定=%s(cycle=%.1fs timeout=%.1fs)"
            % (C.DET_TOPIC, C.IMG_TOPIC, C.RESULT_TOPIC,
               self.auto_trigger, self.serial_sim,
               self.grab_done_mode, self.grab_cycle_s, self.grab_timeout_s))

        # 手眼标定状态：没标定就大声报，别让人拿假坐标去驱动龙门
        hand_eye.warn_if_uncalibrated(self.get_logger())
        # 去畸变：广角镜头边缘偏十几毫米，必须和标定时状态一致
        camera_model.warn_if_disabled(self.get_logger())
        try:
            camera_model.check_hand_eye_consistency(
                getattr(hand_eye, "META", {}) or {}, self.get_logger())
        except Exception as e:                        # noqa: BLE001
            self.get_logger().warn("去畸变/标定一致性检查失败: %s" % (e,))
        # 安全闸状态：一眼看出"现在到底能不能动"
        # 注意：两个调用必须在不同代码行 —— rclpy 按调用点缓存 severity，
        # 同一行用两种 severity 会抛 ValueError。
        for _level, _line in safety.describe_gates(
                self.require_calib, self.allow_uncalibrated,
                hand_eye.is_calibrated(), self.serial_ok, self.serial_sim,
                getattr(C, "SERIAL_SIM_ON_FAIL", False)):
            if _level == "warn":
                self.get_logger().warn(_line)
            else:
                self.get_logger().info(_line)
        if getattr(C, "POS_ENABLE", False):
            self.get_logger().info(
                "夹爪坐标回传已开启 | 帧头=0x%02X 长度=%d 话题=%s"
                % (getattr(C, "POS_FRAME_HEAD", 0x07),
                   getattr(C, "POS_FRAME_LEN", 6),
                   getattr(C, "POS_TOPIC", "/gantry/pos")))

    # ---------------- 串口 ----------------
    def _init_serial(self):
        if self.serial_sim:
            self.get_logger().warn("serial_sim=True：不连下位机，只打印指令")
            self.serial_ok = True
            return
        try:
            from comm import Comm
            self.comm = Comm()
            self.comm.open()
            self.serial_worker = SerialWorker(self.comm, self)
            self.serial_worker.start()
            self.serial_ok = True
            self.get_logger().info("串口已打开 %s @%d" %
                                   (C.SERIAL_PORT, C.SERIAL_BAUD))
        except Exception as e:
            self.comm = None
            self.serial_worker = None
            self.serial_ok = False
            if getattr(C, "SERIAL_SIM_ON_FAIL", False):
                # 台架模式：打不开就当模拟，照常打印
                self.get_logger().warn(
                    "串口打开失败（%s: %s），按 SERIAL_SIM_ON_FAIL=True "
                    "退化为模拟模式（不会真的发给 STM32）" %
                    (type(e).__name__, e))
                self.serial_sim = True
                self.serial_ok = True
            else:
                # 生产模式：明确报错，拒绝下发，并周期性重试
                self.get_logger().error(
                    "串口打开失败（%s: %s）—— 进入【串口故障】状态，拒绝下发。"
                    "检查 %s 是否存在（ls /dev/ttyUSB*）、是否被占用，"
                    "插好后会自动重连（每 %.0fs 试一次）" %
                    (type(e).__name__, e, C.SERIAL_PORT,
                       getattr(C, "SERIAL_REOPEN_S", 5.0)))
                self._publish_state("SERIAL_ERROR")

    def _try_reopen_serial(self):
        """串口掉了以后周期性重连（USB 转串口热插拔能自愈）。"""
        if self.serial_sim or self.serial_ok:
            return
        now = time.time()
        period = float(getattr(C, "SERIAL_REOPEN_S", 5.0))
        if now - self.last_reopen_t < period:
            return
        self.last_reopen_t = now
        try:
            from comm import Comm
            if self.comm is None:
                self.comm = Comm()
            if self.comm.reopen():
                self.serial_ok = True
                self.serial_worker = SerialWorker(self.comm, self)
                self.serial_worker.start()
                self.get_logger().info("串口已恢复 %s @%d，继续正常下发" %
                                       (C.SERIAL_PORT, C.SERIAL_BAUD))
                self._publish_state("SERIAL_OK")
            else:
                self.get_logger().warn("串口仍打不开，%.0fs 后重试" % period,
                                       throttle_duration_sec=15.0)
        except Exception as e:                        # noqa: BLE001
            self.comm = None
            self.get_logger().warn("串口重连异常: %s: %s"
                                   % (type(e).__name__, e),
                                   throttle_duration_sec=15.0)

    # ---------------- 抓取窗口：如何判定"抓完了" ----------------
    def on_send_failed(self):
        """串口发送失败：立即解除抓取窗口，并标记串口掉线等重连。"""
        self.stat_send_fail += 1
        self.serial_ok = False
        self.busy = False
        self.blank_until = 0.0
        self._publish_state("SEND_FAIL")
        self.get_logger().error(
            "串口写失败（USB 转串口掉了？），暂停下发并开始重连",
            throttle_duration_sec=5.0)

    def on_grab_watchdog(self):
        """
        10Hz 检查抓取窗口是否该结束。

          timer 模式：发出去后按 GRAB_CYCLE_S 定时解锁（STM32 无需任何回报）
          byte  模式：等 STM32 在动作结束时回的 GRAB_DONE_BYTE，超时则兜底解锁

        顺带把 STM32 回传的夹爪坐标取出来发话题（POS_ENABLE=True 时）。
        注意：坐标回传和完成字节共用同一条串口，必须都在这里轮询，
        由 comm 的统一解析泵拆分，谁也不会吃掉谁的字节。
        """
        self._poll_gantry_pos()
        if not self.busy:
            return
        now = time.time()
        elapsed = now - self.grab_start
        mode = self.grab_done_mode

        if mode == "byte":
            if self.comm is not None:
                if getattr(C, "GRAB_STATUS_ENABLE", False):
                    # 2 字节协议：0x06 + status（当前未启用）
                    seen, status = self.comm.poll_done_status()
                    if seen:
                        self._end_grab("DONE", elapsed, status)
                        return
                elif self.comm.poll_done():
                    # 1 字节协议：只表示"抓完了"
                    self._end_grab("DONE", elapsed)
                    return
            if elapsed >= self.grab_timeout_s:
                self.get_logger().warn(
                    "抓取完成字节超时(%.1fs)，强制解锁（下位机是否没发完成信号？）"
                    % elapsed)
                self._end_grab("TIMEOUT", elapsed)
        else:
            if elapsed >= self.grab_cycle_s:
                self._end_grab("TIMER", elapsed)

    def _poll_gantry_pos(self):
        """取 STM32 回传的夹爪坐标并发话题（POS_ENABLE=False 时直接返回）。"""
        if not getattr(C, "POS_ENABLE", False) or self.comm is None:
            return
        try:
            pos = self.comm.poll_position()
        except Exception:
            return
        if pos is None or self.pub_pos is None:
            return
        payload = {"x": pos[0], "y": pos[1], "seq": self.comm.position_seq}
        self.pub_pos.publish(String(data=json.dumps(payload,
                                                   ensure_ascii=False)))

    def _end_grab(self, reason, elapsed, status=None):
        """
        结束抓取窗口：解除 busy、开始收臂屏蔽、允许抓下一个物料。
        status 仅在 2 字节协议（GRAB_STATUS_ENABLE=True）下由下位机回报。
        """
        self.busy = False
        self.lock_center = None            # 物料已被搬走，解除位置锁定
        self.blank_until = time.time() + getattr(C, "BLANK_AFTER_DONE_S", 0.8)
        if status is not None:
            ok = (status == getattr(C, "GRAB_STATUS_OK", 0x00))
            label = getattr(C, "GRAB_STATUS_NAMES", {}).get(
                status, "UNKNOWN(0x%02X)" % status)
            self.get_logger().info(
                "抓取结束[%s] 用时 %.2fs 下位机回报=%s，%.1fs 后恢复检测"
                % (reason, elapsed, label,
                   getattr(C, "BLANK_AFTER_DONE_S", 0.8)))
            self._publish_state("GRAB_OK" if ok else "GRAB_FAIL")
            return
        self.get_logger().info(
            "抓取结束[%s] 用时 %.2fs，%.1fs 后恢复检测"
            % (reason, elapsed, getattr(C, "BLANK_AFTER_DONE_S", 0.8)))
        self._publish_state("GRAB_" + reason)

    # ---------------- 订阅回调 ----------------
    def on_image(self, msg):
        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().warn("图像解码失败: %s" % e)
            return
        if frame is None:
            return
        key = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self.frame_lock:
            self.frames.append((key, frame))

    def _pick_frame(self, stamp):
        key = stamp.sec * 1_000_000_000 + stamp.nanosec
        with self.frame_lock:
            if not self.frames:
                return None
            best, best_dt = None, 1 << 62
            for k, f in self.frames:
                dt = abs(k - key)
                if dt < best_dt:
                    best, best_dt = f, dt
            return best

    def on_detection(self, msg):
        self.stat_det += 1
        # 抓取窗口：机械臂/夹具会进入视野并被 YOLO 误检成物料，必须屏蔽
        if self.blank_until and time.time() < self.blank_until:
            self.stat_blank += 1
            return
        targets = []
        for t in msg.targets:
            rois = getattr(t, "rois", None)
            if rois is None:                       # 兼容旧版 tracks 结构
                rois = []
                for tr in (getattr(t, "tracks", None) or []):
                    rois.extend(list(getattr(tr, "rois", []) or []))
            for roi in rois:
                conf = float(roi.confidence)
                if conf < C.YOLO_CONF_THRES:
                    continue
                targets.append({
                    "shape": t.type,
                    "conf": conf,
                    "bbox": [int(roi.rect.x_offset), int(roi.rect.y_offset),
                             int(roi.rect.x_offset + roi.rect.width),
                             int(roi.rect.y_offset + roi.rect.height)],
                })
        if not targets:
            return
        self.stat_hit += 1
        targets.sort(key=lambda d: -d["conf"])

        # 调试：把当前所有有效目标发出去
        self.pub_debug.publish(String(data=json.dumps(
            {"n": len(targets), "top": targets[0]}, ensure_ascii=False)))

        if not (self.armed and not self.busy):
            return
        if time.time() - self.last_fire_t < C.SORT_COOLDOWN_S:
            return
        # 防重复：同一位置已下发过的物料，等它被搬走/移动后再抓
        top = targets[0]
        cx = (top["bbox"][0] + top["bbox"][2]) / 2.0
        cy = (top["bbox"][1] + top["bbox"][3]) / 2.0
        if self.lock_center is not None:
            d = ((cx - self.lock_center[0]) ** 2 +
                 (cy - self.lock_center[1]) ** 2) ** 0.5
            if d < getattr(C, "SAME_TARGET_LOCK_PX", 45) and \
                    time.time() - self.lock_time < getattr(C, "SAME_TARGET_LOCK_S", 20.0):
                return
            self.lock_center = None        # 物料移动了/超时，解除锁定
        self._fire(top, msg)

    def on_start(self, msg):
        if msg.data:
            self.armed = True
            self.busy = False
            self.blank_until = 0.0
            self.lock_center = None          # 显式启动 = 重新允许抓取当前目标
            self.get_logger().info("收到启动信号，进入待触发状态")
            self._publish_state("ARMED")

    def on_reset(self, msg):
        if msg.data:
            self.armed = False
            self.busy = False
            self.blank_until = 0.0
            self.get_logger().info("收到复位信号")
            self._publish_state("RESET")

    # ---------------- 一次分拣流程 ----------------
    def _begin_grab(self, now):
        """进入抓取窗口（屏蔽检测），按模式设定窗口上限。"""
        self.busy = True
        self.grab_start = now
        if self.grab_done_mode == "byte":
            window = self.grab_timeout_s
            if self.comm is not None:
                self.comm.drain()   # 清掉残留字节，避免把上一次的完成信号误当本次
        else:
            window = self.grab_cycle_s
        self.blank_until = now + window + getattr(C, "BLANK_AFTER_DONE_S", 0.8)
        self._publish_state("GRABBING")

    def _fire(self, target, det_msg):
        frame = self._pick_frame(det_msg.header.stamp)
        if frame is None:
            self.get_logger().warn("有检测但无图像帧，跳过（图像话题没收到？）")
            return
        x1, y1, x2, y2 = target["bbox"]
        H, W = frame.shape[:2]
        # 越界保护
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        if x2c - x1c < 10 or y2c - y1c < 10:
            self.get_logger().warn("ROI 太小/越界，跳过: %s" % (target["bbox"],))
            return

        roi = frame[y1c:y2c, x1c:x2c]
        color, color_conf = classify_color(roi)
        if color is None or color_conf < C.COLOR_MIN_CONF:
            self.stat_color_fail += 1
            self.last_fire_t = time.time()      # 失败也进冷却，避免 37Hz 刷屏
            self.get_logger().warn(
                "颜色识别失败 conf=%.2f shape=%s" %
                (color_conf, target["shape"]),
                throttle_duration_sec=2.0)
            self._publish_state("COLOR_FAIL")
            return

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # 广角畸变修正：必须做在"像素->毫米"之前。手眼矩阵是仿射的，
        # 吸收不了非线性畸变（C100 实测 r=0.64 处已修正 14px ≈ 2mm，越往外越大）。
        # planar_calib 标定时也是先去畸变再检角点，两边口径一致。
        cx_u, cy_u = camera_model.undistort_point(cx, cy)
        # 内参样本只覆盖到 valid_radius，超出就是外推：照发，但必须喊出来
        level, why_r = camera_model.check_point(cx, cy)
        if level == "ok":
            pass
        else:
            self.stat_edge += 1
            self.get_logger().warn(
                "目标在去畸变覆盖区之外：%s（原始像素 %.0f,%.0f）→ 该处毫米值不可靠，"
                "把物料往画面中间挪" % (why_r, cx, cy),
                throttle_duration_sec=10.0)
        x_mm, y_mm, z_mm = pixel_to_robot(cx_u, cy_u)

        # ---- 安全闸：没标定 / 串口掉了，一律不许下发 ----
        # 这两条以前只是打日志然后照发，龙门装好后一旦忘了标定就会满场乱走。
        block = safety.check_downlink(
            calibrated=hand_eye.is_calibrated(),
            allow_uncalibrated=self.allow_uncalibrated,
            serial_ok=self.serial_ok,
            serial_sim=self.serial_sim) if self.require_calib else None
        if block is None and not self.require_calib:
            # 标定闸被 config 全局关掉时，串口闸仍然要生效
            block = safety.check_downlink(
                calibrated=True, allow_uncalibrated=True,
                serial_ok=self.serial_ok, serial_sim=self.serial_sim)
        if block is not None:
            code, why = block
            self.stat_blocked += 1
            now_b = time.time()
            self.last_fire_t = now_b          # 进冷却，避免 30Hz 刷屏
            self._publish_state(code)
            self.get_logger().error(
                "拒绝下发[%s]：%s。坐标不发出（想台架演示见 "
                "BOARD_README「安全闸」一节）" % (code, why),
                throttle_duration_sec=10.0)
            # 仍然发一条 result 供调试，但显式标记 blocked，下游应丢弃
            self.pub_result.publish(String(data=json.dumps({
                "stamp": "%d.%09d" % (det_msg.header.stamp.sec,
                                      det_msg.header.stamp.nanosec),
                "shape": target["shape"], "color": color,
                "conf": round(target["conf"], 3),
                "pixel": [round(cx, 1), round(cy, 1)],
                "pixel_undist": [round(cx_u, 1), round(cy_u, 1)],
                "robot_mm": [x_mm, y_mm, z_mm],
                "blocked": code,
                "blocked_reason": why,
            }, ensure_ascii=False)))
            return

        now = time.time()
        self.last_fire_t = now
        self.stat_fire += 1
        self.lock_center = (cx, cy)        # 锁定该物料，防重复下发
        self.lock_time = now
        self._begin_grab(now)              # 进入抓取窗口，期间屏蔽检测
        if not self.auto_trigger:
            self.armed = False          # 单次触发模式：发一次就不再自动触发

        payload = {
            "stamp": "%d.%09d" % (det_msg.header.stamp.sec,
                                  det_msg.header.stamp.nanosec),
            "shape": target["shape"],
            "shape_cn": C.SHAPE_NAMES_CN.get(target["shape"], target["shape"]),
            "color": color,
            "conf": round(target["conf"], 3),
            "color_conf": color_conf,
            "bbox": target["bbox"],
            "pixel": [round(cx, 1), round(cy, 1)],
            "pixel_undist": [round(cx_u, 1), round(cy_u, 1)],
            "undistort": camera_model.is_active(),
            "robot_mm": [x_mm, y_mm, z_mm],
        }
        self.pub_result.publish(String(data=json.dumps(payload,
                                                      ensure_ascii=False)))
        if camera_model.is_active():
            self.get_logger().info(
                "目标 %s/%s conf=%.2f px=(%.0f,%.0f)->去畸变(%.0f,%.0f) -> "
                "(%.1f, %.1f)mm"
                % (payload["shape_cn"], color, payload["conf"],
                   cx, cy, cx_u, cy_u, x_mm, y_mm))
        else:
            self.get_logger().info(
                "目标 %s/%s conf=%.2f px=(%.0f,%.0f) -> (%.1f, %.1f)mm"
                % (payload["shape_cn"], color, payload["conf"],
                   cx, cy, x_mm, y_mm))

        if self.serial_worker is not None and self.serial_ok:
            self.serial_worker.submit(target["shape"], color, x_mm, y_mm)
        else:
            self.get_logger().info(
                "[SIM] 下发 %s/%s (%d,%d)（模拟抓取 %.1fs）" %
                (target["shape"], color, round(x_mm), round(y_mm),
                 self.grab_cycle_s))

        self._publish_state("SENT" if (self.serial_worker and self.serial_ok)
                            else "SIM_SENT")

    # ---------------- 标定热加载 ----------------
    def on_reload_calib(self, request, response):
        """
        重新读 hand_eye.json 和 camera_intrinsic.json，不用重启节点。

            ros2 service call /sort/reload_calib std_srvs/srv/Trigger
        """
        try:
            old = hand_eye.describe()
            hand_eye.reload()
            new = hand_eye.describe()
            self.require_calib = bool(getattr(C, "SAFETY_REQUIRE_CALIB", True))
            ok = camera_model.check_hand_eye_consistency(
                getattr(hand_eye, "META", {}) or {}, self.get_logger())
            msg = "重载完成 | 之前: %s | 现在: %s" % (old, new)
            if hand_eye.is_calibrated() and not self.allow_uncalibrated:
                msg += " | 标定闸已放开，可以正常下发"
            self.get_logger().info(msg)
            response.success = True if ok is not False else False
            response.message = msg
        except Exception as e:                        # noqa: BLE001
            msg = "重载失败: %s: %s" % (type(e).__name__, e)
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
        return response

    def _publish_state(self, s):
        self.pub_state.publish(String(data=s))

    def on_timer(self):
        self._try_reopen_serial()
        self.get_logger().info(
            "统计: 检测消息=%d 含目标=%d 下发=%d 屏蔽=%d 颜色失败=%d "
            "安全闸拦截=%d 串口失败=%d 边缘外推=%d 帧缓存=%d busy=%s 串口=%s"
            % (self.stat_det, self.stat_hit, self.stat_fire, self.stat_blank,
               self.stat_color_fail, self.stat_blocked, self.stat_send_fail,
               self.stat_edge, len(self.frames), self.busy,
               "模拟" if self.serial_sim else
               ("正常" if self.serial_ok else "故障")))

    def destroy_node(self):
        if self.serial_worker:
            self.serial_worker.stop()
        if self.comm:
            try:
                self.comm.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SortVisionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()