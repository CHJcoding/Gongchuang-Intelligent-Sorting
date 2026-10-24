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
from ai_msgs.msg import PerceptionTargets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from color_classifier import classify_color
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

    def submit(self, shape, color, x_mm, y_mm, z_byte):
        try:
            self.q.put_nowait((shape, color, x_mm, y_mm, z_byte))
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
            shape, color, x_mm, y_mm, z_byte = item
            t0 = time.time()
            ok = self.comm.send_grab(shape, color, x_mm, y_mm, z_byte)
            dt = (time.time() - t0) * 1000
            if ok:
                self.node.get_logger().info(
                    "串口已发出 %.1fms | %s/%s (%d,%d) | 等待抓取完成(%s模式)" %
                    (dt, shape, color, round(x_mm), round(y_mm),
                     getattr(C, "GRAB_DONE_MODE", "timer")))
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
        self.auto_trigger = bool(self.get_parameter("auto_trigger").value)
        self.serial_sim = bool(self.get_parameter("serial_sim").value)

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

        # ---- 发布 ----
        self.pub_result = self.create_publisher(String, C.RESULT_TOPIC, 10)
        self.pub_state = self.create_publisher(String, "/sort/state", 10)
        self.pub_debug = self.create_publisher(String, "/sort/debug", 10)

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

        # ---- 串口 ----
        self.comm = None
        self.serial_worker = None
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
               getattr(C, "GRAB_DONE_MODE", "timer"),
               getattr(C, "GRAB_CYCLE_S", 6.0),
               getattr(C, "GRAB_TIMEOUT_S", 12.0)))

    # ---------------- 串口 ----------------
    def _init_serial(self):
        if self.serial_sim:
            self.get_logger().warn("serial_sim=True：不连下位机，只打印指令")
            return
        try:
            from comm import Comm
            self.comm = Comm()
            self.comm.open()
            self.serial_worker = SerialWorker(self.comm, self)
            self.serial_worker.start()
            self.get_logger().info("串口已打开 %s @%d" %
                                   (C.SERIAL_PORT, C.SERIAL_BAUD))
        except Exception as e:
            self.comm = None
            self.serial_worker = None
            self.get_logger().warn(
                "串口打开失败（%s: %s），自动进入模拟模式" %
                (type(e).__name__, e))

    # ---------------- 抓取窗口：如何判定"抓完了" ----------------
    def on_send_failed(self):
        """串口发送失败：立即解除抓取窗口。"""
        self.busy = False
        self.blank_until = 0.0
        self._publish_state("SEND_FAIL")

    def on_grab_watchdog(self):
        """
        10Hz 检查抓取窗口是否该结束。

          timer 模式：发出去后按 GRAB_CYCLE_S 定时解锁（STM32 无需任何回报）
          byte  模式：等 STM32 在动作结束时回的 GRAB_DONE_BYTE，超时则兜底解锁
        """
        if not self.busy:
            return
        now = time.time()
        elapsed = now - self.grab_start
        mode = str(getattr(C, "GRAB_DONE_MODE", "timer")).lower()

        if mode == "byte":
            if self.comm is not None and self.comm.poll_done():
                self._end_grab("DONE", elapsed)
                return
            if elapsed >= getattr(C, "GRAB_TIMEOUT_S", 12.0):
                self.get_logger().warn(
                    "抓取完成字节超时(%.1fs)，强制解锁（下位机是否没发完成信号？）" % elapsed)
                self._end_grab("TIMEOUT", elapsed)
        else:
            if elapsed >= getattr(C, "GRAB_CYCLE_S", 6.0):
                self._end_grab("TIMER", elapsed)

    def _end_grab(self, reason, elapsed):
        """结束抓取窗口：解除 busy、开始收臂屏蔽、允许抓下一个物料。"""
        self.busy = False
        self.lock_center = None            # 物料已被搬走，解除位置锁定
        self.blank_until = time.time() + getattr(C, "BLANK_AFTER_DONE_S", 0.8)
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
        mode = str(getattr(C, "GRAB_DONE_MODE", "timer")).lower()
        if mode == "byte":
            window = getattr(C, "GRAB_TIMEOUT_S", 12.0)
            if self.comm is not None:
                self.comm.drain()   # 清掉残留字节，避免把上一次的完成信号误当本次
        else:
            window = getattr(C, "GRAB_CYCLE_S", 6.0)
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
        x_mm, y_mm, z_mm = pixel_to_robot(cx, cy)

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
            "robot_mm": [x_mm, y_mm, z_mm],
        }
        self.pub_result.publish(String(data=json.dumps(payload,
                                                      ensure_ascii=False)))
        self.get_logger().info("目标 %s/%s conf=%.2f px=(%.0f,%.0f) -> (%.1f, %.1f)mm"
                               % (payload["shape_cn"], color, payload["conf"],
                                  cx, cy, x_mm, y_mm))

        if self.serial_worker is not None:
            self.serial_worker.submit(target["shape"], color, x_mm, y_mm, 0)
        else:
            self.get_logger().info(
                "[SIM] 下发 %s/%s (%d,%d) z=0（模拟抓取 %.1fs）" %
                (target["shape"], color, round(x_mm), round(y_mm),
                 getattr(C, "GRAB_CYCLE_S", 6.0)))

        self._publish_state("SENT" if self.serial_worker else "SIM_SENT")

    def _publish_state(self, s):
        self.pub_state.publish(String(data=s))

    def on_timer(self):
        self.get_logger().info(
            "统计: 检测消息=%d 含目标=%d 下发=%d 屏蔽=%d 颜色失败=%d 帧缓存=%d busy=%s"
            % (self.stat_det, self.stat_hit, self.stat_fire, self.stat_blank,
               self.stat_color_fail, len(self.frames), self.busy))

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