#!/usr/bin/env python3
"""
sort_vision_node.py — 视觉分拣控制节点 (RDK X5 / TROS Humble)
=====================================================================
职责：
  - USB相机采集 (C100)
  - BPU上跑YOLOv5n检测形状
  - HSV颜色投票判色（剔除高光/灰像素，深红暗橙二次判别）
  - 手眼标定：像素 -> 机械臂坐标
  - 状态机：等物料 -> 拍照识别 -> 发抓取指令 -> 等完成 -> 复位
  - 串口发送给STM32（协议参考 chassis_driver.py：帧头0x6C，帧尾0x2F）

订阅/发布：
  订阅: /sort/start       (std_msgs/Bool, 外部触发分拣开始)
  订阅: /sort/reset       (std_msgs/Bool, 复位)
  发布: /sort/result      (std_msgs/String, 识别结果 JSON: shape,color,x,y)
  发布: /sort/state       (std_msgs/String, 当前状态)
  发布: /driver/tx_debug   (std_msgs/String, 串口发送hex调试)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import cv2
import numpy as np
import serial
import struct
import time
import json
import os
import sys

# ============ 本地模块 ============
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from color_classifier import classify_color
from hand_eye import pixel_to_robot


class SortVisionNode(Node):
    def __init__(self):
        super().__init__('sort_vision_node')

        # ==================== 串口配置（对齐chassis_driver.py）====================
        self.SERIAL_PORT = C.SERIAL_PORT
        self.BAUD = C.SERIAL_BAUD
        self.FRAME_HEAD = 0x6C      # 'l'，和chassis_driver保持一致
        self.FRAME_TAIL = 0x2F      # '/'
        self.TASK_GRAB = 0x10       # 自定义：抓取指令

        self.ser = None
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD, timeout=0.1)
            self.get_logger().info(f'串口已打开: {self.SERIAL_PORT}')
        except Exception as e:
            self.get_logger().warn(f'串口打开失败（调试模式）: {e}')

        # ==================== BPU模型加载 ====================
        self.detector = None
        try:
            from hobot_dnn import pyeasy_dnn as dnn
            self.models = dnn.load(C.YOLO_BIN_MODEL)
            self.inp_h = self.models[0].inputs[0].properties.height
            self.inp_w = self.models[0].inputs[0].properties.width
            self.get_logger().info(f'BPU模型加载成功: {C.YOLO_BIN_MODEL}, 输入{self.inp_w}x{self.inp_h}')
        except Exception as e:
            self.get_logger().error(f'BPU模型加载失败: {e}')
            self.models = None

        # ==================== 相机 ====================
        self.cap = cv2.VideoCapture(C.CAM_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if not self.cap.isOpened():
            self.get_logger().error('相机打开失败!')
        self.latest_frame = None

        # ==================== 状态机 ====================
        self.STATE_IDLE = 0
        self.STATE_WAIT = 1
        self.STATE_DETECT = 2
        self.STATE_SEND = 3
        self.STATE_DONE = 4
        self.state = self.STATE_IDLE
        self.retry = 0
        self.max_retry = 3
        self.current_target = None

        # ==================== ROS话题 ====================
        self.create_subscription(Bool, '/sort/start', self.cb_start, 10)
        self.create_subscription(Bool, '/sort/reset', self.cb_reset, 10)
        self.pub_result = self.create_publisher(String, '/sort/result', 10)
        self.pub_state = self.create_publisher(String, '/sort/state', 10)
        self.pub_tx = self.create_publisher(String, '/driver/tx_debug', 10)

        # 定时器：20Hz跑状态机+相机
        self.create_timer(0.05, self.tick)
        self.get_logger().info('sort_vision_node 启动完成，等 /sort/start 触发')

    # ================================================================
    # 话题回调
    # ================================================================
    def cb_start(self, msg: Bool):
        if msg.data and self.state == self.STATE_IDLE:
            self.state = self.STATE_WAIT
            self.retry = 0
            self._pub_state('WAIT')

    def cb_reset(self, msg: Bool):
        if msg.data:
            self.state = self.STATE_IDLE
            self.current_target = None
            self._pub_state('IDLE')

    # ================================================================
    # 主循环
    # ================================================================
    def tick(self):
        # 1. 读相机
        ok, frame = self.cap.read()
        if ok:
            self.latest_frame = frame

        if self.latest_frame is None:
            return

        # 2. 状态机
        if self.state == self.STATE_WAIT:
            # 等一帧稳定后开始检测
            self.state = self.STATE_DETECT
            self._pub_state('DETECT')

        elif self.state == self.STATE_DETECT:
            result = self._run_detect(self.latest_frame)
            if result is None:
                self.retry += 1
                if self.retry >= self.max_retry:
                    self.get_logger().warn(f'检测失败{self.max_retry}次，回IDLE')
                    self.state = self.STATE_IDLE
                    self._pub_state('IDLE')
                return
            self.current_target = result
            self.state = self.STATE_SEND
            self._pub_state('SEND')

        elif self.state == self.STATE_SEND:
            ok = self._send_grab(self.current_target)
            if ok:
                self.state = self.STATE_DONE
                self._pub_state('DONE')
                self.get_logger().info(
                    f"抓取指令已发: {self.current_target['shape']}/{self.current_target['color']} "
                    f"-> ({self.current_target['rx']:.0f},{self.current_target['ry']:.0f})")
            else:
                self.get_logger().error('串口发送失败')
                self.state = self.STATE_IDLE
                self._pub_state('IDLE')

        elif self.state == self.STATE_DONE:
            # 等机械臂抓完（由下位机发完成信号回来，这里先定时）
            time.sleep(C.GRAB_DONE_WAIT_S)
            self.state = self.STATE_IDLE
            self.current_target = None
            self._pub_state('IDLE')

    # ================================================================
    # 视觉推理：YOLO + HSV
    # ================================================================
    def _run_detect(self, frame):
        if self.models is None:
            return None
        h0, w0 = frame.shape[:2]
        img_in = cv2.resize(frame, (self.inp_w, self.inp_h))
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        outputs = self.models[0].forward(img_rgb)

        out = outputs[0].buffer.reshape(-1, 5 + len(C.SHAPE_NAMES))
        sx = w0 / self.inp_w
        sy = h0 / self.inp_h
        cands = []
        for row in out:
            obj_conf = row[4]
            if obj_conf < C.YOLO_CONF_THRES:
                continue
            cls_id = int(np.argmax(row[5:]))
            score = float(obj_conf * row[5 + cls_id])
            if score < C.YOLO_CONF_THRES:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = int(max(0, (cx - bw / 2) * sx))
            y1 = int(max(0, (cy - bh / 2) * sy))
            x2 = int(min(w0, (cx + bw / 2) * sx))
            y2 = int(min(h0, (cy + bh / 2) * sy))
            cands.append((x1, y1, x2, y2, score, cls_id))

        if not cands:
            return None

        # 取置信度最高的
        cands.sort(key=lambda x: x[4], reverse=True)
        x1, y1, x2, y2, score, cls_id = cands[0]

        # HSV颜色投票
        roi = frame[y1:y2, x1:x2].copy()
        color, color_conf = classify_color(roi)
        if color is None or color_conf < 0.6:
            self.get_logger().warn(f'颜色识别失败 conf={color_conf}')
            return None

        # 像素中心 -> 机械臂坐标
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        rx, ry, rz = pixel_to_robot(u, v)

        result = {
            'shape': C.SHAPE_NAMES[cls_id],
            'color': color,
            'conf': score,
            'bbox': (x1, y1, x2, y2),
            'u': u, 'v': v,
            'rx': rx, 'ry': ry, 'rz': rz,
        }
        # 发布结果
        msg = String()
        msg.data = json.dumps(result)
        self.pub_result.publish(msg)
        return result

    # ================================================================
    # 串口发送抓取指令
    # ================================================================
    def _send_grab(self, target):
        if self.ser is None:
            self.get_logger().warn('无串口，模拟发送成功')
            return True
        shape_id = C.SHAPE_NAMES.index(target['shape'])
        color_id = list(C.COLOR_RANGES_HSV.keys()).index(target['color'])
        x_i = int(round(target['rx']))
        y_i = int(round(target['ry']))

        # 帧格式对齐chassis_driver:
        # [0] 帧头0x6C
        # [1] 任务字 0x10=抓取
        # [2] shape_id
        # [3] color_id
        # [4:6] x_int16 小端
        # [6:8] y_int16 小端
        # [8]   data_len
        # [9]   校验和（前面所有字节异或）
        # [10]  帧尾0x2F
        body = struct.pack('<BBB B hh B',
                           self.FRAME_HEAD, self.TASK_GRAB,
                           shape_id, color_id,
                           x_i, y_i,
                           6)
        checksum = 0
        for b in body:
            checksum ^= b
        frame = body + bytes([checksum, self.FRAME_TAIL])

        for _ in range(C.RETRY_TIMES):
            try:
                self.ser.write(frame)
                self.ser.flush()
                # 调试发布
                dbg = String()
                dbg.data = frame.hex()
                self.pub_tx.publish(dbg)
                time.sleep(0.05)
                # 等下位机ACK（约定回0x06）
                ack = self.ser.read(1)
                if ack and ack[0] == 0x06:
                    return True
            except Exception as e:
                self.get_logger().error(f'串口写异常: {e}')
        return False

    def _pub_state(self, s):
        msg = String()
        msg.data = s
        self.pub_state.publish(msg)

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        if self.ser:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SortVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
