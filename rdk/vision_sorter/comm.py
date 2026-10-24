# -*- coding: utf-8 -*-
"""
与下位机(STM32)的串口通信协议。

下行帧（RDK -> STM32，11 字节）:
  [0]   0xAA
  [1]   0x55
  [2]   shape_id   形状编号 0-6（255=未知）
  [3]   color_id   颜色编号 0-9（255=未知）
  [4:6] x_int16    机械臂目标 X(mm)，小端
  [6:8] y_int16    机械臂目标 Y(mm)，小端
  [8]   z_byte     抓取高度档(0=低,1=中,2=高)
  [9]   crc8       从帧头到 z 的校验，poly 0x07
  [10]  0x0D       帧尾

上行（STM32 -> RDK），按 config.GRAB_DONE_MODE：
  * "timer" 模式：STM32 什么都不用回（纯开环定时解锁）
  * "byte"  模式：抓取动作【结束时】回 1 字节 GRAB_DONE_BYTE（0x06）
                 注意这是"抓完了"的动作级信号，不是"收到帧了"的帧级 ACK。
                 若 config.GRAB_STATUS_ENABLE=True（当前未启用），则回 2 字节：
                     0x06 + status    status: 0x00=成功 0x01=空抓 0x02=急停 0x03=其它

另外保留可选的帧级 ACK 逻辑（SEND_WAIT_ACK=True 时启用，默认关闭）。
"""
import threading
import time

import config as C

try:
    import serial
    HAS_SERIAL = True
except ImportError:          # PC 上没装 pyserial 时仍可 import 本模块做组帧校验
    serial = None
    HAS_SERIAL = False

# 形状/颜色和编号的映射，要和下位机固件里的表一致
SHAPE_ID = {name: i for i, name in enumerate(C.SHAPE_NAMES)}
COLOR_ID = {name: i for i, name in enumerate(C.COLOR_RANGES_HSV.keys())}


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_frame(shape_name, color_name, x_mm, y_mm, z_byte=0) -> bytes:
    """组帧（独立出来便于单测/离线校验）。"""
    shape_id = SHAPE_ID.get(shape_name, 255)
    color_id = COLOR_ID.get(color_name, 255)
    x_i = int(round(x_mm))
    y_i = int(round(y_mm))
    # int16 范围保护，越界截断而不是抛异常
    x_i = max(-32768, min(32767, x_i))
    y_i = max(-32768, min(32767, y_i))
    z_b = max(0, min(255, int(z_byte)))
    body = bytes(C.FRAME_HEAD) + bytes([shape_id, color_id]) + \
        struct_pack_hh(x_i, y_i) + bytes([z_b])
    crc = _crc8(body)
    return body + bytes([crc, C.FRAME_TAIL])


def struct_pack_hh(a, b) -> bytes:
    """struct.pack('<hh', a, b) 的轻量替代。"""
    import struct
    return struct.pack("<hh", a, b)


class Comm:
    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()      # 写线程(发送) 与 ROS定时器(读完成字节) 共用串口
        self.done_seen = False
        self._status_buf = b""            # 2 字节协议的分包缓存（未启用时不会用到）

    # ---------------- 打开/关闭 ----------------
    def open(self):
        if not HAS_SERIAL:
            raise RuntimeError("未安装 pyserial：pip3 install pyserial")
        with self.lock:
            self.ser = serial.Serial(
                C.SERIAL_PORT, C.SERIAL_BAUD, timeout=0)   # 非阻塞读
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def close(self):
        with self.lock:
            if self.ser:
                try:
                    self.ser.close()
                finally:
                    self.ser = None

    # ---------------- 发送 ----------------
    def send_grab(self, shape_name, color_name, x_mm, y_mm, z_byte=0,
                  wait_ack=None):
        """
        发一次抓取指令。
          wait_ack=False（默认）：只保证写出串口，立即返回 True
          wait_ack=True         ：等待帧级 ACK 0x06，失败重试 RETRY_TIMES 次
        返回: True 已发出（或收到 ACK）/ False 串口异常或 ACK 超时
        """
        if wait_ack is None:
            wait_ack = bool(getattr(C, "SEND_WAIT_ACK", False))
        frame = build_frame(shape_name, color_name, x_mm, y_mm, z_byte)

        if self.ser is None:
            return False
        try:
            if not wait_ack:
                with self.lock:
                    self.ser.write(frame)
                    self.ser.flush()
                return True
            # 旧的帧级 ACK 模式
            for _ in range(C.RETRY_TIMES):
                with self.lock:
                    self.ser.reset_input_buffer()
                    self.ser.write(frame)
                    self.ser.flush()
                t_end = time.time() + C.TIMEOUT_S
                while time.time() < t_end:
                    b = self._read_nonblock(1)
                    if b and b[0] == C.ACK_BYTE:
                        return True
                    time.sleep(0.01)
            return False
        except Exception:
            return False

    # ---------------- 接收 ----------------
    def _read_nonblock(self, n=1):
        with self.lock:
            if self.ser is None:
                return b""
            try:
                waiting = self.ser.in_waiting
                if waiting <= 0:
                    return b""
                return self.ser.read(min(n, waiting))
            except Exception:
                return b""

    def poll_done(self) -> bool:
        """
        非阻塞检查是否收到"抓取完成"字节。
        收到即返回 True；同时清掉其它杂字节（避免累积）。
        仅 GRAB_DONE_MODE == "byte" 时被调用。
        """
        data = self._read_nonblock(16)
        if not data:
            return False
        if C.GRAB_DONE_BYTE in data:
            self.done_seen = True
            return True
        return False

    def poll_done_status(self):
        """
        【2 字节协议，默认未启用】解析 0x06 + status。
        返回 (seen: bool, status: int | None)。
        只在 config.GRAB_STATUS_ENABLE=True 时由节点调用。
        做了分包缓存：0x06 和 status 分两次到达也能拼上。
        """
        chunk = self._read_nonblock(32)
        if chunk:
            self._status_buf += chunk
        while len(self._status_buf) >= 2:
            i = self._status_buf.find(bytes([C.GRAB_DONE_BYTE]))
            if i < 0:
                self._status_buf = self._status_buf[-1:]   # 只留最后 1 字节，防误伤
                return False, None
            if i + 1 < len(self._status_buf):
                status = self._status_buf[i + 1]
                self._status_buf = b""
                return True, status
            self._status_buf = self._status_buf[i:]        # 只有 0x06，等下一字节
            return False, None
        return False, None

    def drain(self):
        """丢弃输入缓冲里的残留字节（开始新一次抓取前调用）。"""
        with self.lock:
            if self.ser is None:
                return
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
        self.done_seen = False
        self._status_buf = b""


if __name__ == "__main__":
    # 离线自检：打印一帧的十六进制，不需要下位机
    fr = build_frame("YZHU", "green", -12, 407, 0)
    print("frame(%d bytes):" % len(fr), " ".join("%02X" % b for b in fr))
    print("crc8 校验位 = 0x%02X" % fr[9])
    # 联机自检（需要串口）
    if len(__import__("sys").argv) > 1:
        c = Comm()
        c.open()
        print("send:", c.send_grab("YZHU", "green", 200, 300, 0))
        c.close()