# -*- coding: utf-8 -*-
"""
与下位机(STM32)的串口通信协议。

下行帧（RDK -> STM32，10 字节）:
  [0]   0xAA
  [1]   0x55
  [2]   shape_id   形状编号 0-6（255=未知）
  [3]   color_id   颜色编号 0-9（255=未知）
  [4:6] x_int16    机械臂目标 X(mm)，小端
  [6:8] y_int16    机械臂目标 Y(mm)，小端
  [8]   crc8       从帧头到 y 的校验，poly 0x07
  [9]   0x0D       帧尾

注：z 轴已按需求移除（STM32 侧自行决定抓取高度），帧长 11 -> 10 字节。

上行（STM32 -> RDK），三种包可以混在同一条串口上，RDK 侧统一解析：
  1) 抓取完成 1 字节：0x06
       仅 GRAB_DONE_MODE == "byte" 时被调用。这是"抓完了"的动作级信号，
       不是"收到帧了"的帧级 ACK。
  2) 抓取完成 + 结果 2 字节：0x06 + status
       status: 0x00=成功 0x01=空抓/失败 0x02=急停 0x03=其它
       仅 GRAB_STATUS_ENABLE=True 时启用（当前未启用）。
  3) 夹爪坐标 6 字节：0x07 | x int16 LE | y int16 LE | crc8
       crc8 覆盖前 5 字节，poly 0x07（与下行同一算法）。
       仅 POS_ENABLE=True 时被节点消费（当前未启用），但解析器始终兼容，
       收到就存进 latest_position，不消费也不会丢其它包。

注意：三种包必须由同一个解析泵处理，否则谁先读谁把别人的字节吃掉。
本模块所有 poll_* 都走 _pump()，可以任意顺序、任意频率交替调用。

另外保留可选的帧级 ACK 逻辑（SEND_WAIT_ACK=True 时启用，默认关闭）。
"""
import struct
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
# 颜色编号用 config.COLOR_IDS 显式固定，不依赖 COLOR_RANGES_HSV 的字典顺序
COLOR_ID = dict(getattr(C, "COLOR_IDS", {}))

# 上行包标识
DONE_BYTE = int(getattr(C, "GRAB_DONE_BYTE", 0x06))
POS_HEAD = int(getattr(C, "POS_FRAME_HEAD", 0x07))
POS_LEN = int(getattr(C, "POS_FRAME_LEN", 6))
STATUS_NAMES = dict(getattr(C, "GRAB_STATUS_NAMES", {}))
RX_BUF_MAX = 512             # 缓冲区上限，防止串口噪声把内存撑爆


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_frame(shape_name, color_name, x_mm, y_mm) -> bytes:
    """组帧（独立出来便于单测/离线校验）。10 字节，不含 z。"""
    shape_id = SHAPE_ID.get(shape_name, 255)
    color_id = COLOR_ID.get(color_name, 255)
    x_i = int(round(x_mm))
    y_i = int(round(y_mm))
    # int16 范围保护，越界截断而不是抛异常
    x_i = max(-32768, min(32767, x_i))
    y_i = max(-32768, min(32767, y_i))
    body = bytes(C.FRAME_HEAD) + bytes([shape_id, color_id]) + \
        struct_pack_hh(x_i, y_i)
    crc = _crc8(body)
    return body + bytes([crc, C.FRAME_TAIL])


def struct_pack_hh(a, b) -> bytes:
    """struct.pack('<hh', a, b) 的轻量替代。"""
    return struct.pack("<hh", a, b)


def build_pos_frame(x_mm, y_mm) -> bytes:
    """组一帧"夹爪坐标"上行帧（6 字节）。

    这是给【单测和 STM32 工程师当参考实现】用的：STM32 侧应按同样字节序发出。
    """
    x_i = max(-32768, min(32767, int(round(x_mm))))
    y_i = max(-32768, min(32767, int(round(y_mm))))
    body = bytes([POS_HEAD]) + struct.pack("<hh", x_i, y_i)
    return body + bytes([_crc8(body)])


class Comm:
    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()      # 写线程(发送) 与 ROS定时器(读完成字节) 共用串口
        self.done_seen = False
        self.done_status = None           # 2 字节协议的结果码；1 字节协议为 None
        self._rx = bytearray()            # 统一接收缓冲（所有上行包共用）
        self.latest_position = None       # (x_mm, y_mm)，只保留最新一帧
        self.position_seq = 0             # 累计收到多少帧坐标（调试/丢帧观察）
        self.rx_bad = 0                   # CRC 失败 / 半包丢弃计数

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

    def is_open(self):
        """串口当前是否真的可用（USB 转串口拔掉后这里会变 False）。"""
        try:
            return self.ser is not None and bool(self.ser.is_open)
        except Exception:
            return False

    def reopen(self):
        """关掉重开一次（用于 USB 转串口热插拔后自愈）。返回是否成功。"""
        self.close()
        try:
            self.open()
            return self.is_open()
        except Exception:
            self.close()
            return False

    # ---------------- 发送 ----------------
    def send_grab(self, shape_name, color_name, x_mm, y_mm,
                  wait_ack=None):
        """
        发一次抓取指令（10 字节，不含 z）。
          wait_ack=False（默认）：只保证写出串口，立即返回 True
          wait_ack=True         ：等待帧级 ACK 0x06，失败重试 RETRY_TIMES 次
        返回: True 已发出（或收到 ACK）/ False 串口异常或 ACK 超时

        写失败时会把句柄置空（is_open() 变 False），让上层知道"串口掉了"，
        而不是继续往一个死句柄里写、日志里却一切正常。
        """
        if wait_ack is None:
            wait_ack = bool(getattr(C, "SEND_WAIT_ACK", False))
        frame = build_frame(shape_name, color_name, x_mm, y_mm)

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
            self.close()                 # 句柄已不可用，标记掉线，等上层重连
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

    # ---------------- 统一接收解析 ----------------
    def _pump(self):
        """把串口里现有字节全部读进缓冲，然后解析出所有完整包。"""
        while True:
            chunk = self._read_nonblock(64)
            if not chunk:
                break
            self._rx.extend(chunk)
        if len(self._rx) > RX_BUF_MAX:          # 噪声太多时只保留尾部
            del self._rx[:-RX_BUF_MAX // 2]
        self._parse_rx()

    def _parse_rx(self):
        """按 0x06 / 0x07 逐字节扫描，拼出完整包后消费。

        半包（字节不够）保留在缓冲里等下一次；CRC 错的坐标帧丢 1 字节重新同步。
        """
        buf = self._rx
        i, n = 0, len(buf)
        status_mode = bool(getattr(C, "GRAB_STATUS_ENABLE", False))

        while i < n:
            b = buf[i]

            # ---- 抓取完成：0x06 [status] ----
            if b == DONE_BYTE:
                if status_mode:
                    if i + 1 >= n:
                        break                       # 等 status 字节
                    st = buf[i + 1]
                    if st in STATUS_NAMES:          # 合法结果码：0x06 + 结果
                        self.done_seen = True
                        self.done_status = st
                        i += 2
                        continue
                    # 非法结果码：只当 1 字节完成信号，不吞下一字节
                    self.done_seen = True
                    self.done_status = None
                    i += 1
                    continue
                self.done_seen = True
                self.done_status = None
                i += 1
                continue

            # ---- 夹爪坐标：0x07 x(2) y(2) crc(1) ----
            if b == POS_HEAD:
                if i + POS_LEN > n:
                    break                           # 半包，等后续字节
                body = bytes(buf[i:i + POS_LEN - 1])
                crc = buf[i + POS_LEN - 1]
                if _crc8(body) == crc:
                    x = struct.unpack("<h", body[1:3])[0]
                    y = struct.unpack("<h", body[3:5])[0]
                    self.latest_position = (x, y)
                    self.position_seq += 1
                    i += POS_LEN
                    continue
                self.rx_bad += 1
                i += 1                              # CRC 错：丢 1 字节重新找包头
                continue

            i += 1                                  # 杂字节

        del buf[:i]                                 # 只保留没解析完的半包

    def poll_done(self) -> bool:
        """非阻塞检查是否收到"抓取完成"字节；取到即消费（同一字节不会触发两次）。"""
        self._pump()
        if self.done_seen:
            self.done_seen = False
            return True
        return False

    def poll_done_status(self):
        """
        【2 字节协议，默认未启用】取 (seen, status | None)。
        status: 0x00=成功 0x01=空抓 0x02=急停 0x03=其它；1 字节协议下为 None。
        """
        self._pump()
        if self.done_seen:
            self.done_seen = False
            return True, self.done_status
        return False, None

    def poll_position(self):
        """非阻塞取最新夹爪坐标 (x_mm, y_mm)；没有新帧返回 None。"""
        self._pump()
        pos = self.latest_position
        self.latest_position = None
        return pos

    def drain(self):
        """丢弃输入缓冲里的残留字节（开始新一次抓取前调用）。"""
        with self.lock:
            if self.ser is not None:
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
        self.done_seen = False
        self.done_status = None
        self._rx = bytearray()
        self.latest_position = None


if __name__ == "__main__":
    # 离线自检：打印一帧的十六进制，不需要下位机
    fr = build_frame("YZHU", "green", -12, 407)
    print("frame(%d bytes):" % len(fr), " ".join("%02X" % b for b in fr))
    print("crc8 校验位 = 0x%02X" % fr[8])
    # 联机自检（需要串口）
    if len(__import__("sys").argv) > 1:
        c = Comm()
        c.open()
        print("send:", c.send_grab("YZHU", "green", 200, 300))
        c.close()