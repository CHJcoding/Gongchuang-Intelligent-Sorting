# -*- coding: utf-8 -*-
"""
与下位机(STM32)的串口通信协议。

串口设备：STM32 是 USB CDC，默认 /dev/ttyACM0。拔插/重启后可能变 ACM1，
由 resolve_serial_port() 自动挑：/dev/serial/by-id/* → config.SERIAL_PORT
→ config.SERIAL_PORT_CANDIDATES，设备号漂移不用改代码。

下行帧（RDK -> STM32，10 字节）:
  [0]   0xAA
  [1]   0x55
  [2]   material_id 物料编号 0-99 = 形状*10 + 颜色（255=未知）
  [3]   color_id    颜色编号 0-9（= material_id % 10，供 32 交叉校验；255=未知）
  [4:6] x_int16     机械臂目标 X(mm)，小端
  [6:8] y_int16     机械臂目标 Y(mm)，小端
  [8]   crc8        从帧头到 y 的校验，poly 0x07
  [9]   0x0D        帧尾

物料编号表（与 STM32 固件里的 10x10 表逐格对应，2026-10-26 对齐）:
        红0  橙1  黄2  绿3  棕4  蓝5  紫6  品7  黑8  白9
  0 长方体     0    1    2    3    4    5    6    7    8    9
  1 镂空正方体 10   11   12   13   14   15   16   17   18   19
  2 球        20   21   22   23   24   25   26   27   28   29
  3 三棱柱    30   31   32   33   34   35   36   37   38   39
  4 圆锥      40   41   42   43   44   45   46   47   48   49
  5 正方体    50   51   52   53   54   55   56   57   58   59
  6 圆柱      60   61   62   63   64   65   66   67   68   69
  7 五棱柱    70   71   72   73   74   75   76   77   78   79
  8 六棱柱    80   81   82   83   84   85   86   87   88   89
  9 三棱锥    90   91   92   93   94   95   96   97   98   99
  例如 11 = 橙色的镂空正方体。(0=长方体/5=正方体/8=六棱柱 我们的模型还没有对应类)
2026-10-28 用户确认：十个形状现场都有实物（2 号球有用），当前模型缺 0/5/8 三类，
换新的 YOLO 模型后补齐。

注：z 轴已按需求移除（STM32 侧自行决定抓取高度），帧长 11 -> 10 字节。

上行（STM32 -> RDK），四种包可以混在同一条串口上，RDK 侧统一解析：
  1) 抓取完成 1 字节：0x06
       仅 GRAB_DONE_MODE == "byte" 时被调用。这是"抓完了"的动作级信号，
       不是"收到帧了"的帧级 ACK。
  1b) 抓取完成 文本 "OK" / "ok"
       现场 STM32 常用 printf("OK") 表示"抓完放好了"，RDK 也认，
       由 GRAB_DONE_ACCEPT_ASCII 控制（默认 True）。两种写法都支持，
       联调时不用为了凑协议改固件。
  2) 抓取完成 + 结果 2 字节：0x06 + status
       status: 0x00=成功 0x01=空抓/失败 0x02=急停 0x03=其它
       仅 GRAB_STATUS_ENABLE=True 时启用（当前未启用）。
  3) 夹爪坐标 6 字节：0x07 | x int16 LE | y int16 LE | crc8
       crc8 覆盖前 5 字节，poly 0x07（与下行同一算法）。
       仅 POS_ENABLE=True 时被节点消费（当前未启用），但解析器始终兼容，
       收到就存进 latest_position，不消费也不会丢其它包。
  3b) 夹爪坐标 文本（**现场固件实际用的就是这个**，2026-10-28 抓包确认）
       b"POS A:655279 B:3 X:-199.98 Y:-199.97 Adeg:-3599.56 Bdeg:-0.02 valid:3\r\n"
       每 0.2s 一帧（5 Hz），以 b"POS " 开头、\r\n 结尾。
       X/Y 就是夹爪的 mm 坐标（浮点），valid:0 表示还没归零/坐标不可信，此时
       不更新坐标（否则会把假 0,0 当真实位置）。由 POS_TEXT_ENABLE 控制（默认 True）。
       顺带把 A/B 编码器计数、Adeg/Bdeg 角度也存进 latest_pos_text 备用。

注意：三种包必须由同一个解析泵处理，否则谁先读谁把别人的字节吃掉。
本模块所有 poll_* 都走 _pump()，可以任意顺序、任意频率交替调用。

另外保留可选的帧级 ACK 逻辑（SEND_WAIT_ACK=True 时启用，默认关闭）。
"""
import os
import re
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
# 2026-10-26：改成 STM32 的 10x10 表口径 —— 形状号不是 enumerate 顺序，
# 而是"该形状在 32 表里的行号"（见 config.SHAPE_IDS 的表）。
SHAPE_ID = dict(getattr(C, "SHAPE_IDS", None) or
                {name: i for i, name in enumerate(C.SHAPE_NAMES)})
# 颜色编号用 config.COLOR_IDS 显式固定，不依赖 COLOR_RANGES_HSV 的字典顺序
COLOR_ID = dict(getattr(C, "COLOR_IDS", {}))
UNKNOWN_ID = 255


def material_id(shape_name, color_name):
    """物料的整体编号 = 形状*10 + 颜色（与 STM32 的 100 格表一一对应）。

    返回 (id, shape_id, color_id)；任一项未知时 id 返回 None，调用方按 255 发。
    """
    sid = SHAPE_ID.get(shape_name)
    cid = COLOR_ID.get(color_name)
    if sid is None or cid is None:
        return None, (UNKNOWN_ID if sid is None else sid), \
            (UNKNOWN_ID if cid is None else cid)
    return sid * 10 + cid, sid, cid

# 上行包标识
DONE_BYTE = int(getattr(C, "GRAB_DONE_BYTE", 0x06))
POS_HEAD = int(getattr(C, "POS_FRAME_HEAD", 0x07))
POS_LEN = int(getattr(C, "POS_FRAME_LEN", 6))
STATUS_NAMES = dict(getattr(C, "GRAB_STATUS_NAMES", {}))
RX_BUF_MAX = 512             # 缓冲区上限，防止串口噪声把内存撑爆

# ---- 文本坐标帧（现场固件实际格式，2026-10-28 抓包确认）----
# b"POS A:655279 B:3 X:-199.98 Y:-199.97 Adeg:-3599.56 Bdeg:-0.02 valid:3\r\n"
POS_TEXT_PREFIX = b"POS "
POS_TEXT_ENABLE = bool(getattr(C, "POS_TEXT_ENABLE", True))
POS_TEXT_MAX_LEN = 200       # 一行最长容忍；超过还没遇到 \n 就当噪声丢掉
_POS_TXT_X = re.compile(r"\bX:(-?\d+(?:\.\d+)?)")
_POS_TXT_Y = re.compile(r"\bY:(-?\d+(?:\.\d+)?)")
_POS_TXT_VALID = re.compile(r"\bvalid:(-?\d+)")
_POS_TXT_A = re.compile(r"\bA:(-?\d+)")
_POS_TXT_B = re.compile(r"\bB:(-?\d+)")
_POS_TXT_ADEG = re.compile(r"\bAdeg:(-?\d+(?:\.\d+)?)")
_POS_TXT_BDEG = re.compile(r"\bBdeg:(-?\d+(?:\.\d+)?)")


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_frame(shape_name, color_name, x_mm, y_mm) -> bytes:
    """组帧（独立出来便于单测/离线校验）。10 字节，不含 z。

    [2] = **形状号**（config.SHAPE_IDS，0~9），[3] = **颜色号**（config.COLOR_IDS，0~9）。
    2026-10-30 起下行统一走"形状 / 颜色 两字节"口径（config.FRAME_ID_MODE = "split"），
    与 STM32 侧 协议说明_串口帧.md 第 2 节一致。
    把 config.FRAME_ID_MODE 置成 "combined" 时，[2] 退回旧口径"物料整体编号 = 形状*10+颜色"
    （0~99），[3] 仍是颜色号 —— 仅为兼容老固件保留。
    识别不出形状或颜色时，对应字节发 255（UNKNOWN_ID）。
    """
    mid, shape_id, color_id = material_id(shape_name, color_name)
    if C.FRAME_ID_MODE == "combined":
        id_byte = UNKNOWN_ID if mid is None else mid
    else:
        id_byte = UNKNOWN_ID if shape_id is None else shape_id
    x_i = int(round(x_mm))
    y_i = int(round(y_mm))
    # int16 范围保护，越界截断而不是抛异常
    x_i = max(-32768, min(32767, x_i))
    y_i = max(-32768, min(32767, y_i))
    body = bytes(C.FRAME_HEAD) + bytes([id_byte, color_id]) + \
        struct_pack_hh(x_i, y_i)
    if getattr(C, "FRAME_CRC_MODE", "poly07") == "raw":
        crc = int(getattr(C, "FRAME_CRC_FIXED", 0x00)) & 0xFF
    else:
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


def serial_port_candidates(preferred=None):
    """按可靠性排序返回候选串口设备（只排序，不检查是否存在）。

    顺序：/dev/serial/by-id/*（USB CDC 的稳定符号链接，拔插不变号）
          → preferred（参数或 config.SERIAL_PORT）
          → config.SERIAL_PORT_CANDIDATES
    """
    out = []
    stable = getattr(C, "SERIAL_PORT_STABLE", None)
    if stable:
        out.append(stable)
    by_id = "/dev/serial/by-id"
    try:
        if os.path.isdir(by_id):
            for name in sorted(os.listdir(by_id)):
                out.append(os.path.join(by_id, name))
    except OSError:
        pass
    first = preferred or getattr(C, "SERIAL_PORT", "/dev/ttyACM0")
    if first:
        out.append(first)
    for p in (getattr(C, "SERIAL_PORT_CANDIDATES", None) or []):
        if p and p not in out:
            out.append(p)
    return out


def resolve_serial_port(preferred=None):
    """返回第一个真实存在的串口设备名。

    都不存在时返回首选名字（不变空字符串），这样 open() 会抛
    "could not open port /dev/ttyACM0" 这种一眼能看懂的错，而不是空路径的怪错。
    """
    cands = serial_port_candidates(preferred)
    for p in cands:
        try:
            if os.path.exists(p):
                return p
        except OSError:
            continue
    return cands[0] if cands else ""


def usb_devnum(path):
    """设备节点 → 这次 USB 连接的编号（同一口重新枚举后编号会变）。

    设备名（ttyACM0 / ttyACM1）拔插后会漂移，这个 devnum 只在"真的重新插了一次"
    或下位机复位重新枚举时才变，所以它是判断"我手里的句柄是不是已经作废"的硬证据。
    取不到（非 USB 串口、权限不足）时返回 None，调用方按"看不出来"处理。
    """
    try:
        name = os.path.basename(os.path.realpath(path))
        dev = os.path.realpath("/sys/class/tty/%s/device" % name)
    except (OSError, ValueError):
        return None
    for base in (dev, os.path.dirname(dev)):
        try:
            with open(os.path.join(base, "devnum")) as f:
                return "%s:%s" % (os.path.basename(base), f.read().strip())
        except OSError:
            continue
    return None


class Comm:
    def __init__(self):
        self.ser = None
        self.port = None                  # 实际打开的设备名（open() 时解析，日志里打出来）
        self.devnum = None                # 打开时那次 USB 连接的编号（usb_devnum）
        self.lock = threading.Lock()      # 写线程(发送) 与 ROS定时器(读完成字节) 共用串口
        self.done_seen = False
        self.done_status = None           # 2 字节协议的结果码；1 字节协议为 None
        self._rx = bytearray()            # 统一接收缓冲（所有上行包共用）
        self.latest_position = None       # (x_mm, y_mm)，只保留最新一帧
        self.position_seq = 0             # 累计收到多少帧坐标（调试/丢帧观察）
        self.rx_bad = 0                   # CRC 失败 / 半包丢弃计数
        # 文本坐标帧（固件实测格式）：浮点坐标 + valid 标志 + 编码器/角度附送信息
        self.latest_position_f = None     # (x_mm, y_mm) 浮点，保留小数（标定要用）
        self.pos_valid = None             # 固件报的 valid 值（0=未归零/不可信）
        self.latest_pos_text = None       # 整条文本帧解析结果 dict
        self.pos_text_frames = 0          # 收到多少条文本坐标帧（含 valid=0 的）
        self.pos_text_invalid = 0         # 其中 valid=0 被丢弃的条数

    # ---------------- 打开/关闭 ----------------
    def open(self):
        if not HAS_SERIAL:
            raise RuntimeError("未安装 pyserial：pip3 install pyserial")
        port = resolve_serial_port()
        with self.lock:
            self.port = port
            self.devnum = usb_devnum(port)
            self.ser = serial.Serial(
                port, C.SERIAL_BAUD, timeout=0)   # 非阻塞读
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

    def link_state(self):
        """链路健康度：'ok' / 'closed' / 'gone' / 'hungup' / 'replaced'。

        只看 is_open() 是不够的：USB 拔掉之后 pyserial 的 is_open() 仍然返回 True，
        上层就会一直以为串口好好的（现场卡在归零闸 4 分钟就是这么来的）。
        这里补三件事：
          1) 设备节点还在不在（拔线后 /dev/ttyACM* 直接消失）
          2) 句柄还能不能 ioctl（拔线后内核回 EIO，in_waiting 会抛异常）
          3) 是不是同一次 USB 连接（32 复位会重新枚举，devnum 变）
        """
        if self.ser is None:
            return "closed"
        if not self.port or not os.path.exists(self.port):
            return "gone"
        try:
            _ = self.ser.in_waiting       # 真的问一次内核，死句柄抛 EIO
        except Exception:                  # noqa: BLE001
            return "hungup"
        cur = usb_devnum(self.port)
        if self.devnum and cur and cur != self.devnum:
            return "replaced"
        return "ok"

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

    def _consume_pos_text(self, line: bytes):
        """解析一条文本坐标帧 b"POS A:.. B:.. X:.. Y:.. Adeg:.. Bdeg:.. valid:.."。

        valid <= 0 表示固件还没归零 / 坐标不可信，此时**不更新坐标**：
        否则会把刚上电的 0,0 当真实位置，手眼标定会采到一堆假点。
        解析成功时坐标存两份：浮点 latest_position_f（标定要用小数），
        整数 latest_position（老接口 / 日志用）。
        """
        self.pos_text_frames += 1
        text = line.decode("ascii", "replace")
        mx = _POS_TXT_X.search(text)
        my = _POS_TXT_Y.search(text)
        if not mx or not my:
            self.rx_bad += 1
            return
        mv = _POS_TXT_VALID.search(text)
        valid = int(mv.group(1)) if mv else 1
        self.pos_valid = valid

        def _num(pat, cast):
            m = pat.search(text)
            if not m:
                return None
            try:
                return cast(m.group(1))
            except (TypeError, ValueError):
                return None

        x = float(mx.group(1))
        y = float(my.group(1))
        self.latest_pos_text = {
            "x": x, "y": y, "valid": valid,
            "a": _num(_POS_TXT_A, int), "b": _num(_POS_TXT_B, int),
            "adeg": _num(_POS_TXT_ADEG, float), "bdeg": _num(_POS_TXT_BDEG, float),
        }
        if valid <= 0:
            self.pos_text_invalid += 1
            return
        # 固件刚上电/还没回零时会发 valid:3 但坐标和编码器全 0
        # （2026-10-28 实测：POS A:0 B:0 X:0.00 Y:0.00 ... valid:3），
        # 这种"假有效"必须挡住，否则标定会把 (0,0) 当真点采进去。
        a_val = _num(_POS_TXT_A, int)
        b_val = _num(_POS_TXT_B, int)
        if (x == 0.0 and y == 0.0
                and (a_val in (None, 0)) and (b_val in (None, 0))):
            self.pos_text_invalid += 1
            self.pos_valid = 0
            return
        self.latest_position_f = (x, y)
        self.latest_position = (int(round(x)), int(round(y)))
        self.position_seq += 1

    def _parse_rx(self):
        """按 0x06 / 0x07 逐字节扫描，拼出完整包后消费。

        半包（字节不够）保留在缓冲里等下一次；CRC 错的坐标帧丢 1 字节重新同步。
        """
        buf = self._rx
        i, n = 0, len(buf)
        status_mode = bool(getattr(C, "GRAB_STATUS_ENABLE", False))
        ascii_mode = bool(getattr(C, "GRAB_DONE_ACCEPT_ASCII", False))

        while i < n:
            b = buf[i]

            # ---- 夹爪坐标：文本行 "POS A:.. B:.. X:.. Y:.. Adeg:.. Bdeg:.. valid:.." ----
            # 现场固件实际发的就是这个（0x07 二进制帧是备用格式）。
            # 'P'=0x50 不会和 0x06 / 0x07 / 'O'(0x4F) 撞车。
            if POS_TEXT_ENABLE and b == 0x50:
                if n - i < len(POS_TEXT_PREFIX):
                    break                               # 前缀还没收全，等下一批
                if bytes(buf[i:i + len(POS_TEXT_PREFIX)]) == POS_TEXT_PREFIX:
                    nl = buf.find(b"\n", i)
                    if nl < 0:
                        if n - i > POS_TEXT_MAX_LEN:    # 太长没有换行：当噪声
                            self.rx_bad += 1
                            i += 1
                            continue
                        break                           # 半行，等 \r\n
                    line = bytes(buf[i:nl + 1])
                    self._consume_pos_text(line)
                    i = nl + 1
                    continue
                i += 1
                continue

            # ---- 抓取完成：文本 "OK"（STM32 直接 printf("OK") 的情况）----
            # 放在 0x06 之前判断：'O'=0x4F 不会和 0x06/0x07 包头撞车。
            if ascii_mode and b in (0x4F, 0x6F):        # 'O' / 'o'
                if i + 1 >= n:
                    break                               # 半包，等下一个字节
                if buf[i + 1] in (0x4B, 0x6B):          # 'K' / 'k'
                    self.done_seen = True
                    self.done_status = None
                    i += 2
                    continue
                i += 1
                continue

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
                    self.latest_position_f = (float(x), float(y))
                    self.pos_valid = 1
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

    def poll_position_f(self):
        """非阻塞取最新夹爪坐标的**浮点**版本 (x_mm, y_mm)；没有新帧返回 None。

        手眼标定/坐标回算要用这个：整数版四舍五入会丢 0.4mm，几次叠加就吃掉精度。
        """
        self._pump()
        pos = self.latest_position_f
        self.latest_position_f = None
        return pos

    def position_alive(self):
        """串口在用且最近收到过合法坐标帧 → True（判断回传是真通还是假通）。"""
        return bool(self.position_seq) and self.pos_valid is not None

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