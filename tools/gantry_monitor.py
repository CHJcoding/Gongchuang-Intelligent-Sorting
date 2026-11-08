# -*- coding: utf-8 -*-
"""夹爪坐标监视器：读 STM32 回传的 0x07 坐标帧并实时显示。

为什么要它
----------
手眼标定要"真值"，但坐标只有 STM32 知道。让固件周期性回一帧坐标，
这个脚本就能实时看到夹爪在哪，标定时不用靠人眼抄示教器。

帧格式（6 字节，与下行 10 字节帧同一套 CRC）
--------------------------------------------
    [0]   0x07        帧头
    [1:3] x_int16 LE  当前 X(mm)
    [3:5] y_int16 LE  当前 Y(mm)
    [5]   crc8        对前 5 字节算，poly 0x07 / init 0x00

用法
----
    # 没有 STM32 时先验证脚本本身
    python gantry_monitor.py --simulate

    # 不知道对方到底发什么格式时：原始字节转储
    python gantry_monitor.py --raw --raw-duration 10

    # 有 STM32：注意同一时刻只能有一个程序占用串口，
    # 标定时先把 sort_vision_node 停掉（或让它来跑这个功能）
    python gantry_monitor.py --port COM5
    python gantry_monitor.py --port /dev/ttyACM0 --csv trace.csv
"""
import argparse
import random
import struct
import sys
import time

POS_HEAD = 0x07
POS_LEN = 6

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def crc8(data) -> int:
    """CRC-8/SMBUS：poly 0x07，init 0x00，不反转，xorout 0。"""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_pos(x_mm, y_mm) -> bytes:
    """组一帧坐标（给 STM32 侧当参考实现，也用于 --simulate）。"""
    x_i = max(-32768, min(32767, int(round(x_mm))))
    y_i = max(-32768, min(32767, int(round(y_mm))))
    body = bytes([POS_HEAD]) + struct.pack("<hh", x_i, y_i)
    return body + bytes([crc8(body)])


class PosParser:
    """流式解析器：喂字节，吐坐标。半包会留在缓冲里等下一次。"""

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0          # CRC 失败次数
        self.dropped = 0      # 跳过的杂字节数

    def feed(self, data: bytes):
        self.buf.extend(data)
        out = []
        i, n = 0, len(self.buf)
        while i < n:
            if self.buf[i] != POS_HEAD:
                i += 1
                self.dropped += 1
                continue
            if i + POS_LEN > n:
                break                                  # 半包，等后续
            body = bytes(self.buf[i:i + POS_LEN - 1])
            if crc8(body) == self.buf[i + POS_LEN - 1]:
                x, y = struct.unpack("<hh", body[1:5])
                out.append((x, y))
                i += POS_LEN
            else:
                self.bad += 1                          # CRC 错：丢 1 字节重找头
                i += 1
        del self.buf[:i]
        return out


def simulate_stream(n=40, seed=42):
    """造一段带噪声/错包的模拟字节流，返回 (bytes, 预期能解析出的帧数)。"""
    rng = random.Random(seed)
    out = bytearray()
    expected = 0
    for i in range(n):
        x = int(120 + 60 * (1 if i % 20 < 10 else -1))
        y = int(80 + 5 * i)
        if rng.random() < 0.2:
            out += bytes([rng.randrange(256)])         # 杂字节
            out[-1] = out[-1] if out[-1] != POS_HEAD else 0x00
        out += encode_pos(x, y)
        expected += 1
        if rng.random() < 0.15:
            out += encode_pos(x, y)                    # 重复帧（丢帧无所谓）
            expected += 1
        if rng.random() < 0.1:
            bad = bytearray(encode_pos(x, y))
            bad[-1] ^= 0xFF                            # 弄坏 CRC -> 应当被丢弃
            out += bytes(bad)
    return bytes(out), expected


def run_simulate(n=40):
    print("模拟模式：字节流里混入杂字节/重复帧/坏 CRC，验证解析器健壮性")
    stream, expected = simulate_stream(n)
    parser = PosParser()
    frames = parser.feed(stream)
    print("解析出 %d 帧（预期 %d）" % (len(frames), expected))
    for i, (x, y) in enumerate(frames[:8]):
        print("  #%d  X=%d mm  Y=%d mm" % (i, x, y))
    if len(frames) > 8:
        print("  ...")
    print("CRC 失败 %d 次，跳过杂字节 %d 个" % (parser.bad, parser.dropped))
    ok = (len(frames) == expected)
    print("%s" % ("自检通过" if ok else "自检失败：帧数不符"))
    return 0 if ok else 1


def run_raw(port, baud, duration=None):
    """原始字节转储：不管什么格式，收到什么就按十六进制打出来。

    联调第一天最有用——如果 STM32 发的其实不是 0x07 帧（比如回了 ASCII "OK"、
    或者自定义的 AA 55 帧），普通模式会一帧都解不出来、看起来像"串口没数据"，
    而这里能直接把真实字节看清。
    """
    try:
        import serial
    except ImportError:
        print("没装 pyserial：pip3 install pyserial")
        return 2

    ser = serial.Serial(port, baud, timeout=0.05)
    print("已打开 %s @%d —— 原始转储模式（Ctrl+C 退出）" % (port, baud))
    never = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz0123456789 .,:;-+*/()[]{}<>=_#%")
    t0 = time.time()
    total = 0
    try:
        while True:
            data = ser.read(256)
            if data:
                total += len(data)
                hexs = " ".join("%02X" % b for b in data)
                text = "".join(chr(b) if chr(b) in never else "." for b in data)
                print("[%7.2fs] %3d B | %s | %s"
                      % (time.time() - t0, len(data), hexs, text))
            if duration and time.time() - t0 >= duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\n共收到 %d 字节 / %.1f 秒" % (total, time.time() - t0))
    if total == 0:
        print("⚠ 一个字节都没收到：检查 TX/RX 是否接反、波特率、以及 STM32 是否真的在发")
        return 1
    return 0


def run_serial(port, baud, csv_path=None, limit=None):
    try:
        import serial
    except ImportError:
        print("没装 pyserial：pip3 install pyserial（或用 --simulate 先验证）")
        return 2

    ser = serial.Serial(port, baud, timeout=0.05)
    print("已打开 %s @%d，等待 0x07 坐标帧（Ctrl+C 退出）" % (port, baud))
    print("提示：同一时刻只能有一个程序占用串口，标定时请先停掉 sort_vision_node")

    parser = PosParser()
    f = None
    if csv_path:
        f = open(csv_path, "w", encoding="utf-8")
        f.write("t,x_mm,y_mm\n")
        print("记录到 %s" % csv_path)

    t0 = time.time()
    got = 0
    try:
        while True:
            data = ser.read(64)
            if not data:
                continue
            for (x, y) in parser.feed(data):
                got += 1
                ts = time.time() - t0
                print("\rX=%7d mm  Y=%7d mm  | t=%6.2fs  帧=%d  "
                      "坏包=%d 杂字节=%d   " % (x, y, ts, got,
                                                parser.bad, parser.dropped),
                      end="", flush=True)
                if f:
                    f.write("%.3f,%d,%d\n" % (ts, x, y))
                if limit and got >= limit:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        print()
        ser.close()
        if f:
            f.close()
            print("已保存 %s" % csv_path)
        print("共收到 %d 帧，坏包 %d，杂字节 %d"
              % (got, parser.bad, parser.dropped))
    if got == 0:
        print("⚠ 一帧都没收到：检查 STM32 是否在发 0x07 帧、波特率是否一致、"
              "TX/RX 是否接反")
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="STM32 夹爪坐标帧监视器")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="串口，如 COM5 或 /dev/ttyACM0（STM32 是 USB CDC）")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", help="把收到的坐标记成 CSV")
    ap.add_argument("--limit", type=int, help="收到多少帧后退出（测试用）")
    ap.add_argument("--simulate", action="store_true", help="不连串口，验证解析器")
    ap.add_argument("--self-test", action="store_true", help="同 --simulate")
    ap.add_argument("--raw", action="store_true",
                    help="原始字节转储（不知道对方发的是什么格式时先用这个）")
    ap.add_argument("--raw-duration", type=float, default=None,
                    help="--raw 模式跑多少秒后自动退出")
    args = ap.parse_args(argv)

    if args.simulate or args.self_test:
        return run_simulate()
    if args.raw:
        return run_raw(args.port, args.baud, args.raw_duration)
    return run_serial(args.port, args.baud, args.csv, args.limit)


if __name__ == "__main__":
    sys.exit(main())