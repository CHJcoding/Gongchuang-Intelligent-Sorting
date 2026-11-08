
# -*- coding: utf-8 -*-
"""帧字节自检：确认我们"真正发到线上"的 10 字节与预期逐字节一致。

用法：
  python3 tools/verify_frame_bytes.py --shape cube --color red --x 78 --y 33
  python3 tools/verify_frame_bytes.py --shape cube --color red --x 78 --y 33 --diff "AA 55 05 00 4E 00 21 00 EA 0D"

检查项：
  1) 组帧结果 + CRC8(poly 0x07) 独立重算
  2) 经 pty 模拟真实 CDC ACM（默认 OPOST|ONLCR 开着）用 pyserial 发一遍，回读比对
  3) 与外部给的参考帧做逐字节 diff（--diff）
"""
import argparse
import os
import pty
import select
import sys
import termios

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_HERE), "vision_sorter"),
           os.path.join(os.path.dirname(_HERE), "rdk", "vision_sorter"),
           "/root/vision_ws/vision_sorter"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import comm  # noqa: E402


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def wire_roundtrip(payload: bytes) -> bytes:
    """在 pty 上模拟真机：从端 tty 默认带 OPOST|ONLCR，pyserial 写，主端读。"""
    m, s = pty.openpty()
    a = termios.tcgetattr(s)
    a[1] |= (termios.OPOST | termios.ONLCR | termios.OCRNL)
    a[0] |= (termios.ICRNL | termios.INLCR | termios.IGNCR)
    termios.tcsetattr(s, termios.TCSANOW, a)
    path = os.ttyname(s)
    try:
        import serial
        ser = serial.Serial(path, 115200, timeout=0.2)
        ser.write(payload)
        ser.flush()
        ser.close()
    finally:
        pass
    out = b""
    while True:
        r, _, _ = select.select([m], [], [], 0.3)
        if not r:
            break
        try:
            out += os.read(m, 4096)
        except OSError:
            break
    os.close(s)
    os.close(m)
    return out


def hx(b) -> str:
    return " ".join("%02X" % x for x in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--color", required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--diff", default="", help="参考帧，如 'AA 55 05 00 4E 00 21 00 EA 0D'")
    a = ap.parse_args()

    frame = comm.build_frame(a.shape, a.color, a.x, a.y)
    mid, sid, cid = comm.material_id(a.shape, a.color)
    print("物料号 %s / 形状号 %s / 颜色号 %s / 模式 %s"
          % (mid, sid, cid, getattr(comm.C, "FRAME_ID_MODE", "split")))
    print("组帧      : %s" % hx(frame))
    print("长度      : %d" % len(frame))
    print("帧头/帧尾 : %s / %s" % (hx(frame[:2]), hx(frame[9:])))
    print("X/Y 回读  : %d / %d"
          % (int.from_bytes(frame[4:6], "little", signed=True),
             int.from_bytes(frame[6:8], "little", signed=True)))

    ok = True
    my_crc = crc8(frame[:8])
    if my_crc != frame[8]:
        ok = False
        print("[FAIL] CRC 不一致：我算 %02X，帧里 %02X" % (my_crc, frame[8]))
    else:
        print("[OK]   [8] = CRC8(poly 07) 覆盖 [0..7] = %02X" % my_crc)

    got = wire_roundtrip(frame)
    if got == frame:
        print("[OK]   经 pty(模拟真机 tty 属性) + pyserial 发出去，回读逐字节一致")
    else:
        ok = False
        print("[FAIL] 线上实到 %s" % hx(got))

    if a.diff:
        ref = bytes(int(t, 16) for t in a.diff.replace(",", " ").split())
        print("参考帧    : %s" % hx(ref))
        if len(ref) != len(frame):
            ok = False
            print("[FAIL] 长度不同：%d vs %d" % (len(ref), len(frame)))
        else:
            diffs = [i for i in range(len(ref)) if ref[i] != frame[i]]
            if diffs:
                print("[DIFF] 不同的字节：" + ", ".join(
                    "[%d] 我们 %02X / 参考 %02X" % (i, frame[i], ref[i]) for i in diffs))
            else:
                print("[OK]   与参考帧逐字节一致")
    print("结论: " + ("全部通过" if ok else "有问题，别发"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
