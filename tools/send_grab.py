# -*- coding: utf-8 -*-
"""手动给 STM32 下发一帧抓取指令（调试/标定用，绕过 safety 闸）。

协议里只有这一种下行帧，没有"点动"指令：发一帧 = 让 32 去 (x,y) 抓物料。
因此每次发送都会真的动，默认要求 --yes 确认。

用法（板端 /root/vision_ws）：
  python3 tools/send_grab.py --x 88 --y 33 --dry          # 只打印字节，不发
  python3 tools/send_grab.py --x 88 --y 33 --yes          # 真发，发完读 5s 回包
  python3 tools/send_grab.py --x 88 --y 33 --yes --read-s 8
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'vision_sorter'))

import comm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, required=True, help="目标 X(mm)")
    ap.add_argument("--y", type=float, required=True, help="目标 Y(mm)")
    ap.add_argument("--shape", default="", help="形状名，如 cube/cylinder；空=未知(255)")
    ap.add_argument("--color", default="", help="颜色名，如 red；空=未知(255)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry", action="store_true", help="只打印帧，不打开串口")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    ap.add_argument("--read-s", type=float, default=5.0, help="发送后读回包秒数")
    a = ap.parse_args()

    frame = comm.build_frame(a.shape, a.color, a.x, a.y)
    mid, sid, cid = comm.material_id(a.shape, a.color)
    print("物料编号: %s  形状号: %s  颜色号: %s" % (mid, sid, cid))
    print("帧(%d 字节): %s" % (len(frame), " ".join("%02X" % b for b in frame)))
    if a.dry:
        print("[dry] 未发送")
        return

    if not a.yes:
        print("这一帧会让 32 真动到 (%.1f, %.1f) 并执行抓取。" % (a.x, a.y))
        if input("继续? 输入 yes: ").strip().lower() != "yes":
            print("已取消")
            return

    import serial
    s = serial.Serial(a.port, a.baud, timeout=0.2)
    s.reset_input_buffer()
    s.write(frame); s.flush()
    print("[已发] %s" % a.port)
    t0 = time.time(); buf = b""
    while time.time() - t0 < a.read_s:
        buf += s.read(256)
    s.close()
    print("[回包 %d 字节] %s" % (len(buf), repr(buf[:300])))


if __name__ == "__main__":
    main()
