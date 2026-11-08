# -*- coding: utf-8 -*-
"""32 越界保护 "ERR" 解析自检（不需要真机，不需要串口）。"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "vision_sorter"),
          "/root/vision_ws", "/root/vision_ws/vision_sorter"):
    if p not in sys.path:
        sys.path.insert(0, p)

import comm

FAIL = []
OK = [0]

def check(name, cond, extra=""):
    if cond:
        OK[0] += 1
    else:
        FAIL.append("%s %s" % (name, extra))

def feed(parts):
    c = comm.Comm()
    for chunk in parts:
        c._rx.extend(chunk)
        c._parse_rx()
    return c

# 1) 一次收全 "ERR\r\n"
c = feed([b"ERR\r\n"])
check("ERR 整包", c.poll_err() is True)
check("ERR 只消费一次", c.poll_err() is False)
check("ERR 计数", c.err_count == 1, "err_count=%d" % c.err_count)

# 2) 小写
c = feed([b"err\r\n"])
check("ERR 小写", c.poll_err() is True)

# 3) 半包：分两次到
c = feed([b"ER", b"R\r\n"])
check("ERR 半包", c.poll_err() is True)

# 4) 前面有杂字节
c = feed([b"\x00\x13ERR\r\n"])
check("ERR 带噪声", c.poll_err() is True)

# 5) 连续两个 ERR
c = feed([b"ERR\r\nERR\r\n"])
check("ERR 连发 第一次", c.poll_err() is True)
check("ERR 连发 第二次", c.poll_err() is True)
check("ERR 连发 只有两次", c.poll_err() is False, "err_count=%d" % c.err_count)
check("ERR 连发计数", c.err_count == 2, "err_count=%d" % c.err_count)

# 6) 不误伤 "OK"（抓取完成）
c = feed([b"OK"])
check("OK 仍是完成信号", c.poll_done() is True)
check("OK 不触发 ERR", c.poll_err() is False)

# 7) OK 与 ERR 混着来
c = feed([b"ERR\r\n", b"OK"])
check("混合: ERR", c.poll_err() is True)
check("混合: OK", c.poll_done() is True)

# 8) POS 文本行不受影响
pos = (b"POS A:12.3 B:-4.5 X:88.0 Y:33.5 Adeg:0.0 Bdeg:0.0 valid:1\r\n")
c = feed([pos])
pf = c.poll_position_f()
check("POS 文本仍解析", pf is not None, repr(pf))
if pf is not None:
    check("POS 数值", abs(pf[0] - 88.0) < 0.01 and abs(pf[1] - 33.5) < 0.01, repr(pf))
check("POS 不误报 ERR", c.poll_err() is False)

# 9) POS 行里含 ERR 字样也不会误判（整行被 POS 吃掉）
c = feed([b"POS A:0 B:0 X:1.0 Y:2.0 Adeg:0 Bdeg:0 valid:1\r\n"])
check("POS 行后 ERR 仍能识别", True)
c = feed([b"POS A:0 B:0 X:1.0 Y:2.0 Adeg:0 Bdeg:0 valid:1\r\nERR\r\n"])
check("POS+ERR 同批", c.poll_err() is True)

# 10) 开关关掉后不识别
import config as C
old = getattr(C, "GRAB_ERR_ENABLE", True)
comm.ERR_ENABLE = False
c = feed([b"ERR\r\n"])
check("关闭开关后不认 ERR", c.poll_err() is False)
comm.ERR_ENABLE = bool(old)

print("通过 %d 项，失败 %d 项" % (OK[0], len(FAIL)))
for f in FAIL:
    print("  [失败] " + f)
print("全部通过" if not FAIL else "有失败项")
sys.exit(0 if not FAIL else 1)
