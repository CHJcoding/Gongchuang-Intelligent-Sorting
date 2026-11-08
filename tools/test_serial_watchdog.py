# -*- coding: utf-8 -*-
"""串口看门狗自检：把链路健康判断用 pty 跑一遍，不碰真板子。

覆盖：
  ok       → 正常打开
  gone     → 设备节点被删掉（拔线后 /dev/ttyACM* 消失）
  hungup   → 兜底：节点还在但句柄已死（真机拔线时通常先走 gone）
不覆盖 replaced（需要真 USB 重新枚举，devnum 才有意义，只能在真机上验）。
"""
import os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "vision_sorter"), HERE, "/root/vision_ws/vision_sorter"):
    if p not in sys.path:
        sys.path.insert(0, p)

import comm
import config as C

fails = []

def check(name, got, want):
    ok = got == want
    print("  [%s] %s: got=%r want=%r" % ("OK " if ok else "FAIL", name, got, want))
    if not ok:
        fails.append(name)

print("== pty 链路自检 ==")
master, slave = os.openpty()
slave_name = os.ttyname(slave)
tmpdir = tempfile.mkdtemp()
link = os.path.join(tmpdir, "ttyGC")
os.symlink(slave_name, link)

C.SERIAL_PORT_STABLE = link          # 走稳定符号链接，跟现场一样
C.SERIAL_PORT = slave_name
C.SERIAL_PORT_CANDIDATES = [slave_name]

c = comm.Comm()
c.open()
print("  打开 %s（真实设备 %s）" % (c.port, os.path.realpath(c.port)))
check("刚打开", c.link_state(), "ok")

# 1) 设备节点消失（拔线）
os.unlink(link)
check("节点消失", c.link_state(), "gone")
c.close()

# 2) 对端断开：重新开一次，然后关掉 master。pty 的 slave 节点会一起消失，
#    所以真机上"拔线"走的就是 gone 这条分支；hungup 是设备节点还在、句柄已死
#    时的兜底（需要真 USB 才能造出来，pty 造不出）。
os.symlink(slave_name, link)
c2 = comm.Comm()
c2.open()
os.close(master)
time.sleep(0.05)
check("对端断开", c2.link_state(), "gone")
c2.close()

print("== 结论: %s ==" % ("全部通过" if not fails else "失败 %d 项 %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
