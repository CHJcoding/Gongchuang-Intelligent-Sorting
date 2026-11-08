# -*- coding: utf-8 -*-
import time, shutil
p = "/root/vision_ws/restart_vision.sh"
s = open(p, encoding="utf-8").read()
old = """IP=$(hostname -I | tr " " "\\n" | grep -qx "10.161.15.50" && echo 10.161.15.50 || hostname -I | awk '{print $1}')"""
assert old in s, "没匹配到 IP 那行"
new = """# 取真正有默认路由的网卡地址：这台板 eth0(192.168.127.10)/usb0(192.168.128.10) 常年 NO-CARRIER
# 却仍挂着旧 IP，hostname -I 的第一个往往是它们，打印出去就是打不开的地址（实测踩过）。
IP=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.*src \\([0-9.]*\\).*/\\1/p' | head -1)
[ -z "$IP" ] && IP=$(hostname -I | awk '{print $1}')"""
shutil.copy2(p, p + ".bak." + time.strftime("%m%d_%H%M"))
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("已修：预览地址改为取默认路由网卡")
