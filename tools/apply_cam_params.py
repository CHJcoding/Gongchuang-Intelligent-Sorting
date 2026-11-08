#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开机把 cam_params.json 里的摄像头参数写到 /dev/video0（由网页调参工具 /save 生成）。"""
import json
import os
import subprocess
import sys
import time

P = "/root/vision_ws/cam_params.json"
DEV = "/dev/video0"

def main():
    if not os.path.exists(P):
        print("没有 %s，跳过" % P)
        return 0
    try:
        vals = json.load(open(P, encoding="utf-8"))
    except Exception as e:
        print("读 %s 失败: %s" % (P, e))
        return 1
    ok = fail = 0
    for k, v in vals.items():
        r = subprocess.run(["v4l2-ctl", "-d", DEV, "-c", "%s=%s" % (k, v)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
            print("设置 %s=%s 失败: %s" % (k, v, (r.stderr or "").strip()))
    print("摄像头参数已应用: 成功 %d 失败 %d" % (ok, fail))
    return 0

if __name__ == "__main__":
    sys.exit(main())
