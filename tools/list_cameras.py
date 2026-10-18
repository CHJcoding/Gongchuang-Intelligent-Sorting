# -*- coding: utf-8 -*-
"""列出所有可用摄像头编号"""
import cv2

print("正在探测摄像头（0~9）...")
available = []
for i in range(10):
    # DSHOW 是 Windows 下最稳的后端
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 读一帧确认不是假打开
        ok, _ = cap.read()
        if ok:
            print(f"  相机 {i}: 可用  {w}x{h}")
            available.append(i)
        else:
            print(f"  相机 {i}: 能打开但读不到画面")
        cap.release()
    else:
        print(f"  相机 {i}: 不可用")

print(f"\n可用相机编号: {available}")
if available:
    print(f"检测脚本里用 --source {available[0]} 即可")
