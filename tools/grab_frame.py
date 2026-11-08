#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机空闲时直接抓一帧，确认视野内容。"""
import sys
import cv2

dev = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'
w, h = 640, 480
cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
if not cap.isOpened():
    print('open failed:', dev)
    sys.exit(1)
# 丢掉前几帧等曝光稳定
for _ in range(10):
    cap.read()
ok, frame = cap.read()
cap.release()
if not ok:
    print('read failed')
    sys.exit(1)
print('frame shape:', frame.shape)
cv2.imwrite('/root/vision_ws/scene_raw.jpg', frame)
print('saved /root/vision_ws/scene_raw.jpg')