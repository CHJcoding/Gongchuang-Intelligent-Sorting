#!/bin/bash
# 摄像头调参页（对齐官方标定教程 §2.2）
cd /root/vision_ws
source /opt/tros/humble/setup.bash
exec python3 /root/vision_ws/tools/cam_tune_server.py --port 8002
