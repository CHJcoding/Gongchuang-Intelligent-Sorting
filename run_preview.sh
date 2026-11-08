#!/bin/bash
# 网页预览（MJPEG，端口 8001）
source /opt/tros/humble/setup.bash
cd /root/vision_ws
P=/root/vision_ws/preview_server.py
[ -f "$P" ] || P=/root/vision_ws/tools/preview_server.py
exec python3 "$P" --port 8001 --topic /image_annotated
