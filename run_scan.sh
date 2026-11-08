#!/bin/bash
# 台架实拍验证：本地 BPU 推理 + 判色，不触发串口/龙门。
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter || exit 1

exec python3 sort_vision_node.py --ros-args \
  -p auto_trigger:=false \
  -p scan_all:=true \
  -p scan_interval:=0.5 \
  -p serial_sim:=true
