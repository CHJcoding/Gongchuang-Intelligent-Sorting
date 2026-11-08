#!/bin/bash
# 台架快速识别模式：扫描日志 0.3s 一次，抓取节拍压到 1s
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter || exit 1
exec python3 sort_vision_node.py --ros-args \
  -p auto_trigger:=true \
  -p scan_all:=true \
  -p scan_interval:=0.3 \
  -p grab_cycle_s:=1.0 \
  -p grab_timeout_s:=2.0