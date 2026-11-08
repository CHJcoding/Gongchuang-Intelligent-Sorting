#!/bin/bash
# 一键启动业务节点（判色 + 坐标 + 串口）
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter || exit 1
exec python3 sort_vision_node.py "$@"
