#!/bin/bash
# 台架演示用启动器：没有 hand_eye.json、没有 STM32 时，两道安全闸都显式放开。
# 生产环境请直接用 run_sort.sh（默认闭嘴不瞎发）。
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter
exec python3 sort_vision_node.py --ros-args \
    -p serial_sim:=true \
    -p allow_uncalibrated:=true \
    "$@"