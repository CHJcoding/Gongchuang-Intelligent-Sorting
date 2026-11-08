#!/bin/bash
set +u
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter

# 清掉可能残留的 echo 进程
pkill -f "ros2 topic echo" 2>/dev/null
pkill -f "sort_vision[_]node" 2>/dev/null
sleep 1

nohup python3 sort_vision_node.py --ros-args \
  -p auto_trigger:=true -p serial_sim:=true > /tmp/node2.log 2>&1 &
sleep 6

echo "===== /vision/result 一条（JSON）====="
timeout 6 ros2 topic echo /vision/result --once --field data > /tmp/result.txt 2>/dev/null
cat /tmp/result.txt
echo ""
echo "===== 节点日志（最近 5 行）====="
grep -E "目标|SIM|统计" /tmp/node2.log | tail -5

pkill -f "sort_vision[_]node" 2>/dev/null
pkill -f "ros2 topic echo" 2>/dev/null
echo "===== 已清理 ====="