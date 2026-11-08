#!/bin/bash
set +u
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter

pkill -f "ros2 topic echo" 2>/dev/null
pkill -f "sort_vision[_]node" 2>/dev/null
sleep 1

# 1) 先起订阅，等它连上
timeout 30 ros2 topic echo /vision/result --field data > /tmp/res.txt 2>/dev/null &
ECHO_PID=$!
sleep 2

# 2) 再起节点（首帧即触发一次下发）
nohup timeout 12 python3 sort_vision_node.py --ros-args \
  -p auto_trigger:=true -p serial_sim:=true > /tmp/node3.log 2>&1 &

sleep 12
pkill -f "sort_vision[_]node" 2>/dev/null
kill $ECHO_PID 2>/dev/null
pkill -f "ros2 topic echo" 2>/dev/null
sleep 1

echo "===== /vision/result 捕获到的 JSON ====="
head -c 1200 /tmp/res.txt
echo ""
echo "===== 节点下发日志 ====="
grep -E "目标|SIM" /tmp/node3.log | tail -4