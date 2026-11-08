#!/bin/bash
set +u
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter

pkill -f "ros2 topic echo" 2>/dev/null
pkill -f "sort_vision[_]node" 2>/dev/null
sleep 1

# 1) 起节点：自动触发关闭，等 /sort/start
nohup python3 sort_vision_node.py --ros-args \
  -p auto_trigger:=false -p serial_sim:=true > /tmp/node4.log 2>&1 &
sleep 5

# 2) 起订阅（节点已存在，话题类型可发现）
timeout 10 ros2 topic echo /vision/result --once > /tmp/res2.txt 2>/dev/null &
sleep 2

# 3) 手动触发
ros2 topic pub /sort/start std_msgs/msg/Bool "data: true" --once > /dev/null 2>&1
sleep 4

pkill -f "sort_vision[_]node" 2>/dev/null
pkill -f "ros2 topic echo" 2>/dev/null
sleep 1

echo "===== /vision/result JSON ====="
cat /tmp/res2.txt
echo ""
echo "===== 节点日志 ====="
grep -E "启动信号|目标|SIM" /tmp/node4.log | tail -5