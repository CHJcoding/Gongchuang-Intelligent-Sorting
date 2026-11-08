#!/bin/bash
# 验证抓取流控：timer 模式下 发帧 -> 窗口屏蔽 -> 定时解锁 -> 再抓
set +u
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter

pkill -f "sort_vision[_]node" 2>/dev/null
pkill -f "fake_det[_]pub" 2>/dev/null
sleep 1

# 1) 起节点（模拟串口，自动触发）
nohup timeout 26 python3 sort_vision_node.py --ros-args \
  -p auto_trigger:=true -p serial_sim:=true > /tmp/grab_test.log 2>&1 &
sleep 3

# 2) 起假检测（=物料一直在视野里、一直没被搬走）
nohup timeout 22 python3 /root/vision_ws/tools/fake_det_pub.py > /tmp/fake.log 2>&1 &
sleep 20

pkill -f "sort_vision[_]node" 2>/dev/null
pkill -f "fake_det[_]pub" 2>/dev/null
sleep 1

echo "===== 关键日志（时间线）====="
grep -E "启动 \||目标|SIM|抓取结束|统计" /tmp/grab_test.log
echo
echo "===== 校验 ====="
fire=$(grep -c "SIM\] 下发" /tmp/grab_test.log)
done_cnt=$(grep -c "抓取结束" /tmp/grab_test.log)
echo "下发次数 = $fire"
echo "抓取结束次数 = $done_cnt"
if [ "$done_cnt" -ge 1 ]; then echo "结果: 定时解锁生效 OK"; else echo "结果: 未触发解锁 FAIL"; fi