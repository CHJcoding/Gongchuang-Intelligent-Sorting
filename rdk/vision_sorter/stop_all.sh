#!/bin/bash
# stop_all.sh —— 一键停掉所有相机相关进程（换脚本/换参数前先跑这个）
#
# 相机只能被一个进程打开。最常见的"板端报错"就是：
#   hobot_usb_cam: Cannot open device /dev/video0, exiting  (exit code -6)
# 原因不是相机坏了，是已经有一个 launch 占着它（比如终端里还挂着上一次的）。
#
# 用法： bash /root/vision_ws/stop_all.sh

echo "=== 停守护 / launch / 相机节点 ==="
screen -S det  -X quit 2>/dev/null && echo "  停了 screen det"
screen -S sort -X quit 2>/dev/null && echo "  停了 screen sort"
pkill -f cam_watchdog      2>/dev/null && echo "  停了 cam_watchdog"
pkill -f sort_vision_node  2>/dev/null && echo "  停了 sort_vision_node"
pkill -f 'ros2 launch'     2>/dev/null && echo "  停了 ros2 launch"
pkill -f dnn_node_example  2>/dev/null
pkill -f hobot_usb_cam     2>/dev/null
pkill -f hobot_codec       2>/dev/null
pkill -f 'sbin/nginx'      2>/dev/null && echo "  停了 nginx"
screen -wipe >/dev/null 2>&1
sleep 3

echo
echo "=== 检查结果 ==="
LEFT=$(ps -ef | grep -cE '[r]os2 launch|[d]nn_node_example|[h]obot_usb_cam|[c]am_watchdog|[s]ort_vision_node|[s]bin/nginx')
if [ "$LEFT" -eq 0 ]; then
    echo "✅ 全部已停（没有残留进程）"
else
    echo "⚠ 还有 $LEFT 个残留："
    ps -ef | grep -E '[r]os2 launch|[d]nn_node_example|[h]obot_usb_cam|[c]am_watchdog|[s]ort_vision_node|[s]bin/nginx' | awk '{print "   ", $2, $8, $9, $10}'
fi

echo
echo -n "相机占用情况: "
if fuser /dev/video0 >/dev/null 2>&1; then
    fuser -v /dev/video0 2>&1 | sed 's/^/  /'
else
    echo "空闲（可以启动）"
fi
echo -n "8000 端口: "
ss -lntp 2>/dev/null | grep -q ':8000' && echo "仍被占（等一下或再跑一次）" || echo "已释放"