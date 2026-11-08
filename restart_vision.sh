#!/bin/bash
# restart_vision.sh —— 一键重启整条视觉链路（相机 → 识别 → 预览）
#
# 用法：  bash /root/vision_ws/restart_vision.sh
# 停：    bash /root/vision_ws/stop_all.sh
#
# 快版思路：所有进程同时起，不再固定 sleep/轮询 ros2 topic hz，
#   改成「起完立刻用轻量探针验收就绪」：/image 出帧、识别节点打第一条 [状态]、预览 healthz 通。
#   实测 25.6s → 见脚本末尾打印的 耗时。

K=/root/vision_ws
cd "$K" || exit 1
source /opt/tros/humble/setup.bash

say() { echo "[$(date '+%H:%M:%S')] $*"; }
T0=$(date +%s%N)

say "1/4 停掉旧进程"
screen -S det -X quit >/dev/null 2>&1
screen -S prev -X quit >/dev/null 2>&1
screen -S sort -X quit >/dev/null 2>&1
pkill -f preview_server.py >/dev/null 2>&1
pkill -f '[c]am_watchdog.sh' >/dev/null 2>&1
pkill -f '[c]am_only.sh' >/dev/null 2>&1
pkill -f sort_vision_node >/dev/null 2>&1
pkill -f 'ros2 launch' >/dev/null 2>&1
pkill -f dnn_node_example >/dev/null 2>&1
pkill -f hobot_usb_cam >/dev/null 2>&1
pkill -f hobot_codec >/dev/null 2>&1
pkill -f 'lib/websocket/websocket' >/dev/null 2>&1
pkill -f 'sbin/nginx' >/dev/null 2>&1
# 等进程真的退出（最多 3s），不睡固定时长
for _ in $(seq 1 50); do
    pgrep -f "[h]obot_usb_cam|[s]ort_vision_node.py|[p]review_server.py|[c]am_watchdog.sh|[c]am_only.sh|[r]os2 launch" >/dev/null || break
    sleep 0.1
done
screen -wipe >/dev/null 2>&1

# 旧 watchdog 被杀后，确认 flock 已释放再启动新实例。
LOCK_READY=0
for _ in $(seq 1 50); do
    if ! pgrep -f '[c]am_watchdog.sh' >/dev/null 2>&1 && \
       flock -n /var/run/gc_cam_watchdog.lock true; then
        LOCK_READY=1
        break
    fi
    sleep 0.1
done
if [ "$LOCK_READY" -ne 1 ]; then
    say "✗ cam_watchdog 锁未释放，拒绝启动第二条相机链路"
    say "  查：ps -ef | grep -E 'cam_watchdog|cam_only'"
    exit 1
fi

say "2/4 同时拉起相机 / 识别 / 预览"
screen -dmS det "$K/cam_watchdog.sh"
: > "$K/sort.log"
screen -dmS sort bash -c "$K/run_sort.sh > $K/sort.log 2>&1"
: > "$K/preview.log"
screen -dmS prev bash -c "$K/run_preview.sh > $K/preview.log 2>&1"

say "3/4 等就绪（/image 出帧 → 识别首条状态 → 预览可用）"
if python3 "$K/tools/wait_image.py" 20 3 >/dev/null 2>&1; then
    say "    /image 已出帧"
else
    say "✗ /image 20s 没出帧。查：tail -30 $K/det.log   或   screen -r det"
    exit 1
fi

for _ in $(seq 1 150); do          # 识别节点打第一条 [状态]（含模型加载）最多 30s
    grep -qa "\[状态\]" "$K/sort.log" && break
    sleep 0.2
done
for _ in $(seq 1 50); do           # 预览 healthz 最多 10s
    curl -s -m 1 http://127.0.0.1:8001/healthz >/dev/null 2>&1 && break
    sleep 0.2
done

echo
say "==== 状态 ===="
echo -n "  相机节点: "; pgrep -f hobot_usb_cam >/dev/null && echo "运行中" || echo "✗ 没起来"
echo -n "  识别节点: "; pgrep -f "python3 sort_vision_node.py" >/dev/null && echo "运行中" || echo "✗ 没起来"
echo -n "  预览服务: "; pgrep -f preview_server.py >/dev/null && echo "运行中" || echo "✗ 没起来"
echo -n "  预览数据: "; curl -s -m 4 http://127.0.0.1:8001/healthz || echo "(没响应)"
echo
grep -a "\[状态\]" "$K/sort.log" | tail -1 | cut -c1-200
echo
# 取真正有默认路由的网卡地址：eth0/usb0 常年 NO-CARRIER 却仍挂着旧 IP
IP=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.*src \([0-9.]*\).*/\1/p' | head -1)
[ -z "$IP" ] && IP=$(hostname -I | awk '{print $1}')
echo "  预览地址: http://$IP:8001/"
echo "  识别日志: tail -f $K/sort.log"
echo "  screen:   screen -r det | screen -r sort | screen -r prev"
say "总耗时: $(( ($(date +%s%N) - T0) / 1000000 )) ms"
