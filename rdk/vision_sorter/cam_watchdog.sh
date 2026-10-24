#!/bin/bash
# cam_watchdog.sh —— 检测节点的自动重启守护
#
# 背景：C100 相机接在 Genesys USB2 hub 后面，供电余量不足，会每隔几分钟
# 从 USB 总线上掉一次（内核 -71/-19）。掉了以后 hobot_usb_cam 直接 abort(-6)
# 退出，而 ros2 launch 默认【不会】重启死掉的节点 —— 于是整条链就永久卡死，
# 表现就是一堆 "has not received image" + "process has died"。
#
# 这个脚本负责：掉了就整条 launch 重启，让流程自己恢复。
#
# 用法：
#   screen -dmS det /root/vision_ws/cam_watchdog.sh     # 后台守护
#   screen -r det                                       # 看日志
#   screen -S det -X quit                               # 停止
# 日志：/root/vision_ws/watchdog.log / /root/vision_ws/det.log

LAUNCH_LOG=/root/vision_ws/det.log
WD_LOG=/root/vision_ws/watchdog.log
VIDEO=/dev/video0
CHECK_INTERVAL=5          # 秒
MAX_QUICK_RESTART=5       # 60 秒内连续重启超过这个数就退避
QUICK_WINDOW=60

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WD_LOG"
}

# 重启原因判定：只看这几个已知失败特征（不用跑 ros2 节点，开销为 0）
# $1 = 本周期已运行秒数（启动初期有 warmup，不判"收不到图"）
need_restart() {
    local uptime_s="$1"
    # 相机设备节点没了
    [ ! -e "$VIDEO" ] && { echo "相机设备 $VIDEO 消失"; return 0; }
    # 启动失败
    grep -qa 'Cannot open device' "$LAUNCH_LOG" 2>/dev/null && { echo "hobot_usb_cam 打不开设备（多半是相机被别的进程占着）"; return 0; }
    # 相机节点死了（exit -6 / -11）
    grep -qa 'hobot_usb_cam-1\]: process has died' "$LAUNCH_LOG" 2>/dev/null && { echo "hobot_usb_cam 进程死亡"; return 0; }
    # 下游连续 5 秒收不到图（掉线后典型症状）——启动前 30 秒不算，那是 warmup
    if [ "$uptime_s" -gt 30 ]; then
        grep -qa 'has not received image' "$LAUNCH_LOG" 2>/dev/null && { echo "下游收不到图像"; return 0; }
    fi
    return 1
}

kill_launch() {
    pkill -f 'ros2 launch dnn_node_example' 2>/dev/null
    pkill -f 'dnn_node_example/example'     2>/dev/null
    pkill -f hobot_usb_cam                  2>/dev/null
    pkill -f hobot_codec                    2>/dev/null
    pkill -f 'sbin/nginx'                   2>/dev/null
    sleep 3
}

log "================ 守护启动 ================"
log "监控 $VIDEO，每 ${CHECK_INTERVAL}s 检查一次"

RESTARTS=0
WINDOW_START=$(date +%s)

while true; do
    # ---- 启动 ----
    kill_launch
    : > "$LAUNCH_LOG"
    log "启动检测节点..."
    setsid /root/vision_ws/run_det.sh >> "$LAUNCH_LOG" 2>&1 &
    LAUNCH_PID=$!
    sleep 15                       # 给它起来的时间（含 BPU 初始化）

    # ---- 监控 ----
    START_TS=$(date +%s)
    while kill -0 "$LAUNCH_PID" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"
        UP=$(( $(date +%s) - START_TS ))
        REASON=$(need_restart "$UP")
        if [ -n "$REASON" ]; then
            log "发现问题（已运行 ${UP}s）：$REASON —— 重启"
            break
        fi
    done

    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        log "launch 主进程已退出 —— 重启"
    fi

    # ---- 退避：防止相机彻底坏了之后疯狂重启 ----
    now=$(date +%s)
    if [ $((now - WINDOW_START)) -gt "$QUICK_WINDOW" ]; then
        WINDOW_START=$now
        RESTARTS=0
    fi
    RESTARTS=$((RESTARTS + 1))
    if [ "$RESTARTS" -gt "$MAX_QUICK_RESTART" ]; then
        log "⚠ ${QUICK_WINDOW}s 内已重启 ${RESTARTS} 次，相机可能已彻底失效（检查线和供电），等 60s 再试"
        kill_launch
        sleep 60
        WINDOW_START=$(date +%s)
        RESTARTS=0
    fi
done