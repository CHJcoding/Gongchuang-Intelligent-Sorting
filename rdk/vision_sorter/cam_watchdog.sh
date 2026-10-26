#!/bin/bash
# cam_watchdog.sh —— 检测节点守护（v2：温和重启，避免自己把相机搞掉线）
#
# v1 的教训：
#   用 pkill 硬杀 + 立刻重启，会在推流中途把 UVC 设备复位，内核报
#   "USB disconnect" / "Failed to resubmit video URB (-19)"，新实例又抢不到，
#   于是"重启 -> 掉线 -> 打不开 -> 再重启"自我放大。
#   实测：单进程干净运行 90 秒 0 次掉线；反复 pkill 时几十秒就掉一次。
#
# v2 策略：
#   1) 启动前先清干净（含孤儿 websocket —— 每条死掉的 launch 都会留一个）
#   2) 等相机真正可用（探测打开成功）再启动，不硬来
#   3) 重启先 SIGINT 优雅退出，等进程真的没了再动
#   4) 判活只看"日志是否还在长" —— 节点正常时每秒都在写 fps 行
#   5) 频繁失败就退避，绝不疯狂重启
#
# 用法：
#   screen -dmS det /root/vision_ws/cam_watchdog.sh
#   screen -r det              # 看日志（Ctrl+A D 退出但不停止）
#   screen -S det -X quit      # 停止
# 日志：/root/vision_ws/watchdog.log / /root/vision_ws/det.log

LAUNCH_LOG=/root/vision_ws/det.log
WD_LOG=/root/vision_ws/watchdog.log
# 相机设备：优先用 udev 稳定链接 /dev/cam0。
# 为什么不用 /dev/video0：USB 一旦弹一下重枚举，节点号会变（实测变成 /dev/video1），
# 写死 /dev/video0 的 launch 会永远报 "Cannot open device"。
# 规则见 /etc/udev/rules.d/60-gc-camera.rules（按 idVendor 0c45 建 cam0 链接）。
CAM_DEV="${GC_CAM_DEV:-/dev/cam0}"
CHECK=5              # 每 5 秒检查一次
STALE=18             # 日志超过 18 秒没更新 = 卡死
QUICK_WINDOW=120     # 退避窗口
MAX_QUICK=4          # 窗口内最多重启几次

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WD_LOG"; }

# ---- 设备解析：cam0 不在就扫一遍 /dev/video*，挑能出 MJPG 的那个 ----
resolve_cam() {
    if [ -e "$CAM_DEV" ]; then
        return 0
    fi
    local v
    for v in /dev/video*; do
        [ -e "$v" ] || continue
        if timeout 5 v4l2-ctl -d "$v" --list-formats 2>/dev/null | grep -q MJPG; then
            CAM_DEV="$v"
            log "⚠ /dev/cam0 不存在，退用 $v（建议补 udev 规则，别让节点号漂）"
            return 0
        fi
    done
    return 1
}

# ---- 清掉所有相机相关残留（含孤儿 websocket）----
cleanup_all() {
    pkill -f 'ros2 launch dnn_node_example' 2>/dev/null
    sleep 1
    pkill -f 'dnn_node_example/example' 2>/dev/null
    pkill -f hobot_usb_cam  2>/dev/null
    pkill -f hobot_codec    2>/dev/null
    pkill -f 'lib/websocket/websocket' 2>/dev/null
    pkill -f 'sbin/nginx'   2>/dev/null
    sleep 2
    local left
    left=$(ps -ef | grep -cE '[r]os2 launch dnn_node|[d]nn_node_example/example|[h]obot_usb_cam|[h]obot_codec|[l]ib/websocket/websocket')
    [ "$left" -eq 0 ] || log "⚠ 清理后仍有 $left 个残留进程"
}

# ---- 等相机可用：能真的打开并取到格式才算 ready ----
wait_camera_ready() {
    local i
    for i in $(seq 1 15); do
        if [ -e "$CAM_DEV" ] && timeout 5 v4l2-ctl -d "$CAM_DEV" --get-fmt-video >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

log "================ 守护启动 (v2) ================"
RESTARTS=0
WINDOW_START=$(date +%s)

while true; do
    cleanup_all
    resolve_cam
    export GC_CAM_DEV="$CAM_DEV"
    if ! wait_camera_ready; then
        log "⚠ 相机 $CAM_DEV 打不开（设备消失或不可用），等 15s 再试"
        sleep 15
        continue
    fi

    : > "$LAUNCH_LOG"
    log "相机就绪，启动检测节点..."
    setsid /root/vision_ws/run_det.sh >> "$LAUNCH_LOG" 2>&1 &
    LAUNCH_PID=$!

    # ---- 判活：日志是否还在增长（正常时每秒都有 fps 行）----
    FAIL=""
    START_TS=$(date +%s)
    while true; do
        sleep "$CHECK"
        UP=$(( $(date +%s) - START_TS ))

        if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
            FAIL="launch 主进程退出"; break
        fi
        [ -e "$CAM_DEV" ] || { FAIL="相机设备消失"; break; }

        # 前 25 秒是 warmup（BPU 初始化慢），不判卡死
        if [ "$UP" -gt 25 ]; then
            AGE=$(( $(date +%s) - $(stat -c %Y "$LAUNCH_LOG" 2>/dev/null || echo 0) ))
            if [ "$AGE" -gt "$STALE" ]; then
                FAIL="日志 ${AGE}s 没更新（节点卡死/相机掉了）"; break
            fi
            if grep -qaE 'hobot_usb_cam-1\]: process has died|Cannot open device' "$LAUNCH_LOG"; then
                FAIL="相机节点死亡"; break
            fi
        fi
    done

    log "需要重启：$FAIL（这次跑了 $(( $(date +%s) - START_TS ))s）"

    # ---- 优雅停止 ----
    kill -INT "$LAUNCH_PID" 2>/dev/null
    for i in 1 2 3 4 5 6; do
        kill -0 "$LAUNCH_PID" 2>/dev/null || break
        sleep 1
    done
    cleanup_all

    # ---- 退避 ----
    now=$(date +%s)
    if [ $((now - WINDOW_START)) -gt "$QUICK_WINDOW" ]; then
        WINDOW_START=$now; RESTARTS=0
    fi
    RESTARTS=$((RESTARTS+1))
    if [ "$RESTARTS" -ge "$MAX_QUICK" ]; then
        log "⚠ ${QUICK_WINDOW}s 内已重启 ${RESTARTS} 次，检查相机线和供电；暂停 60s"
        sleep 60
        WINDOW_START=$(date +%s); RESTARTS=0
    fi
    sleep 3
done