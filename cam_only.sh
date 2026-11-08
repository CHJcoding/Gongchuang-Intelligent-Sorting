#!/bin/bash
# cam_only.sh —— 只跑「相机 + 网页预览」，并给守护脚本喂心跳。
#
# 背景：
#   v8 模型让官方 dnn_node_example 必崩（float32 reg vs int32 parser），
#   业务节点自己推理，不需要它。详见 cam_only.launch.py 头部注释。
#
# 这个脚本做三件事：
#   1) 起 cam_only.launch.py（usb_cam + websocket），用 ${GC_CAM_DEV:-/dev/cam0}
#   2) 相机节点起来后把 brightness/saturation 打到标定时用的值
#      （默认 brightness=32、saturation=128；可用环境变量覆盖）
#   3) 每 5 秒往 det.log 写一行心跳 —— 守护脚本靠「日志是否还在长」判活，
#      没有心跳它 18 秒后就会误判卡死并重启整条链路
#
# 用法：平时由 cam_watchdog.sh 调用，不用手工执行。

source /opt/tros/humble/setup.bash
export CAM_TYPE=usb

DEV="${GC_CAM_DEV:-/dev/cam0}"
# 默认不碰相机曝光/饱和度 —— 现场识别效果是在驱动默认值（brightness=50、
# saturation=60）下调出来的，擅自改会把颜色判据带偏。
# 需要时用环境变量显式指定，例如 GC_CAM_BRIGHTNESS=32 GC_CAM_SATURATION=128
BRIGHT="${GC_CAM_BRIGHTNESS:-}"
SAT="${GC_CAM_SATURATION:-}"
LAUNCH=/root/vision_ws/cam_only.launch.py

echo "[cam_only] $(date '+%H:%M:%S') 启动相机链路 device=$DEV bright=${BRIGHT:-默认} sat=${SAT:-默认}"
ros2 launch "$LAUNCH" device:="$DEV" &
LPID=$!

trap 'kill -INT $LPID 2>/dev/null; wait $LPID 2>/dev/null; exit 0' INT TERM

# 等相机节点把设备参数初始化完（它会重置 ctrl）；只有显式给了才打我们的值
(
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    [ -e "$DEV" ] || continue
    [ -n "$BRIGHT" ] && v4l2-ctl -d "$DEV" --set-ctrl "brightness=$BRIGHT" 2>/dev/null
    [ -n "$SAT" ] && v4l2-ctl -d "$DEV" --set-ctrl "saturation=$SAT" 2>/dev/null
  done
  sleep 2
  echo "[cam_only] 相机参数: $(v4l2-ctl -d "$DEV" --list-ctrls 2>/dev/null | grep -E 'brightness|saturation' | tr '\n' ' ')"
) >> /root/vision_ws/det.log 2>&1 &

while kill -0 "$LPID" 2>/dev/null; do
  sleep 5
  echo "[hb] $(date '+%H:%M:%S') camera-alive dev=$DEV"
done
wait "$LPID"
echo "[cam_only] $(date '+%H:%M:%S') launch 已退出（code=$?）"