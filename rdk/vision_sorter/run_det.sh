#!/bin/bash
# 检测 launch：设备号由环境变量 GC_CAM_DEV 给（守护脚本会导出），默认 /dev/cam0。
# 不要写死 /dev/video0 —— USB 重枚举后节点号会漂（实测变成 /dev/video1），
# 写死的 launch 会永远报 "Cannot open device /dev/video0" / "Invalid v4l2 format"。
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb
exec ros2 launch dnn_node_example dnn_node_example.launch.py \
    dnn_example_config_file:=/root/vision_ws/yolov5_best_config.json \
    dnn_example_image_width:=640 \
    dnn_example_image_height:=480 \
    device:=${GC_CAM_DEV:-/dev/cam0}