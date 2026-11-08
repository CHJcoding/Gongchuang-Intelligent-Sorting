#!/bin/bash
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb

exec ros2 launch /root/vision_ws/camera_web.launch.py \
  device:=${GC_CAM_DEV:-/dev/cam0}
