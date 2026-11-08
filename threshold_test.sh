#!/bin/bash
set +u
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb

cd /root/vision_ws
cp -n yolov5_best_config.json yolov5_best_config.025.bak 2>/dev/null

echo "=== 阈值 0.25 -> 0.50 ==="
sed -i 's/"score_threshold": 0.25/"score_threshold": 0.50/' yolov5_best_config.json
grep score_threshold yolov5_best_config.json

pkill -f "dnn_node[_]example" 2>/dev/null
pkill -f "hobot[_]usb[_]cam" 2>/dev/null
sleep 2

nohup ros2 launch dnn_node_example dnn_node_example.launch.py \
  dnn_example_config_file:=/root/vision_ws/yolov5_best_config.json \
  dnn_example_image_width:=640 dnn_example_image_height:=480 \
  device:=/dev/video0 > /tmp/preview05.log 2>&1 &
sleep 15

timeout 60 python3 /root/vision_ws/tools/count_boxes.py 2>&1 | tail -14

echo "=== 帧率 ==="
grep -E "Sub img fps" /tmp/preview05.log | tail -1
echo "=== 节点保持运行 ==="
pgrep -af "dnn_node[_]example" | head -1