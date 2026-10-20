# RDK X5 板端部署说明

## 环境依赖（RDK X5 Ubuntu 已自带大部分）
```bash
pip3 install pyserial numpy opencv-python
# hobot_dnn 系统自带，不用装
```

## 文件清单（已按板端改好）
- `config.py` — 所有参数集中
- `camera.py` — C100 USB相机多线程采集（V4L2，已关AE/AWB）
- `color_tuner.py` — 现场调HSV阈值
- `color_classifier.py` — HSV颜色投票+反光剔除+深红暗橙判别
- `detector.py` — YOLOv5n BPU推理（pyeasy_dnn）
- `hand_eye.py` — 像素→机械臂坐标+标定
- `comm.py` — 串口协议（CRC8+ACK重试）
- `state_machine.py` — 主状态机

## 上板步骤

### 1. 模型转换（PC或开发机上做）
```
best.pt -> best.onnx (用ultralytics export)
best.onnx -> best.bin (用地平线 hb_model_transform)
把 .bin 放到 RDK 的 vision_sorter/weights/ 目录
```

### 2. 传文件到RDK X5
```bash
scp -r vision_sorter/ ubuntu@<RDK_IP>:~/
```

### 3. 板上跑顺序
```bash
cd ~/vision_sorter
ls /dev/video*           # 确认相机节点
python3 camera.py        # 先看画面
python3 color_tuner.py  # 现场补光下调10种颜色HSV，填回config.py
python3 hand_eye.py     # 跑标定，填HAND_EYE_M
python3 detector.py      # 测试BPU推理，看检测框
python3 state_machine.py  # 全流程
```

## 已针对你的硬件做的适配
- 相机：`CAP_V4L2` + MJPG，关掉自动曝光/白平衡（手动补光必做）
- 模型：`hobot_dnn.pyeasy_dnn` 加载.bin，BPU硬解
- 串口：`/dev/ttyUSB0`，timeout用秒
- 反光：自动剔除V>230高光和S<40灰像素
- 深红/暗橙：R/(R+G)比值二次判别
