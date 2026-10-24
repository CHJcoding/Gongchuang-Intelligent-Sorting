# RDK X5 板端部署说明（D1 架构：订阅官方检测节点）

> 现状基准：2026-10-21 板端实测（10.132.154.122）+ 2026-10-24 目录整理。
> 本地源码位置：`GC/rdk/vision_sorter/`
> 板端部署位置：`/root/vision_ws/vision_sorter/`

## 一、架构：检测归官方，颜色/坐标/串口归我们

```
C100 USB相机
   │
   ├── /hbmem_img (NV12 零拷贝) ──► dnn_node_example (BPU, yolov5_best.bin 672)
   │                                      │
   │                                      └── /hobot_dnn_detection (ai_msgs)
   │                                                  │
   └── /image (CompressedImage jpeg) ─────────────┐   │
                                                  ▼   ▼
                                        sort_vision_node.py
                                         ① 形状/位置（来自检测）
                                         ② 颜色（同帧 ROI 做纯 HSV 投票）
                                         ③ 像素→机械臂毫米（HAND_EYE_M）
                                          ├── /vision/result (JSON)
                                          └── 串口 AA55+CRC8 ──► STM32
```

**关键点**：检测和图像是两条独立话题，必须按时间戳配对同一帧（节点内 20 帧缓存 + 取最近 stamp）。

## 二、板端只需要这 5 个文件

| 文件 | 作用 |
|---|---|
| `sort_vision_node.py` | 主节点：订阅检测+图像、判色、坐标变换、串口下发 |
| `config.py` | 全部参数：HSV 阈值、手眼矩阵、话题名、串口、触发策略 |
| `color_classifier.py` | 纯 HSV 阈值投票（无 RGB、无中性色分支） |
| `comm.py` | 串口协议：`AA 55 … CRC8 0D`，ACK 0x06 |
| `hand_eye.py` | `pixel_to_robot()`；`calibrate()` 是 PC 端标定工具 |

以下模块**是旧的「自己跑 BPU」架构，新节点不 import，不必上板**：
`camera.py`、`detector.py`、`state_machine.py`、`color_tuner.py`、`yolo_postprocess.py`

## 三、上板步骤

### 1. 依赖
```bash
pip3 install pyserial numpy opencv-python   # hobot_dnn 系统自带
```

### 2. 传文件
```bash
# 本地 GC 根目录执行
scp -r rdk/vision_sorter/ root@<板子IP>:/root/vision_ws/
```
板端已存在 `/root/vision_ws/yolov5_best.bin`、`yolov5_best_config.json`、`labels.list` 时为增量覆盖。

### 3. 启动顺序（两个终端）

**终端 A：官方检测节点**
```bash
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb
ros2 launch dnn_node_example dnn_node_example.launch.py \
  dnn_example_config_file:=/root/vision_ws/yolov5_best_config.json \
  dnn_example_image_width:=640 dnn_example_image_height:=480 \
  device:=/dev/video0
```

**终端 B：视觉分拣节点**
```bash
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter
python3 sort_vision_node.py            # 默认 auto_trigger=true
# 无 STM32 时加串口模拟： python3 sort_vision_node.py --ros-args -p serial_sim:=true
# 真机触发模式：          python3 sort_vision_node.py --ros-args -p auto_trigger:=false
```

预览：`http://<板子IP>:8000`（websocket 在 8080，nginx 代理到 8000）

## 四、两个「config」别搞混

| 文件 | 管什么 |
|---|---|
| `/root/vision_ws/yolov5_best_config.json` | **官方检测节点**配置：模型路径、anchors、`score_threshold=0.80`、7 类 |
| `/root/vision_ws/vision_sorter/config.py` | **我们的节点**配置：HSV 阈值、手眼矩阵、话题、串口、触发 |

调**颜色**改后者；调**检测置信度**改前者。

## 五、板端实测数据（2026-10-21）

| 指标 | 值 |
|---|---|
| 相机 | 640×480 MJPG（C100） |
| 模型输入 | (1,3,672,672) NV12 |
| 模型输出 | 3 个尺度 NHWC F32：`(1,84,84,36)` / `(1,42,42,36)` / `(1,21,21,36)` |
| 推理耗时 | pre 3ms + infer 10ms + post 11~14ms |
| 帧率 | 相机输入 30~60fps，Smart fps 约 37 |
| 检测框坐标空间 | **原图 640×480**，与本地 torch 误差 1~3px，无需缩放 |
| 稳定性 | 阈值 0.80 时每帧 1 个框，连续 60 帧零空帧 |

## 六、已知坑（都已踩过，别再踩）

1. **`ai_msgs/Target` 字段是 `rois`，不是 `tracks`** —— 用 `tracks` 会在回调里抛 `AttributeError`。
2. **`score_threshold` 别低于 0.50**：0.25 时一帧能吐 45 个重叠误检，下游完全没法用。
3. **官方前处理是硬件 VPS 直接拉伸，没有 letterbox**：640×480 → 672×672 纵向拉 1.4 倍，形状失真。根治方案是把 bin 重建为 640×480（待定）。
4. **`/image` 是 `sensor_msgs/CompressedImage`**（jpeg），不是 `sensor_msgs/Image`。
5. **串口 color_id 依赖 `COLOR_RANGES_HSV` 的 dict 插入顺序**：调整阈值可以，**增删 key 或换顺序会让编号错位**。
6. **`HAND_EYE_M` 目前是占位值**，坐标换算结果是假的，标定前别当真。

## 七、还没闭环的事

- [ ] 手眼标定（机械臂到货后按 `hand_eye.py` 采 9 点，验收平均 ≤5mm、最大 ≤8mm）
- [ ] 相机去畸变接入 live 管线（`camera_intrinsic.json` 已有，640×480，但节点未做 `undistort`）
- [ ] 串口协议与 STM32 固件对齐（视觉侧 AA55+CRC8 vs 控制侧 `chassis_driver.py` 的 0x6C 帧）
- [ ] 决定是否重建 640×480 的 bin 以消除拉伸失真