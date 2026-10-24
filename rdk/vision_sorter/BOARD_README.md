# RDK X5 板端部署说明（D1 架构：订阅官方检测节点）

> 现状基准：2026-10-21 板端实测 + 2026-10-24 目录整理 / 抓取流控改造。
> 板子 IP：**10.161.15.50**（固定，历史曾用 10.132.154.122）
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

## 五、抓取流控：一发一抓，RDK 怎么知道「抓完了」

**问题**：串口是单向的（RDK→STM32），帧发出去 RDK 并不知道机械臂抓完没有。更麻烦的是——**抓取时机械臂会进入相机视野，YOLO 会把夹具/手臂误检成物料**，所以抓取期间必须屏蔽检测，否则会立刻发出一帧错误坐标。

所以「抓取窗口」是硬需求，不只是为了节拍。窗口何时结束，两种模式二选一（`config.py: GRAB_DONE_MODE`）：

| 模式 | STM32 要做什么 | RDK 怎么解锁 | 特点 |
|---|---|---|---|
| `"timer"`（默认） | **什么都不用做** | 发帧后等 `GRAB_CYCLE_S`（默认 6s）定时解锁 | 最省事，时间是估的 |
| `"byte"`（推荐） | 抓取动作**结束时**发 1 字节 `0x06` | 收到完成字节立即解锁，`GRAB_TIMEOUT_S` 兜底 | 自适应、节拍最快 |

**`0x06` 的两种语义别混**：
- **帧级 ACK** = 收到帧就回 → "我收到这条指令了"（需要重试握手）→ 你说不要的是这个
- **动作级 DONE** = 抓取做完才回 → "我抓完了"（不需要重试）→ 只需一行 `Serial_SendByte(0x06)`

```
发出帧 → busy=True，进入屏蔽期（忽略所有检测）
   ├─ byte 模式：收到 0x06 → 解锁（最快）
   ├─ timer 模式：到 GRAB_CYCLE_S → 解锁
   └─ 超时 GRAB_TIMEOUT_S → 强制解锁 + 告警（防永卡）
解锁后再屏蔽 BLANK_AFTER_DONE_S(0.8s)，等机械臂收回、离开视野
```

相关参数（全在 `config.py`）：
```python
GRAB_DONE_MODE   = "timer"  # "timer" | "byte"
GRAB_CYCLE_S     = 6.0      # timer：一次完整抓取循环时长（实测后调准）
GRAB_DONE_BYTE   = 0x06     # byte：完成字节
GRAB_TIMEOUT_S   = 12.0     # 兜底强制解锁
BLANK_AFTER_DONE_S = 0.8    # 抓完后的收臂屏蔽
GRAB_BLANK_DETECT  = True   # 抓取期间屏蔽检测
SEND_WAIT_ACK      = False  # 不等帧级 ACK
```

**怎么标定 `GRAB_CYCLE_S`**：手动触发一次抓取，掐表从「帧发出」到「机械臂回到待命位」，取实测值 + 20% 余量。

### 可选：2 字节协议（代码已实现，但**暂未启用**）

1 字节只表示「抓完了」；2 字节还能带回「抓成功没有」：

```
STM32 → RDK:   0x06  +  status
                       0x00 = 成功
                       0x01 = 空抓/失败
                       0x02 = 急停
                       0x03 = 其它错误
```

前提是机械臂有**限位开关 / 夹持检测 / 真空压力开关**之类，否则压根拿不到成功与否。

启用方法（`config.py`）：

```python
GRAB_STATUS_ENABLE = True      # 现在是 False（注释着不启用）
GRAB_DONE_MODE     = "byte"
```

启用后 RDK 会打印 `抓取结束[DONE] 用时 4.3s 下位机回报=成功`，并把 `/sort/state` 发成
`GRAB_OK` / `GRAB_FAIL`，方便上层做「失败重试」或「报警」。启用前请确认 STM32 固件
**真的会发 2 字节**，否则会一直等到 `GRAB_TIMEOUT_S` 才解锁。

### 下行帧格式（comm.py，11 字节）

| 字节 | 内容 | 说明 |
|---|---|---|
| 0-1 | `AA 55` | 帧头 |
| 2 | shape_id | 0=BQ 1=LK 2=SLZ 3=SMT 4=WLZ 5=YZ 6=YZHU，未知=255 |
| 3 | color_id | 0=red 1=orange 2=yellow 3=green 4=cyan 5=blue 6=purple 7=magenta 8=black 9=white |
| 4-5 | x int16 小端 | 机械臂 X(mm) |
| 6-7 | y int16 小端 | 机械臂 Y(mm) |
| 8 | z_byte | 抓取高度档 0/1/2 |
| 9 | crc8 | 对字节 0~8，poly 0x07 |
| 10 | `0x0D` | 帧尾 |

> ⚠️ `color_id` 由 `COLOR_RANGES_HSV` 的 **dict 插入顺序**生成。调阈值可以，**增删 key 或换顺序会让编号错位**。

## 六、板端实测数据（2026-10-21）

| 指标 | 值 |
|---|---|
| 相机 | 640×480 MJPG（C100） |
| 模型输入 | (1,3,672,672) NV12 |
| 模型输出 | 3 个尺度 NHWC F32：`(1,84,84,36)` / `(1,42,42,36)` / `(1,21,21,36)` |
| 推理耗时 | pre 3ms + infer 10ms + post 11~14ms |
| 帧率 | 相机输入 30~60fps，Smart fps 约 37 |
| 检测框坐标空间 | **原图 640×480**，与本地 torch 误差 1~3px，无需缩放 |
| 稳定性 | 阈值 0.80 时每帧 1 个框，连续 60 帧零空帧 |

## 七、已知坑（都已踩过，别再踩）

1. **`ai_msgs/Target` 字段是 `rois`，不是 `tracks`** —— 用 `tracks` 会在回调里抛 `AttributeError`。
2. **`score_threshold` 别低于 0.50**：0.25 时一帧能吐 45 个重叠误检，下游完全没法用。
3. **官方前处理是硬件 VPS 直接拉伸，没有 letterbox**：640×480 → 672×672 纵向拉 1.4 倍，形状失真。根治方案是把 bin 重建为 640×480（待定）。
4. **`/image` 是 `sensor_msgs/CompressedImage`**（jpeg），不是 `sensor_msgs/Image`。
5. **串口 color_id 依赖 `COLOR_RANGES_HSV` 的 dict 插入顺序**：调整阈值可以，**增删 key 或换顺序会让编号错位**。
6. **`HAND_EYE_M` 目前是占位值**，坐标换算结果是假的，标定前别当真。

## 八、还没闭环的事

- [ ] 手眼标定（机械臂到货后按 `hand_eye.py` 采 9 点，验收平均 ≤5mm、最大 ≤8mm）
- [ ] 相机去畸变接入 live 管线（`camera_intrinsic.json` 已有，640×480，但节点未做 `undistort`）
- [ ] 抓取循环时长标定：实测后把 `GRAB_CYCLE_S` 调准（或改 `GRAB_DONE_MODE="byte"` 由 STM32 回完成字节）
- [ ] 串口协议与 STM32 固件对齐（视觉侧 AA55+CRC8 vs 控制侧 `chassis_driver.py` 的 0x6C 帧）
- [ ] 决定是否重建 640×480 的 bin 以消除拉伸失真