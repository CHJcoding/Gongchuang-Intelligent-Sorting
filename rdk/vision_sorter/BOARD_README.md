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
| `comm.py` | 串口协议：下行 `AA 55 … CRC8 0D`；上行 `0x06`/`0x07` 统一解析 |
| `hand_eye.py` | `pixel_to_robot()`；优先读 `hand_eye.json`（标定结果），读不到用占位值并告警 |

标定产物 `hand_eye.json` **不是代码文件**，由 PC 上的 `tools/planar_calib.py` 生成后
再传到板端（放 `/root/vision_ws/vision_sorter/hand_eye.json` 或 `/root/vision_ws/hand_eye.json` 都会被自动找到）。

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

### 下行帧格式（comm.py，10 字节）

| 字节 | 内容 | 说明 |
|---|---|---|
| 0-1 | `AA 55` | 帧头 |
| 2 | shape_id | 0=BQ 1=LK 2=SLZ 3=SMT 4=WLZ 5=YZ 6=YZHU，未知=255 |
| 3 | color_id | 0=red 1=orange 2=yellow 3=green 4=cyan 5=blue 6=purple **7=预留** 8=black 9=white |
| 4-5 | x int16 小端 | 机械臂 X(mm) |
| 6-7 | y int16 小端 | 机械臂 Y(mm) |
| 8 | crc8 | 对字节 0~7，poly 0x07 |
| 9 | `0x0D` | 帧尾 |

示例（圆锥/绿，x=-12，y=407）：`AA 55 06 03 F4 FF 97 01 FE 0D`

> **z 轴已移除**（2026-10-24）：抓取高度由 STM32 自己决定，帧长 11 → 10 字节，CRC 位置 9 → 8。
> 改前后务必确认 STM32 固件已按 10 字节解析，否则会错位。
>
> ⚠️ `color_id` 由 `config.py` 的 **`COLOR_IDS`** 显式固定，**不再**跟随 `COLOR_RANGES_HSV` 的
> 字典顺序。所以增删/重排 HSV 区间**不会**让编号漂移——这是 2026-10-24 删掉 magenta 时
> 特意解耦的（7 号保留不用，避免 black/white 从 8/9 前移）。改编号只改 `COLOR_IDS`。

### 上行帧格式（RDK 侧统一解析，三个包可以混流）

STM32 → RDK 的所有上行包走**同一条串口**，由 `comm.py` 的 `_pump()` 统一拆分。
三个 `poll_*` 可以任意顺序、任意频率调用，不会互相吃掉字节。

| 包 | 字节 | 用途 | 启用开关 |
|---|---|---|---|
| 抓取完成 | `0x06` | 「抓完了」动作级信号 | `GRAB_DONE_MODE="byte"` |
| 完成+结果 | `0x06` + `status` | 抓成功/失败/急停/其它 | `GRAB_STATUS_ENABLE=True`（当前 False） |
| **夹爪坐标** | `0x07` + x int16 + y int16 + crc8 | 回传当前坐标（手眼标定取真值） | `POS_ENABLE=True`（当前 False） |

坐标帧细节：

| 字节 | 内容 | 说明 |
|---|---|---|
| 0 | `0x07` | 帧头 |
| 1-2 | x int16 小端 | 当前 X(mm)，有符号 |
| 3-4 | y int16 小端 | 当前 Y(mm)，有符号 |
| 5 | crc8 | 对字节 0~4，poly 0x07 / init 0x00（与下行同一算法） |

参考实现：`comm.build_pos_frame(x, y)`（本地可离线生成测试帧），
PC 侧实时查看：`python tools/gantry_monitor.py --port COMx --csv trace.csv`。

> **坐标必须是和下行帧同一个坐标系**，否则标定结果会整体平移。
> 固件没做好之前 `POS_ENABLE` 必须保持 `False`，否则收不到 0x07 会让标定工具超时。

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
5. ~~**串口 color_id 依赖 `COLOR_RANGES_HSV` 的 dict 插入顺序**~~ → **已修**：编号改由 `COLOR_IDS` 显式固定，增删/重排 HSV 区间不会再让 STM32 侧编号漂移。
6. **`HAND_EYE_M` 仍是占位值**：`hand_eye.json` 不存在时节点启动会打 `ERROR` 并反复告警，此时坐标是假的。别拿假坐标驱动龙门实动。
7. **棋盘格不铺平 = 标定白做**：A4 纸翘 2mm，坐标就偏几毫米。必须贴死或用亚克力压平，且**铺在物料所在的那一层高度**上。
8. **棋盘格的 180° 歧义无法用残差发现**：棋盘对 180° 旋转是对称的，正序/反向两种配对的残差**完全一样**。方向只能靠 `--rot/--mirror` 校正，平移靠实抓 `--probe` 校正。

## 八、手眼标定：用棋盘格，不看夹爪

**核心认识**：夹爪/针尖在画面里只有 1~2 像素宽，人眼读数误差就有几毫米；
而且**相机不可能自己知道龙门坐标**——它只看得见像素，坐标真值只能在 STM32 里。
所以：让棋盘格提供「像素↔毫米」的比例和方向（不需要龙门参与），
龙门坐标只用来做最后一步对齐（1 个锚点，或干脆抓一次量偏差）。

### 8.1 前置：先做相机内参（去畸变）

```bash
# PC 上，打印 tools/chessboard_9x6.png（A4，100%，不许缩放）
python tools/camera_intrinsic_calib.py      # 空格拍 15 张 → c 解算
# 产出 camera_intrinsic.json（640×480 的 mtx + dist）
```

内参不能跳：畸变在**画面边缘**能造成几 mm~十几 mm 误差，而物料很可能正好在边缘。

### 8.2 拍一张标定图

1. 棋盘格**平铺在物料所在平面**（和物料同高，不是夹爪抬起的高度）
2. 贴死压平，不能有翘曲
3. 拍一张，确认棋盘**完整入镜**（不出画面、不过曝）
4. 存成 `board.jpg`

### 8.3 解算

```bash
python tools/planar_calib.py --image board.jpg
```

输出：

- 角点检出数量（应为 9×6=54）
- 抽样角点表：`像素(u,v)` / `棋盘(mm)` / `最终坐标(mm)`
- 重投影残差（达标线：平均 ≤5mm 且 最大 ≤8mm）
- `hand_eye.json`（2×3 矩阵 + 元数据）

**看抽样表核对方向**：如果「最终坐标」的 X 随棋盘横移反向 → `--mirror x`；
Y 反向 → `--mirror y`；两轴同时反（180°）→ `--rot 180`。

### 8.4 对齐到龙门坐标系（二选一）

| 方式 | 命令 | 需要什么 |
|---|---|---|
| **锚点法** | `--anchor U V X Y` | 夹爪针尖点住某个角点，读 STM32 显示的坐标 |
| **实抓补偿法**（最省事） | `--probe DX DY` | 抓一次物料，钢尺量出偏了多少 |

`--probe` 直接改矩阵第 3 列（平移项），不动比例和方向，通常 2 次迭代收敛到 ±2mm：

```bash
python tools/planar_calib.py --probe 12 -8    # 抓出来 X 偏 +12、Y 偏 -8
```

### 8.5 上板

```bash
scp hand_eye.json root@10.161.15.50:/root/vision_ws/vision_sorter/
```

节点启动时会打印：

```
[hand_eye] 已标定 | 来源=/root/vision_ws/vision_sorter/hand_eye.json | 残差 平均1.20mm 最大2.40mm | 点数=54
```

没标定则是 `ERROR` 告警 + 每次下发前 `WARN`，防止拿假坐标动龙门。

### 8.6 可选：让 STM32 回传坐标（方案 B）

标定时不想手抄示教器坐标，就让固件周期性发 `0x07` 坐标帧（格式见上文），
然后：

```python
POS_ENABLE = True          # config.py
```

- RDK 侧：`/gantry/pos` 话题（JSON），10Hz
- PC 侧：`python tools/gantry_monitor.py --port COM5 --csv trace.csv`
- 帧解析要和 `0x06` 完成字节混流，`comm.py` 的统一解析泵已经处理好，不需要额外改动

> 固件没做之前保持 `POS_ENABLE=False`。

### 8.7 自检（不需要相机/不需要板子）

```bash
python tools/planar_calib.py --self-test          # 合成棋盘图验证整条链路
python tools/test_protocol.py                    # 协议 + 标定数学，59 项
python tools/gantry_monitor.py --simulate        # 坐标帧解析器抗噪
```

## 九、还没闭环的事

- [ ] 手眼标定（棋盘格铺平 → `planar_calib.py` → `--probe` 补偿，验收平均 ≤5mm、最大 ≤8mm）
- [ ] **确认 STM32 固件已按 10 字节（无 z）解析**，CRC 在索引 8、帧尾在索引 9
- [ ] 若采用方案 B：STM32 固件增加 `0x07` 坐标回传帧，然后 `POS_ENABLE=True`
- [ ] 相机去畸变接入 live 管线（`camera_intrinsic.json` 已有，640×480，但节点未做 `undistort`）
- [ ] 抓取循环时长标定：实测后把 `GRAB_CYCLE_S` 调准（或改 `GRAB_DONE_MODE="byte"` 由 STM32 回完成字节）
- [ ] 串口协议与 STM32 固件对齐（视觉侧 AA55+CRC8 vs 控制侧 `chassis_driver.py` 的 0x6C 帧）
- [ ] 决定是否重建 640×480 的 bin 以消除拉伸失真