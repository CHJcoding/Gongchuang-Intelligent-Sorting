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

## 二、板端只需要这 7 个 .py + 4 个 .sh

| 文件 | 作用 |
|---|---|
| `sort_vision_node.py` | 主节点：订阅检测+图像、判色、去畸变、坐标变换、串口下发 |
| `config.py` | 全部参数：HSV 阈值、手眼矩阵、话题名、串口、触发策略、去畸变开关 |
| `color_classifier.py` | 纯 HSV 阈值投票（无 RGB、无中性色分支） |
| `comm.py` | 串口协议：下行 `AA 55 … CRC8 0D`；上行 `0x06`/`0x07` 统一解析 |
| `hand_eye.py` | `pixel_to_robot()`；优先读 `hand_eye.json`（标定结果），读不到用占位值并告警 |
| `safety.py` | 下发前的两道闸：没标定 / 串口掉了 → 拒绝下发（纯逻辑，可离线测） |
| `camera_model.py` | 去畸变（`undistortPoints`）+ 内参自检；料盘边缘修正 ~14px，必须在像素→毫米之前做 |
| `stop_all.sh` | 无脑清场：screen、launch、相机节点、孤儿 websocket、nginx、端口 8000 |
| `cam_watchdog.sh` | 检测节点守护（v2）：温和重启、等相机就绪、退避、自动解析相机设备 |
| `run_det.sh` | 检测 launch 入口（设备号走 `GC_CAM_DEV`，默认 `/dev/cam0`） |
| `run_sort.sh` | 业务节点入口 |

另需一次性安装 `rdk/udev/60-gc-camera.rules`（把 C100 固定成 `/dev/cam0`），见第三节 2.5。

标定产物 `hand_eye.json` **不是代码文件**，由 `tools/planar_calib.py` 生成后
再传到板端（放 `/root/vision_ws/vision_sorter/hand_eye.json` 或 `/root/vision_ws/hand_eye.json` 都会被自动找到）。

内参 `camera_intrinsic.json` 同理：放 `/root/vision_ws/` 或 `vision_sorter/` 都会被自动找到。
**它和 `hand_eye.json` 是一对**——标定时用没用去畸变，推理时就必须一样，节点启动会比对并报错。
**内参一改，旧手眼矩阵立刻作废**（像素坐标系变了），必须重标。

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
# 内参（标定产物，必须和这一版代码配套）
scp camera_intrinsic.json root@<板子IP>:/root/vision_ws/
```

### 2.5 装 udev 规则（**一次性，别跳过**）
```bash
# 让 C100 固定成 /dev/cam0。不做这步，USB 一弹节点号就漂，
# /dev/video0 消失后所有写死它的启动脚本都会失败。
scp rdk/udev/60-gc-camera.rules root@<板子IP>:/etc/udev/rules.d/
ssh root@<板子IP> 'udevadm control --reload-rules && udevadm trigger --subsystem-match=video4linux'
ssh root@<板子IP> 'ls -l /dev/cam0'    # 应指向 videoN（index 0 那个）
```

### 3. 启动顺序（两个终端）

> 板端已放好整套脚本（仓库里 `rdk/vision_sorter/` 下同名文件）：
> `stop_all.sh` 清场、`cam_watchdog.sh` 托管检测节点、`run_det.sh` / `run_sort.sh` 两个入口。
> **推荐直接用守护脚本，别手动起第二份 launch**（一个相机只能被一个进程打开）。
> 四个脚本都已进仓库，直接在本地改、scp 覆盖，不要再在板子上手改（改了会漂）。

```bash
bash /root/vision_ws/stop_all.sh                       # 先清场（含孤儿 websocket）
screen -dmS det /root/vision_ws/cam_watchdog.sh        # 检测节点交给守护托管
screen -r det                                          # 看日志（Ctrl+A 再 D 退出，不停止）
# 等 det.log 出现 30fps 之后，再起业务节点：
screen -dmS sort /root/vision_ws/run_sort.sh
```

**终端 A：官方检测节点**（`run_det.sh` 里就是这段）
```bash
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb
ros2 launch dnn_node_example dnn_node_example.launch.py \
  dnn_example_config_file:=/root/vision_ws/yolov5_best_config.json \
  dnn_example_image_width:=640 dnn_example_image_height:=480 \
  device:=/dev/cam0
```

**终端 B：视觉分拣节点**（`run_sort.sh` 里就是这段）
```bash
source /opt/tros/humble/setup.bash
cd /root/vision_ws/vision_sorter
python3 sort_vision_node.py            # 默认 auto_trigger=true
# 无 STM32 时台架演示（两道安全闸都要显式放开，见第十节「安全闸」）：
#   python3 sort_vision_node.py --ros-args -p serial_sim:=true -p allow_uncalibrated:=true
# 真机触发模式：          python3 sort_vision_node.py --ros-args -p auto_trigger:=false
# 现场实测完节拍直接覆盖：  ... -p grab_cycle_s:=4.5
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
8. **棋盘格的 180° 歧义无法用残差发现**：棋盘对 180° 旋转是对称的，正序/反向两种配对的残差**完全一样**。方向只能靠 `--rot/--mirror` 手动校正，或者给 3 个锚点让它自动解（推荐）。
9. **去畸变开关和手眼标定必须同口径**：`hand_eye.json` 记了标定时有没有去畸变，节点启动会比对，不一致直接 `ERROR`。表现为"稳定地偏十几毫米"，现场极难定位。
10. **`undistortPoints` 只能用在"像素→毫米"之前**：检测框来自带畸变的原图，颜色统计用带畸变 ROI 没关系（判色不看几何），但**坐标计算必须先去畸变**。
11. **去畸变会让画面边缘"收"进来**：这组内参四角修正 79.6px（旧的可笑内参报的是 202px），等于四周内容整体往中心缩。物料贴在画面最边上时，去畸变前后位置差很多，别拿旧截图对像素。
12. **未标定时节点不再"只警告照发"了**：现在直接拒绝下发（`safety.py`），日志会打 `BLOCKED_UNCALIBRATED`，`下发=` 计数不涨。这是刻意的——台架演示要加 `-p allow_uncalibrated:=true`，详见第十节。别把它当 bug 去查。
13. **重投影误差小 ≠ 内参对**：旧内参重投影 0.289px（比新的 0.257px 还漂亮），但它把直线掰弯了（单应残余 2.5px、边缘折叠）。判内参要用「去畸变后平面棋盘的单应残余」，不是 `calibrateCamera` 自己报的残差——那个是拟合残差，模型过拟合时反而更小。
14. **切向项 p1/p2 在手持标定里基本是假的**：纸一弯，切向项就被喂成 p1≈-0.02，修正量变得方向相关，边缘还会不可逆。本项目固定 p1=p2=0（`--recalib-flags no_tangent`），牺牲 0.05px 单应残余换各向同性和全幅可逆。
15. **`cv2.undistortPoints` 在画面极角不收敛**：5 次固定点迭代，归一化半径 >1.1 处误差能到几十 px（牛顿法解同一模型是 0）。所以别拿画面最角上的像素验证内参，也别在那里摆物料。
16. **`/dev/video0` 是独占的，且 USB 掉线后会变成 `/dev/video1`**：第二条 launch 会报 `Cannot open device`；USB 弹一下重枚举后 `/dev/video0` 直接消失（这时 `hobot_usb_cam` 报的是 `Invalid v4l2 format`，极容易误判成相机坏了）。**已根治**：装 `rdk/udev/60-gc-camera.rules` 建固定链接 `/dev/cam0`，`run_det.sh` / `cam_watchdog.sh` 一律用 `/dev/cam0`；链接缺失时守护会自动扫 `/dev/video*` 挑能出 MJPG 的那个兜底。
17. **别在非交互 shell 里跑 ROS2**：`.bashrc` 只在交互 shell 里 source TROS，`bash -c "ros2 ..."` 会报 `ModuleNotFoundError: No module named 'rclpy'`。任何脚本里跑 ROS2 之前必须显式 `source /opt/tros/humble/setup.bash`。
18. **`--square` 一定是量出来的，不是印出来的**：A4 打印会被驱动缩放 96%~100%，本项目实测按 24mm 用；格子边长错 4%，平移量就整体错 4%。

## 八、手眼标定：用棋盘格，不看夹爪

**核心认识**：夹爪/针尖在画面里只有 1~2 像素宽，人眼读数误差就有几毫米；
而且**相机不可能自己知道龙门坐标**——它只看得见像素，坐标真值只能在 STM32 里。
所以：让棋盘格提供「像素↔毫米」的比例和方向（不需要龙门参与），
龙门坐标只用来做最后一步对齐（1 个锚点，或干脆抓一次量偏差）。

### 8.1 前置：相机内参（去畸变）——**已重标，2026-10-26**

内参文件：`camera_intrinsic.json`（C100，640×480）。**推理链路已经接上**
（`camera_model.py` + `sort_vision_node._fire()`），文件在位就自动启用。启动打印：

```
[camera] 去畸变已启用 | 内参=/root/vision_ws/camera_intrinsic.json | fx=314.9 fy=317.0
         cx=320.4 cy=255.4 | 模型=no_tangent | 标定重投影误差=0.257px
         | 可信半径=0.64(画面归一化) | 边缘最大修正=79.6px
  ⚠ 画面四角往返误差 40px：那里归一化半径 >1.1，cv2 迭代不收敛 + 内参外推，别信画面角上的毫米值
```

**旧内参是坏的，已经换掉**。判断依据不是重投影误差（旧的 0.289px 也很漂亮），
而是「去畸变后平面棋盘的单应残余」——正确去畸变后，网格角点必须能被一个 2D 单应
精确解释，不会掰弯：

| 内参 | 单应残余（12 张） | 正逆一致性 | 结论 |
|---|---|---|---|
| 旧（fx=394.6，含切向项） | **2.498 px** | r=1.0 差 **379 px**（模型折叠） | 把直线掰弯了，丢掉 |
| 新 no_tangent（fx=314.9） | **1.315 px** | 全幅 ≤0.22 px | 采用 |
| 干脆不去畸变 | 1.871 px | — | 旧内参比它更差 |

**这组内参的修正量**（用来判断"要不要去畸变"）：

| 归一化半径 r | 旧 vs 新坐标差 | 说明 |
|---|---|---|
| 0.20 | 2.7 px | 正中间，可忽略 |
| 0.40 | 6.6~7.4 px | 料盘内圈，约 1mm |
| 0.50 | 9.1~14.1 px | 料盘边缘，1.5~2mm |
| 0.64 | 14.8~26.2 px | 再往外，3~4mm |

手眼矩阵是 2×3 仿射，吸收不了非线性畸变 → 边缘物料会稳定地偏出去，
这就是当初「边缘有几 mm 系统误差」的来源。

**两条硬约束（别踩）**：

1. **`valid_radius = 0.64`**：样本角点只覆盖到这里（p90=0.555 + 余量）。
   超出部分是外推，节点会打 `目标在去畸变覆盖区之外…毫米值不可靠`，
   统计行 `边缘外推=N`。料盘在工作区内，所以正常；**别把物料摆到画面最边上**。
2. **只有一个径向模型，切向项 p1=p2=0**：手持打印纸不平会把切向项喂成假值
   （拟合出来 p1≈-0.02，真实镜头 <0.005），导致修正量方向相关（实测 5.3 倍各向异性）、
   边缘不可逆。纯径向模型各向同性（r=0.5 处 8 个方向极差 0.05px）、全幅可逆。
   代价只是单应残余多 0.05px，值得。

关于「画面四角往返误差 40px」：这不是内参坏了。`cv2.undistortPoints` 是 5 次固定点
迭代，在归一化半径 >1.1 的极角上本来就不收敛（牛顿法解同一模型误差是 0）。
那里也超出 `valid_radius`，所以只作提示，不会关掉去畸变。

**内参自检（防"折叠模型"）**：节点和 `planar_calib.py` 都会检查
「画面内径向单调 + 工作区正逆一致 ≤0.5px」，不通过就**拒绝启用**（`camera_model`
报 `畸变模型不合理`，`planar_calib` 直接抛错）。因为折叠的模型会让
`undistortPoints` 返回垃圾坐标，用它算毫米/标手眼必然是错的，而且表现是"稳定地偏"。

**重标内参**（换镜头、动过对焦、或者怀疑不准时）：

```bash
# 板子上：起采集页，按提示把棋盘推到画面各处，够了会自动解算
python3 tools/board_calib_server.py --square 24        # 浏览器开 http://<板子IP>:8080
# 本地：判据 + 选模型（no_tangent 是本项目选定的模型）
python tools/eval_intrinsic.py --captures <图目录> --square 24 --try-variants
python tools/eval_intrinsic.py --captures <图目录> --square 24 \
       --recalib-flags no_tangent --recalib-out camera_intrinsic.json
```

`--square` 只影响棋盘位姿的平移量，不影响 fx/fy/cx/cy 和畸变系数；
但**打印出来的格子真实边长要量一下**（这次按 24mm 算，打印版标的是 25mm，差 4%，
对 K、D 无影响，写进 json 的 `square_mm_note` 了）。

> ⚠ **一致性铁律**：`hand_eye.json` 里记了标定时到底有没有去畸变。
> 节点启动会拿它和当前开关比对，不一致直接 `ERROR`。
> **换内参 ≠ 换开关**：内参一变，像素坐标系就变了，**旧的手眼矩阵立刻作废**，必须重标。
> 这条也适用于本次重标——之前如果已经用旧内参标过手眼，那份矩阵不能再用。

### 8.2 拍一张标定图

1. 棋盘格**平铺在物料所在平面**（和物料同高，不是夹爪抬起的高度）
2. 贴死压平，不能有翘曲
3. 拍一张，确认棋盘**完整入镜**（不出画面、不过曝）
4. 存成 `board.jpg`

### 8.3 解算

先 **dry-run** 看检测质量和方向（不写文件，方向没定前别真写）：

```bash
python tools/planar_calib.py --image board.jpg --dry-run
```

确认没问题后正式写盘：

```bash
python tools/planar_calib.py --image board.jpg
```

输出：

- 角点检出数量（应为 9×6=54）
- **格子像素尺寸自检**（≥15px 才稳；<12px 警告，<6px 直接报错让靠近或换大棋盘）
- 抽样角点表：`像素(u,v)` / `棋盘(mm)` / `最终坐标(mm)`
- 重投影残差（达标线：平均 ≤5mm 且 最大 ≤8mm，与是否加 `--rot/--mirror` 无关）
- `hand_eye.json`（2×3 矩阵 + 元数据）

**看抽样表核对方向**：如果「最终坐标」的 X 随棋盘横移反向 → `--mirror x`；
Y 反向 → `--mirror y`；两轴同时反（180°）→ `--rot 180`。

### 8.4 对齐到龙门坐标系（三选一）

| 方式 | 命令 | 需要什么 | 精度 |
|---|---|---|---|
| **3 锚点法**（推荐） | `--anchors-file anchors.json` | STM32 回传坐标 + 夹爪点 3 个角点 | 最好，方向/手性全自动 |
| **锚点法** | `--anchor U V X Y` | 夹爪点住某个角点，读坐标 | 好，但旋转/镜像靠猜 |
| **实抓补偿法**（最省事） | `--probe DX DY` | 抓一次，钢尺量偏差 | ±2mm，要迭代 |

#### 3 锚点法（不用猜 `--rot/--mirror`）

棋盘格本身的摆放角度如果和龙门轴不平行，只修平移会留下系统性旋转误差
（偏 3° → 离锚点 200mm 处就偏 10mm）。给 3 个分散的锚点，
`planar_calib` 会直接解出 **比例 + 旋转 + 平移 + 手性**：

```bash
# ① 采锚点（板子上，串口独占，先停掉 sort_vision_node）
python3 tools/collect_anchors.py --port /dev/ttyUSB0 --corners 0 8 53 \
        --image /root/board.jpg --out /root/vision_ws/anchors.json

# ② 解算（自动读锚点文件）
python3 tools/planar_calib.py --image /root/board.jpg \
        --anchors-file /root/vision_ws/anchors.json
```

采的时候把夹爪尖依次降到该角点**正上方并贴住桌面**，每次按回车；
工具自动读 STM32 的 `0x07` 坐标（多帧取中位数），不用手抄。

角点序号看 `planar_calib` 打印的抽样表。**要选分散的点**（如 0 / 8 / 53）：
三点共线解不出旋转，工具会警告。

输出里会明确告诉你解出了什么：

```
[对齐] 3 个锚点 -> 比例 1.0000 | 不镜像(纯旋转+平移) | 锚点残差 平均0.00mm 最大0.00mm
```

- 比例应 ≈ 1.0，偏离 >3% → `--square` 填错或锚点配错
- 「镜像/不镜像」是自动判定的，**不用再手动 `--mirror`**

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

### 8.6 让 STM32 回传坐标（方案 B，RDK 侧已完成）

RDK 侧**全部已实现并测过**，只等固件发 `0x07` 帧：

| 环节 | 状态 |
|---|---|
| `comm.py` 解析 6 字节坐标帧、与 `0x06` 混流 | ✅ 已完成（`_pump` 统一解析泵） |
| 节点消费坐标 + 发布 `/gantry/pos`（JSON） | ✅ 已完成（`POS_ENABLE=True` 时） |
| `tools/gantry_monitor.py` 实时看坐标 / 存 CSV | ✅ 已完成 |
| `tools/collect_anchors.py` 采锚点（回车记录，多帧中位数） | ✅ 已完成 |
| STM32 固件发 `0x07` | ⬜ 等固件（协议见 `rdk/协议说明_串口帧.md`） |

固件好了之后只需要改两个开关：

```python
POS_ENABLE = True          # config.py：解析并发布坐标
GRAB_DONE_MODE = "byte"    # config.py：用 0x06 做完成判定，不再靠定时器猜
```

验证：

```bash
python3 tools/gantry_monitor.py --port /dev/ttyUSB0     # 板上，实时打印坐标
python tools/gantry_monitor.py --port COM7              # PC 上
```

> 固件没做之前保持 `POS_ENABLE=False`（默认就是 False）。解析器始终兼容，
> 收到 `0x07` 也会存进 `latest_position`，只是不发布、不消费。

### 8.7 自检（不需要相机 / 不需要板子 / 不需要 STM32）

```bash
python tools/planar_calib.py --self-test          # 合成棋盘：检测→求解→对齐→3锚点→JSON
python tools/test_protocol.py                     # 协议 + 标定 + 内参质量闸 + 安全闸，119 项
python tools/gantry_monitor.py --simulate         # 坐标帧解析器抗噪
python tools/collect_anchors.py --simulate        # 锚点采集流程（含坏 CRC/半包）
python tools/hsv_calib.py --data 标定图片          # HSV 阈值标定（见第九节）
```

## 九、颜色阈值标定：别手调，用照片算

现场物料打印出来后，**每种颜色拍一叠照片**，用工具算出建议阈值——手调阈值没有客观
依据，也说不清"到底会不会和别的颜色混"。

### 9.1 拍照要求

```
标定图片/
  green/  1.jpg  2.jpg ...      # 目录名 = config.py 里的英文色名
  red/    1.jpg  2.jpg ...
  blue/   ...
```

* 物料放在画面**正中间**，占画面 1/3 以上（工具只取中心 40% 区域统计）
* 用**最终现场的光照**（补光灯开着），不要用手机闪光灯
* 每种颜色 **≥8 张**，覆盖不同摆放角度
* 格式 jpg/png 都行

### 9.2 出报告

```bash
python tools/hsv_calib.py --data 标定图片 --report hsv_report.md
```

报告里有四块：

| 块 | 内容 | 怎么用 |
|---|---|---|
| 一、H/S/V 分布 | 每种颜色的 H 区间、S/V 的 p1/p50/p99 | 看实测值，别猜 |
| 二、冲突检查 | 哪两种颜色的 H 区间重叠、重叠几度 | 重叠就得靠 S/V 再分，或物料本身太接近 |
| 三、当前阈值表现 | 每种颜色的**召回率**、被别的颜色抢走多少 | 召回 <60% 说明现在根本判不出来 |
| 四、建议阈值 | 可直接粘贴进 `config.py` 的代码 | 复制替换 `COLOR_RANGES_HSV` |

只看建议片段：`python tools/hsv_calib.py --data 标定图片 --apply`

### 9.3 两个已知教训

1. **S 下限不能拍脑袋抬到 70**：现场实测绿色物料的 S 中位数只有 **39**，
   而 `config.py` 现在是 70 → 一上手就全判失败。工具给的建议是"实测 1% 分位数再留余量"。
2. **红色的 H 跨 0°**（H≈178 和 H≈2 是同一个红）：工具用圆统计自动识别，
   建议区间会拆成 `[0, hi]` + `RED_HSV_LOWER2` 两段，别自己拍一段 `[0,10]` 就完事。

## 十、安全闸与故障处理（上线前必读）

### 10.1 两道闸：默认都是"关着"的

`sort_vision_node` 下发改坐标之前会过两道检查（`safety.py`），任何一道不过就**拒绝下发**，
只打 ERROR 日志 + 发状态，不会偷偷把坐标发出去：

| 闸 | 什么时候关 | 原因 |
|---|---|---|
| **标定闸** | `hand_eye.json` 不存在 | 此时坐标来自 `config.HAND_EYE_M` 占位值，是**假的**。龙门装好后一旦忘了标定就会满场乱走 |
| **串口闸** | `/dev/ttyUSB0` 打不开或写失败 | 此时字节根本没发给 STM32，日志里却"看起来在抓" |

启动时会连打两行横幅，一眼看出现在能不能动：

```
安全闸[标定]: 🔒 关闭中（未标定，拒绝下发；要演示请加 -p allow_uncalibrated:=true）
安全闸[串口]: 🔒 故障（打不开，拒绝下发并周期性重试）
```

状态同时发到 `/sort/state`：`BLOCKED_UNCALIBRATED` / `BLOCKED_SERIAL_DOWN` /
`SERIAL_ERROR` / `SERIAL_OK`。统计行里有 `安全闸拦截=` 和 `串口失败=` 两个计数。

**台架演示要显式放开**（两道都要）：

```bash
python3 sort_vision_node.py --ros-args -p serial_sim:=true -p allow_uncalibrated:=true
```

> 也可以改 `config.py` 的 `SAFETY_REQUIRE_CALIB=False` / `ALLOW_UNCALIBRATED=True` /
> `SERIAL_SIM_ON_FAIL=True`，但**上线前务必改回来**。

### 10.2 串口掉了会自愈

USB 转串口会掉（热插拔、供电、驱动）。节点行为：

1. 写失败 → 立刻解除抓取窗口、`serial_ok=False`、状态 `SEND_FAIL`
2. 每 `SERIAL_REOPEN_S`（默认 5s）重试打开一次
3. 打开成功 → 状态 `SERIAL_OK`，继续正常下发

`SERIAL_SIM_ON_FAIL=False`（默认）时**不会**静默退化成模拟模式——这条很重要，
否则现场会出现"日志里抓得很欢、实际一个字节没发"的假象。

### 10.3 标定热加载：不用重启节点

```bash
ros2 service call /sort/reload_calib std_srvs/srv/Trigger
```

重读 `hand_eye.json` 和 `camera_intrinsic.json`，并重新做一致性检查。
标定完直接在终端跑这一句就行，节点和 `screen` 会话都不用动。

### 10.4 节拍参数不用改代码

```bash
python3 sort_vision_node.py --ros-args -p grab_cycle_s:=4.5 -p grab_timeout_s:=10.0
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `grab_cycle_s` | 6.0 | timer 模式下一次抓取循环多长。**实测办法**：放下料 → 手动发一条 `/sort/start` → 秒表记到机械臂完全退出视野，重复 5 次取中位数 +10% 余量 |
| `grab_timeout_s` | 12.0 | byte 模式的兜底超时 |
| `grab_done_mode` | timer | 改 `byte` 就走 STM32 的 `0x06` 完成字节 |

## 十一、还没闭环的事

- [ ] 手眼标定（棋盘格铺平 → `planar_calib.py` → 3 锚点或 `--probe`，验收平均 ≤5mm、最大 ≤8mm）
- [ ] **确认 STM32 固件已按 10 字节（无 z）解析**，CRC 在索引 8、帧尾在索引 9
- [ ] 若采用方案 B：STM32 固件增加 `0x07` 坐标回传帧，然后 `POS_ENABLE=True` + `GRAB_DONE_MODE="byte"`
- [ ] 实机确认去畸变后落点精度（中心 vs 边缘各抓一次，对比偏差）
- [x] 相机去畸变接入 live 管线（`camera_model.py` + `_fire()`，含标定/推理口径一致性检查）
- [x] RDK 侧坐标回传接收（解析 + `/gantry/pos` 话题 + 锚点采集工具）
- [ ] 抓取循环时长标定：实测后调 `grab_cycle_s`（或改 `GRAB_DONE_MODE="byte"`）
- [ ] 串口协议与 STM32 固件对齐（视觉侧 AA55+CRC8 vs 控制侧 `chassis_driver.py` 的 0x6C 帧）
- [ ] 决定是否重建 640×480 的 bin 以消除拉伸失真
- [x] 安全闸：未标定 / 串口掉线一律拒绝下发（`safety.py`，含启动横幅与状态上报）
- [x] 串口掉线自愈（周期性重连，不再静默退化成模拟模式）
- [x] `/sort/reload_calib` 标定热加载服务
- [x] 节拍参数命令行覆盖（`grab_cycle_s` / `grab_timeout_s` / `grab_done_mode`）
- [x] HSV 阈值标定工具（`tools/hsv_calib.py`，照片 → 分位数 → 建议阈值 + 冲突检查）
- [ ] 各颜色实物拍照片跑 `hsv_calib.py`，替换 `COLOR_RANGES_HSV`（等物料打印）