# 板端代码快照（2026-11-08）

本分支是 RDK X5 板端 `/root/vision_ws` 的**代码快照**，直接从板子导出（IP `192.168.1.117`，`hostname ubuntu`）。

> 说明：板端是当前实际运行的系统（用户在板上直接改），主机 `GC` 仓库的 `VISION_YTM` 分支与板端已经分叉。本分支只做“把板端现状存进 GitHub”，**不改动主机 GC 目录下的任何文件**。

## 目录结构（与板上 /root/vision_ws 一一对应）

| 路径 | 内容 |
|---|---|
| `vision_sorter/` | 生产业务代码（11 个 .py）：`sort_vision_node.py` 主节点、`config.py` 参数、`comm.py` 串口、`grasp_refine.py` 抓取点精修、`camera_model.py` 去畸变、`hand_eye.py` 手眼、`safety.py` 安全闸、`yolov8_bpu.py` 板端推理、`color_classifier.py` 判色、`refine_diag.py`、`test_protocol.py` |
| `tools/` | 标定/抓帧/预览/评估工具（约 70 个 .py + `chessboard_9x6.png`） |
| `*.sh` | 启动与运维脚本：`cam_watchdog.sh`（守护）、`stop_all.sh`（一键停）、`run_sort.sh`/`run_fast.sh`/`run_det.sh`、`restart_vision.sh`、`cam_only.sh` 等 |
| `*.launch.py` | `cam_only.launch.py`、`camera_web.launch.py` |
| `*.json` | `camera_intrinsic.json`（内参）、模型工作配置 `gc_1029_config.json` / `gc_1032b_v2_config.json` / `gc_best_1029_workconfig.json` / `yolov5_best_config.json`、`det.json` |
| `*.list` / `*.txt` | `labels.list`、`gc_1029_labels.list`、`gc_best_1029_labels.list`、`labels.txt`、`启动说明.txt` |
| `协议说明_串口帧.md` | STM32 串口协议规范 |

## 未打包的内容（体积大或非代码，GitHub 不收）

- 模型权重：`*.bin` / `*.onnx` / `*.pt`（`gc_best_1029.bin`、`gc_1032b_v2.bin`、`fp32_best.onnx` 等，共约 120MB）
- 现场截图/证据图：`*.jpg` / `*.png`（除 `tools/chessboard_9x6.png`）
- 日志：`*.log`
- 数据集目录：`dataset*/`、`_capture/`、`_bintest/`、`_fp32/`、`archive/`、`config/`（官方示例 workconfig + 测试图）
- 备份与缓存：`*.bak*`、`__pycache__/`

## 当前架构要点（2026-11-08）

- `run_det.sh` → `cam_only.sh`：只起相机 + 预览（官方 `dnn_node_example` 已摘除，v8 模型它的 C++ parser 读 float32 reg 输出会 SIGSEGV）。
- `run_sort.sh` → `sort_vision_node.py`：业务节点**自己做推理**（`config.USE_LOCAL_DETECTION=True`，走 `yolov8_bpu.py`），判色 → 抓取点精修 → 手眼换算 → 串口下发。
- `run_fast.sh`：台架快速识别模式（扫描 0.3s、抓取节拍 1s）。
