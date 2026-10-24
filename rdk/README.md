# rdk/ —— RDK X5 板端部署根目录

本目录放**所有要跑在 RDK X5 板子上的代码和文档**，与板端 `/root/vision_ws/` 结构对应。

```
rdk/
├── README.md                  ← 本文件
├── chassis_driver.py          ← 控制侧节点（订阅 /cmd_vel_muxed、/scanner/parsed、
│                                 /vision/result、/servo/target、/extend/target，
│                                 打包下发 STM32；pack_servo/pack_extend 待填）
├── vision_sorter/             ← 视觉侧（D1 架构：订阅官方 dnn_node_example）
│   ├── BOARD_README.md        ← ★ 板端部署 + 启动步骤 + 已知坑，先看这个
│   ├── sort_vision_node.py    ← 主节点
│   ├── config.py              ← HSV 阈值 / 手眼矩阵 / 话题 / 串口 / 触发
│   ├── color_classifier.py    ← 纯 HSV 阈值投票
│   ├── comm.py                ← 串口 AA55+CRC8
│   ├── hand_eye.py            ← 像素→机械臂坐标（含 PC 端标定工具）
│   └── (camera/detector/state_machine/color_tuner/yolo_postprocess)  ← 旧架构，不上板
├── bin模型导入RDK.txt          ← 文档
├── RDKX5_N10.txt
├── 下位机数据接收.txt
└── 模型转换全流程（YOLOv8 → RDK X5）.txt
```

## 部署对应关系

| 本地 | 板端 |
|---|---|
| `rdk/vision_sorter/*.py` | `/root/vision_ws/vision_sorter/` |
| `rdk/chassis_driver.py` | 未上传（用户待定） |
| — | `/root/vision_ws/yolov5_best.bin`（模型，不入 git） |
| — | `/root/vision_ws/yolov5_best_config.json`（官方检测节点配置，**只在板上**） |

## 两个节点、两根串口

视觉侧 `comm.py` 用 **`AA 55 … CRC8 0D`** 把「形状+颜色+坐标」发给 STM32；
控制侧 `chassis_driver.py` 用 **`0x6C … 0x2F`** 发运动/舵机/伸缩。

两套帧格式**尚未统一**，等 STM32 固件确认后收敛。见 `vision_sorter/BOARD_README.md` §七。