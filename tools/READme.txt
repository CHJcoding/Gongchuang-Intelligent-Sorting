├── tools/                   # PC 调试脚本
│   ├── detect_webcam.py         # 摄像头实时检测
│   ├── preview_cam.py          # 相机纯预览
│   ├── list_cameras.py         # 列出相机编号
│   ├── camera_intrinsic_calib.py  # 棋盘格内参标定（去畸变用）
│   ├── chessboard_9x6.png       # 棋盘格标定图（A4 打印，100% 不缩放）
│   ├── planar_calib.py         # ★ 平面/手眼标定：棋盘格照片 -> 像素↔毫米 2x3 矩阵
│   │                           #   --self-test 无需相机；--probe 实抓偏差补偿
│   ├── gantry_monitor.py       # ★ 读 STM32 回传的 0x07 坐标帧（--simulate 可离线测）
│   ├── test_protocol.py        # ★ 离线单测：串口协议 + 标定数学（59 项）
│   ├── validate_postprocess.py # 板端解码结果 vs torch 的对照验证
│   └── test_*.py                # 加载/推理测试脚本