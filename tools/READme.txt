├── tools/                   # PC / 板端调试脚本
│   ├── detect_webcam.py         # 摄像头实时检测
│   ├── preview_cam.py          # 相机纯预览
│   ├── list_cameras.py         # 列出相机编号
│   ├── camera_intrinsic_calib.py  # 棋盘格内参标定（产出 camera_intrinsic.json，去畸变用）
│   ├── chessboard_9x6.png       # 棋盘格标定图（A4 打印，100% 不缩放）
│   ├── planar_calib.py         # ★ 平面/手眼标定：棋盘格照片 -> 像素↔毫米 2x3 矩阵
│   │                           #   --self-test 无需相机；--dry-run 只预览不写盘
│   │                           #   --anchors-file 用采到的锚点自动解旋转+手性
│   │                           #   --probe 实抓偏差补偿；--no-undistort 做内参 A/B 对比
│   ├── collect_anchors.py      # ★ 采标定锚点：读 STM32 回传坐标，回车记录（--simulate 离线测）
│   ├── gantry_monitor.py       # ★ 读 STM32 回传的 0x07 坐标帧（--simulate 可离线测）
│   ├── hsv_calib.py            # ★ HSV 阈值标定：每种颜色一叠照片 -> 分位数/建议区间/冲突检查
│   │                           #   --apply 只打印 config 片段；--report x.md 出完整报告
│   ├── test_protocol.py        # ★ 离线单测：串口协议 + 标定数学 + 去畸变 + 安全闸（99 项）
│   ├── validate_postprocess.py # 板端解码结果 vs torch 的对照验证
│   └── test_*.py                # 加载/推理测试脚本
│
├── 标定产物
│   ├── camera_intrinsic.json    # 相机内参（入库：C100，重投影 0.289px；换镜头才需要重标）
│   ├── hand_eye.json            # 手眼矩阵（不入库：每台机器不同，planar_calib 生成）
│   └── anchors.json             # 锚点（不入库：collect_anchors 生成，喂给 --anchors-file）

C100相机检测预览
screen -r det      # 看检测节点（Ctrl+A 然后 D 退出，不停止）
screen -r sort     # 看业务节点
bash /root/vision_ws/stop_all.sh    # 出问题先无脑清场，再重起