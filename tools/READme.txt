├── tools/                   # PC / 板端调试脚本
│   ├── detect_webcam.py         # 摄像头实时检测
│   ├── preview_cam.py          # 相机纯预览
│   ├── list_cameras.py         # 列出相机编号
│   ├── board_calib_server.py   # ★ 内参标定采集端（跑在板子上）：浏览器打开推流页，
│   │                           #   按提示把棋盘推到画面各处，够了自动解算落盘
│   │                           #   --square 格子边长mm  --min-samples 张数  --port 8080
│   ├── eval_intrinsic.py       # ★ 内参裁判：去畸变后平面棋盘的单应残余（不参与拟合）
│   │                           #   --try-variants 比 4 种模型；--recalib-flags no_tangent
│   │                           #   --recalib-out 写 camera_intrinsic.json
│   ├── camera_intrinsic_calib.py  # 旧的单机标定（本地有图时用；棋盘格 空格拍/c 解算）
│   ├── chessboard_9x6.png       # 棋盘格标定图（A4 打印，100% 不缩放，量一下格子实际边长）
│   ├── planar_calib.py         # ★ 平面/手眼标定：棋盘格照片 -> 像素↔毫米 2x3 矩阵
│   │                           #   --self-test 无需相机；--dry-run 只预览不写盘
│   │                           #   --anchors-file 用采到的锚点自动解旋转+手性
│   │                           #   --probe 实抓偏差补偿；--no-undistort 做内参 A/B 对比
│   │                           #   读内参时先自检，折叠的内参直接拒跑
│   ├── collect_anchors.py      # ★ 采标定锚点：读 STM32 回传坐标，回车记录（--simulate 离线测）
│   ├── gantry_monitor.py       # ★ 读 STM32 回传的 0x07 坐标帧（--simulate 可离线测）
│   ├── hsv_calib.py            # ★ HSV 阈值标定：每种颜色一叠照片 -> 分位数/建议区间/冲突检查
│   │                           #   --apply 只打印 config 片段；--report x.md 出完整报告
│   ├── test_protocol.py        # ★ 离线单测：协议 + 标定数学 + 去畸变 + 内参质量闸 + 安全闸（119 项）
│   ├── validate_postprocess.py # 板端解码结果 vs torch 的对照验证
│   └── test_*.py                # 加载/推理测试脚本
│
├── 标定产物
│   ├── camera_intrinsic.json    # 相机内参（入库；2026-10-26 重标：fx=314.9 fy=317.0，
│   │                           #   no_tangent 模型，重投影 0.257px，valid_radius=0.635）
│   ├── hand_eye.json            # 手眼矩阵（不入库：每台机器不同，planar_calib 生成）
│   └── anchors.json             # 锚点（不入库：collect_anchors 生成，喂给 --anchors-file）

内参标定的完整来回（这次走通的）：
  1) 板子上：python3 tools/board_calib_server.py --square 24
     浏览器开 http://<板子IP>:8080 ，按页面提示把棋盘推到画面各处（四方向各摆几次）
  2) 12 张够了会自动解算；本地再跑一遍裁判确认：
     python tools/eval_intrinsic.py --captures capture --square 24 --try-variants
  3) 选定模型后出正式文件：
     python tools/eval_intrinsic.py --captures capture --square 24 \
            --recalib-flags no_tangent --recalib-out camera_intrinsic.json
  4) 传到板子 /root/vision_ws/ ，重启业务节点看启动那行 [camera]
  ⚠ 内参一变，旧 hand_eye.json 立刻作废，必须重标手眼。

C100相机检测预览
screen -r det      # 看检测节点（Ctrl+A 然后 D 退出，不停止）
screen -r sort     # 看业务节点
bash /root/vision_ws/stop_all.sh    # 出问题先无脑清场，再重起