# -*- coding: utf-8 -*-
"""从 ROS 话题抓一帧存成文件（**不占相机设备**，检测节点在跑也能用）。

为什么需要它：`/dev/video0`（= `/dev/cam0`）是独占的，只要 `hobot_usb_cam`
在跑，第二个进程 `cv2.VideoCapture` 一律 `open failed`。而标定/看视野本来就
该用「和推理同一帧」，直接从 `/image` 话题拿原始 JPEG 字节更好：

  * 不重新编码 —— 落盘字节 == 推理输入字节，像素坐标系完全一致
  * 不抢设备 —— 检测链路照常跑

用法:
    python3 tools/snap_image.py                        # -> /root/vision_ws/board.jpg
    python3 tools/snap_image.py --out /root/vision_ws/x.jpg
    python3 tools/snap_image.py --topic /image_annotated   # 带框的图（看识别用）
    python3 tools/snap_image.py --check                # 顺手验一下棋盘格能不能检出

退出码: 0 成功 / 2 超时没收到帧
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class Snapper(Node):
    def __init__(self, topic, out, skip, check):
        super().__init__("snap_image")
        self.out = out
        self.skip = skip              # 丢掉前 N 帧再存（等自动曝光/白平衡稳定）
        self.check = check
        self.seen = 0
        self.saved = None             # (bytes, w, h, encoding)
        self.create_subscription(CompressedImage, topic, self.cb,
                                 qos_profile_sensor_data)

    def cb(self, msg):
        self.seen += 1
        if self.seen <= self.skip:
            return
        raw = bytes(msg.data)
        if not raw:
            return
        head = raw[:2]
        if head[:1] != b"\xff" and head[:2] != b"\xff\xd8":
            self.get_logger().warn("帧不是 JPEG 开头（前两字节 %r），仍尝试落盘"
                                   % head)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn("解码失败，跳过")
            return
        with open(self.out, "wb") as fh:
            fh.write(raw)
        h, w = img.shape[:2]
        self.saved = (len(raw), w, h, getattr(msg, "format", "?"))
        self.grey_mean = float(img.mean())
        self.img = img


def check_chessboard(img, cols=9, rows=6):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(grey, (cols, rows), flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
    return ok, corners


def main(argv=None):
    ap = argparse.ArgumentParser(description="从 ROS 话题抓一帧存文件（不占相机）")
    ap.add_argument("--out", default="/root/vision_ws/board.jpg")
    ap.add_argument("--topic", default="/image",
                    help="默认 /image（原始）；/image_annotated 是带框的")
    ap.add_argument("--skip", type=int, default=2, help="丢掉前 N 帧再存（默认 2）")
    ap.add_argument("--timeout", type=float, default=15.0, help="等帧超时秒数")
    ap.add_argument("--check", action="store_true", help="存完验一下棋盘格 9x6")
    args = ap.parse_args(argv)

    rclpy.init()
    node = Snapper(args.topic, args.out, args.skip, args.check)
    t0 = time.time()
    try:
        while rclpy.ok() and node.saved is None and time.time() - t0 < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.saved is None:
        print("[snap] %0.1fs 内没收到 %s 的帧（见到 %d 帧就不够存）"
              % (args.timeout, args.topic, node.seen))
        print("[snap] 检查：ros2 topic list | grep image；"
              "screen -r det 看检测节点是否在发图")
        return 2

    nbytes, w, h, fmt = node.saved
    print("[snap] 已存 %s" % args.out)
    print("[snap] %dx%d  %s  %.1f KB  灰度均值 %.1f"
          % (w, h, fmt, nbytes / 1024.0, node.grey_mean))
    if node.grey_mean < 20:
        print("[snap] ⚠ 画面几乎全黑，检查镜头盖/曝光")
    elif node.grey_mean > 230:
        print("[snap] ⚠ 画面几乎全白，检查过曝")

    if args.check:
        ok, corners = check_chessboard(node.img)
        if ok:
            print("[snap] 棋盘格 9x6：检出 %d 个角点 ✔" % len(corners))
            idx = {0: "左上#0", 8: "右上#8", 45: "左下#45", 53: "右下#53",
                   10: "内#10", 16: "内#16", 37: "内#37", 43: "内#43"}
            for i, name in sorted(idx.items()):
                if i < len(corners):
                    u, v = corners[i][0]
                    print("      %-8s (%7.1f, %6.1f)" % (name, u, v))
        else:
            print("[snap] 棋盘格 9x6：没检出（图里有棋盘吗？整幅都入镜了吗？）")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())