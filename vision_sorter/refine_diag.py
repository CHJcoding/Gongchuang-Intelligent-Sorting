#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refine_diag.py —— 抓取点精修的现场诊断工具（板端跑，只读话题）

为什么需要它：抓取点精修要判断"颜色掩膜有没有资格算质心"。现场日志里只有
一个「覆盖率/实心」数字，一旦某颗料被判回退，光看数字分不清是掩膜糊到背景、
还是只覆盖了半个物体（反光/阴影/颜色区间偏了）。本工具把两个候选掩膜的
每一步都摊开，并把裁图和掩膜存成 PNG，肉眼一看就知道该调哪个阈值。

输出（每颗料一段）：
  * 框内 HSV 中位数 + S/V 分位数（判断颜色区间是否把物体切掉了一半）
  * 粗掩膜的 fill/贴合框/ratio/solidity/coverage 及被拒原因
  * 自适应掩膜的同样几项及被拒原因
  * 生产判定结果（ok / 抓取点 / 回退原因）
  * 落盘：_crop.png / _mask_coarse.png / _mask_adapt.png

用法：
  source /opt/tros/humble/setup.bash
  python3 refine_diag.py --rounds 8 --min-gap 1.5 --out /root/vision_ws/diag
取图：scp -r root@板:/root/vision_ws/diag .
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/vision_ws/diag")
    ap.add_argument("--rounds", type=int, default=8, help="采样多少轮检测消息")
    ap.add_argument("--min-gap", type=float, default=1.5, help="两轮最小间隔(s)")
    ap.add_argument("--max-boxes", type=int, default=3, help="每轮最多诊断几个框")
    ap.add_argument("--zoom", type=int, default=4, help="存图放大倍数")
    ap.add_argument("--image-topic", default="/image")
    ap.add_argument("--det-topic", default="/hobot_dnn_detection")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as C                                   # noqa: F401
    import grasp_refine as GR
    from color_classifier import classify_color

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from ai_msgs.msg import PerceptionTargets

    def px(m):
        return int(np.count_nonzero(m)) if m is not None else 0

    def hsv_stats(hsv_crop, mask):
        p = hsv_crop[mask > 0]
        if p.shape[0] < 5:
            return None
        h, s, v = p[:, 0].astype(float), p[:, 1].astype(float), p[:, 2].astype(float)
        return (float(np.median(h)), float(np.median(s)), float(np.median(v)),
                float(np.percentile(s, 10)), float(np.percentile(s, 90)),
                float(np.percentile(v, 10)), float(np.percentile(v, 90)))

    def zoom(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, None, fx=args.zoom, fy=args.zoom,
                          interpolation=cv2.INTER_NEAREST)

    class Diag(Node):
        def __init__(self):
            super().__init__("refine_diag")
            self.frame = None
            self.round = 0
            self.last_t = 0.0
            self.n_img = 0
            self.n_det = 0
            self.create_subscription(CompressedImage, args.image_topic,
                                     self.on_img, qos_profile_sensor_data)
            self.create_subscription(PerceptionTargets, args.det_topic,
                                     self.on_det, 10)
            self.create_timer(3.0, self.on_tick)
            self.get_logger().info("等待 /image 与 %s ..." % args.det_topic)

        def on_tick(self):
            self.get_logger().info("状态: 图像=%d 帧 检测=%d 条 已诊断=%d 轮"
                                   % (self.n_img, self.n_det, self.round))

        def on_img(self, msg):
            self.n_img += 1
            try:
                self.frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                                          cv2.IMREAD_COLOR)
            except Exception:
                self.frame = None

        def on_det(self, msg):
            self.n_det += 1
            now = time.time()
            if self.frame is None or now - self.last_t < args.min_gap:
                return
            if self.round >= args.rounds:
                return
            # 与生产节点完全一致：形状在 msg.targets[i].type，框在 targets[i].rois[j]
            boxes = []
            for t in msg.targets:
                rois = getattr(t, "rois", None)
                if rois is None:
                    rois = []
                    for tr in (getattr(t, "tracks", None) or []):
                        rois.extend(list(getattr(tr, "rois", []) or []))
                for roi in rois:
                    conf = float(roi.confidence)
                    if conf < 0.25:
                        continue
                    boxes.append((conf,
                                  [int(roi.rect.x_offset), int(roi.rect.y_offset),
                                   int(roi.rect.x_offset + roi.rect.width),
                                   int(roi.rect.y_offset + roi.rect.height)],
                                  str(getattr(t, "type", "?"))))
            if not boxes:
                return
            self.round += 1
            self.last_t = now
            self.diag(boxes)

        def diag(self, boxes):
            frame = self.frame
            H, W = frame.shape[:2]
            boxes = [(c, [max(0, b[0]), max(0, b[1]),
                          min(W, b[2]), min(H, b[3])], s)
                     for c, b, s in boxes]
            boxes.sort(key=lambda b: -b[0])
            for i, (conf, box, shape) in enumerate(boxes[:args.max_boxes]):
                try:
                    self.diag_one(frame, box, shape, i, conf)
                except Exception as e:                   # 诊断工具不能拖垮采样
                    print("  ⚠ 诊断异常: %r" % (e,))

        def diag_one(self, frame, box, shape, idx, conf):
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return
            label, cconf = classify_color(roi)
            tag = "r%02d_b%d_%s_%s" % (self.round, idx, shape, label or "none")
            print("\n[%s] 框=%s 尺寸=%dx%d YOLO置信=%.2f 判色=%s(%.2f)"
                  % (tag, box, x2 - x1, y2 - y1, conf, label, cconf))
            if label is None:
                print("    ⚠ 颜色没判出来 —— 生产上这颗直接拒发")
                return

            # ---- 复刻 refine() 的裁剪，只是为了把中间量打出来 ----
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            margin = getattr(C, "GRASP_MASK_MARGIN", 0.12)
            mx, my = int(round(bw * margin)), int(round(bh * margin))
            X1, Y1 = max(0, x1 - mx), max(0, y1 - my)
            X2, Y2 = min(frame.shape[1], x2 + mx), min(frame.shape[0], y2 + my)
            crop = frame[Y1:Y2, X1:X2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            box_local = (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1)
            coarse = GR.color_mask(hsv, label)
            adapt = GR._adaptive_mask(hsv, coarse, box_local)

            for name, mk in (("粗掩膜", coarse), ("自适应", adapt)):
                if mk is None:
                    print("    %s: 无法生成" % name)
                    continue
                st = GR._analyse(mk, X1, Y1, box_local)
                if st is None:
                    print("    %s: 无有效连通域（像素 %d）" % (name, px(mk)))
                    continue
                why = GR._reject_reasons(st, bw, bh,
                                         getattr(C, "GRASP_MIN_FILL", 0.10))
                print("    %s: 像素=%d fill=%.2f 贴合框=%s ratio=(%.2f,%.2f) "
                      "solidity=%.2f coverage=%.2f %s"
                      % (name, st["mask_px"], st["mask_px"] / (bw * bh),
                         st["tight_bbox"], st["tw"] / bw, st["th"] / bh,
                         st["solidity"], st["coverage"],
                         ("✓可信" if not why else "✗ " + "/".join(why))))
                s = hsv_stats(hsv, mk)
                if s:
                    print("       框内HSV中位=(%.0f,%.0f,%.0f) "
                          "S[p10,p90]=[%.0f,%.0f] V[p10,p90]=[%.0f,%.0f]"
                          % s)

            res = GR.refine(frame, box, label)
            print("    生产判定: ok=%s method=%s 点=(%.1f,%.1f) 原因=%s"
                  % (res["ok"], res.get("method"),
                     res["point"][0], res["point"][1], res.get("reason")))

            if not args.no_save:
                d = os.path.join(args.out, tag)
                cv2.imwrite(d + "_crop.png", zoom(crop))
                if coarse is not None:
                    cv2.imwrite(d + "_mask_coarse.png", zoom(coarse))
                if adapt is not None:
                    cv2.imwrite(d + "_mask_adapt.png", zoom(adapt))
                print("    图: %s_*.png" % d)

    rclpy.init()
    node = Diag()
    try:
        while rclpy.ok() and node.round < args.rounds:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("采样结束：%d 轮，图在 %s"
                               % (node.round, args.out))
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass          # 超时被 SIGTERM 时 context 已经关了，别再抛
    return 0


if __name__ == "__main__":
    sys.exit(main())