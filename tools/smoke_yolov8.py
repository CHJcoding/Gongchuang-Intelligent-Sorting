# -*- coding: utf-8 -*-
"""板端冒烟：抓一帧 → 自己推理解码 → 打印框 + 颜色一致性自检 → 存标注图。

用法（板端）：
    python3 tools/smoke_yolov8.py [--dev /dev/cam0] [--conf 0.25] [--save /tmp/annot.jpg]

自检原理：物料是单色塑料件，如果解码/坐标映射正确，框内应该
有清晰的颜色主色（覆盖率高）；如果框整体偏移或尺度错，覆盖率会很低。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 仓库布局是 rdk/vision_sorter/，板端布局是 vision_sorter/，两个都试
for _cand in (os.path.join(ROOT, "rdk", "vision_sorter"), os.path.join(ROOT, "vision_sorter")):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)

from yolov8_bpu import Yolov8Bpu, sigmoid  # noqa: E402

LABELS = "/root/vision_ws/gc_1029_labels.list"
BIN = "/root/vision_ws/gc_best_1029.bin"

# 和 config.py 保持一致的粗分色（只用来判断框对不对，不做业务判定）
HSV_RANGES = {
    "red": [((0, 40, 40), (12, 255, 255)), ((160, 40, 40), (180, 255, 255))],
    "green": [((35, 40, 40), (95, 255, 255))],
    "blue": [((95, 40, 40), (135, 255, 255))],
    "pink": [((140, 35, 90), (170, 255, 255))],
}


def open_cam(dev, width=640, height=480, fps=30):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("打不开相机 %s" % dev)
    for _ in range(8):          # 丢前期几帧，等 AWB/曝光稳定
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("读帧失败")
    return frame


def color_coverage(bgr, box):
    """返回 {颜色: 覆盖率和}，覆盖率 = 该色像素 / 框内像素。"""
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(bgr.shape[1], int(x2)), min(bgr.shape[0], int(y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return {}, 0
    patch = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    total = patch.shape[0] * patch.shape[1]
    out = {}
    for name, ranges in HSV_RANGES.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        out[name] = float((m > 0).sum()) / total
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/cam0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--save", default="/tmp/annot.jpg")
    ap.add_argument("--frames", type=int, default=10, help="连续跑几帧取平均耗时")
    args = ap.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    print("labels(%d): %s" % (len(labels), labels))

    frame = open_cam(args.dev)
    print("抓帧: %dx%d" % (frame.shape[1], frame.shape[0]))
    cv2.imwrite("/tmp/cap.jpg", frame)

    det = Yolov8Bpu(args.bin, labels, conf_thres=args.conf, iou_thres=args.iou)

    # 先看一眼原始 cls 值域，确认是概率还是 logits
    arrs, ratio, dw, dh = det.raw_forward(frame)
    cls0 = np.asarray(arrs[0], np.float32)
    print("cls[0] 值域: min=%.3f max=%.3f  (有负数或 >1 说明是 logits)"
          % (float(cls0.min()), float(cls0.max())))
    print("reg[1] dtype=%s 值域 min=%.3f max=%.3f"
          % (arrs[1].dtype, float(np.asarray(arrs[1], np.float32).min()),
             float(np.asarray(arrs[1], np.float32).max())))

    t0 = time.time()
    N = max(1, args.frames)
    for _ in range(N):
        results = det.detect(frame)
    dt = (time.time() - t0) / N
    print("推理+解码: %.1f ms/帧 -> %.1f fps" % (dt * 1000, 1.0 / dt))
    print("检出 %d 个目标" % len(results))

    vis = frame.copy()
    for i, d in enumerate(results):
        cov, total = color_coverage(frame, (d.x1, d.y1, d.x2, d.y2))
        cov_s = " ".join("%s=%.2f" % (k, v) for k, v in sorted(cov.items(), key=lambda kv: -kv[1])[:2])
        print("  #%d %-6s conf=%.2f box=(%.0f,%.0f)-(%.0f,%.0f) %.0fx%.0f 框内颜色: %s"
              % (i, d.name, d.score, d.x1, d.y1, d.x2, d.y2, d.w, d.h, cov_s))
        cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
        cv2.putText(vis, "%s %.2f" % (d.name, d.score), (int(d.x1), max(12, int(d.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(args.save, vis)
    print("标注图: %s" % args.save)


if __name__ == "__main__":
    main()