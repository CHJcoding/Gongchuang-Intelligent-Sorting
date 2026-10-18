# -*- coding: utf-8 -*-
"""YOLOv5 摄像头实时检测（物料识别效果验证）

用法：
    python detect_webcam.py                 # 默认相机 0，置信度 0.25
    python detect_webcam.py --source 1 --conf 0.3 --imgsz 640

按键：q / ESC 退出
"""
import argparse
import sys
import os

# 使用本地 yolov5 仓库（D:\yolov5，master 版，兼容 v6.2 权重）
REPO = r"D:\yolov5"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# torch 2.6+ 加载 v6.2 老权重需要兼容包装
import inspect
import torch

_orig_load = torch.load
_supports_wo = "weights_only" in inspect.signature(torch.load).parameters

def _load_compat(*args, **kwargs):
    kwargs.pop("weights_only", None)
    if _supports_wo:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

torch.load = _load_compat

import cv2
import numpy as np
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# 类别中文名（按训练 data 顺序，best.pt names）
CLS_CN = {
    "BQ": "半球", "LK": "镂空", "SLZ": "三棱柱",
    "SMT": "四面体", "WLZ": "五棱柱", "YZ": "圆柱", "YZHU": "圆锥",
}
CLS_COLOR = (0, 255, 0)


def load_model(weights, device):
    model = attempt_load(weights, device=device)
    model.eval()
    names = getattr(model, "names", None)
    if not names and hasattr(model, "module"):
        names = getattr(model.module, "names", None)
    if isinstance(names, (list, tuple)):
        names = {i: str(n) for i, n in enumerate(names)}
    return model, names


def detect_frame(model, names, img0, device, conf=0.25, iou=0.45, imgsz=640):
    # letterbox -> RGB -> CHW -> 归一化 -> 推理 -> NMS -> 回原图坐标
    img = letterbox(img0, imgsz, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t)[0]
    pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)
    out = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[1:], det[:, :4], img0.shape).round()
            for *xyxy, confv, cl in reversed(det):
                out.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),
                            int(xyxy[3]), int(cl), float(confv)))
    return out


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 摄像头物料检测")
    parser.add_argument("--weights", default=r"C:\Users\34559\Desktop\GC\best.pt")
    parser.add_argument("--source", type=int, default=2, help="摄像头编号，默认 0")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=640, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=480, help="摄像头采集高度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[加载模型] {args.weights} @ {device}")

    # torch.load 兼容包装只在本脚本生效，加载完恢复
    model, names = load_model(args.weights, device)
    torch.load = _orig_load
    print(f"[类别] {names}")
    print(f"[相机] 打开 {args.source} ...")

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[错误] 无法打开相机 {args.source}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("[就绪] 按 q / ESC 退出")
    fps_time = cv2.getTickCount()
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        dets = detect_frame(model, names, frame, device,
                            conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        for x1, y1, x2, y2, cl, confv in dets:
            label = names.get(int(cl), str(int(cl)))
            # cv2.putText 不支持中文，标签直接用模型类名（英文/拼音，如 YZ、YZHU）
            text = f"{label} {confv:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), CLS_COLOR, 2)
            cv2.putText(frame, text, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLS_COLOR, 2)

        now = cv2.getTickCount()
        dt = (now - fps_time) / cv2.getTickFrequency()
        if dt >= 1.0:
            fps = 1.0 / dt
            fps_time = now
        cv2.putText(frame, f"FPS {fps:.1f}  det {len(dets)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLOv5 物料检测 (q/ESC 退出)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[结束]")


if __name__ == "__main__":
    main()
