# -*- coding: utf-8 -*-
"""离线验证：对 dataset 里的真实物料图片跑一次检测，确认推理链路正常"""
import sys, os, glob, inspect

sys.path.insert(0, r"D:\yolov5")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(r"C:\Users\34559\Desktop\GC\best.pt", device=device)
model.eval()
names = getattr(model, "names", None)
if not names and hasattr(model, "module"):
    names = getattr(model.module, "names", None)
print("names =", names)

# 找几张真实物料图片
imgs = sorted(glob.glob(r"C:\Users\34559\Desktop\GC\工训分拣\camera_calib\dataset\*\*.jpg"))[:5]
for p in imgs:
    img0 = cv2.imread(p)
    if img0 is None:
        continue
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB + HWC2CHW
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device).float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    res = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[1:], det[:, :4], img0.shape).round()
            for *xyxy, confv, cl in reversed(det):
                res.append((names.get(int(cl)), float(confv)))
    print(os.path.basename(p), "->", res if res else "无检测")
