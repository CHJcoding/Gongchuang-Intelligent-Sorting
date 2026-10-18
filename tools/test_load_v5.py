# -*- coding: utf-8 -*-
"""验证 D:\\yolov5 (master) 能否加载 v6.2 训练的 best.pt"""
import sys, os, inspect, traceback

REPO = r"D:\yolov5"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch

# torch 2.6+ 默认 weights_only=True 会拒绝 YOLOv5 老权重，包装 torch.load 强制兼容
_orig_load = torch.load
_supports_wo = "weights_only" in inspect.signature(torch.load).parameters

def _load_compat(*args, **kwargs):
    kwargs.pop("weights_only", None)
    if _supports_wo:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

torch.load = _load_compat

try:
    from models.experimental import attempt_load
    model = attempt_load(r"C:\Users\34559\Desktop\GC\best.pt", device="cpu")
    model.eval()
    names = getattr(model, "names", None)
    if not names and hasattr(model, "module"):
        names = getattr(model.module, "names", None)
    if isinstance(names, (list, tuple)):
        names = {i: str(n) for i, n in enumerate(names)}
    print("加载成功")
    print("names =", names)
    print("类型 =", type(model).__name__)
finally:
    torch.load = _orig_load
