# -*- coding: utf-8 -*-
"""测试 ultralytics 能否加载 YOLOv5 6.2 的 best.pt"""
import sys, traceback

try:
    from ultralytics import YOLO
    print("ultralytics import OK", file=sys.stderr)
    m = YOLO(r"C:\Users\34559\Desktop\GC\best.pt")
    print("加载成功", file=sys.stderr)
    print("names =", m.names, file=sys.stderr)
except Exception:
    traceback.print_exc()
    sys.exit(1)
