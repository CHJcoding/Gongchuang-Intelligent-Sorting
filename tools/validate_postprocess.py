# -*- coding: utf-8 -*-
"""
validate_postprocess.py — 本地验证 NHWC 后处理是否正确（不需要开发板）
================================================================================
思路（同一输入、两套解码，互为对照）：
    A. 参考真值：torch 加载 best.pt，模型内部自己解码 → NMS → 框
    B. 待验证：onnxruntime 跑 best.onnx 拿 3 个 NHWC raw → yolo_postprocess 解码 → NMS → 框
    预处理只用一套（yolov5 官方 letterbox），确保差异只来自解码本身。

另外单独验证：自写 letterbox 与 yolov5 官方 letterbox 是否逐像素一致。

用法：
    python validate_postprocess.py                # 默认抽 5 张
    python validate_postprocess.py --num 10
"""
import argparse
import sys
import os
import glob
import inspect

import cv2
import numpy as np

YOLOV5_REPO = r"D:\yolov5"
if YOLOV5_REPO not in sys.path:
    sys.path.insert(0, YOLOV5_REPO)

import torch

# torch 2.6+ 加载 v6.2 老权重需要兼容（weights_only 默认改了）
_orig_load = torch.load
_supports_wo = "weights_only" in inspect.signature(torch.load).parameters


def _load_compat(*a, **k):
    k.pop("weights_only", None)
    if _supports_wo:
        k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _load_compat

from models.experimental import attempt_load              # noqa: E402
from utils.augmentations import letterbox as v5_letterbox  # noqa: E402
from utils.general import non_max_suppression             # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "vision_sorter"))
import yolo_postprocess as yp                             # noqa: E402

GC = r"C:\Users\34559\Desktop\GC"
PT = os.path.join(GC, "best.pt")
ONNX = os.path.join(GC, "best.onnx")
IMG_DIR = os.path.join(GC, "mydata", "images", "test")
NC = 7
IMGSZ = 672
CONF = 0.25
IOU = 0.45


def iou_xyxy(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def check_letterbox():
    """自写 letterbox vs yolov5 官方 letterbox。"""
    print("=" * 70)
    print("[测试 1] letterbox 与 yolov5 官方实现一致性")
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))[:3]
    ok_all = True
    for p in imgs:
        im = cv2.imread(p)
        mine, r1, (px1, py1) = yp.letterbox(im, (IMGSZ, IMGSZ), stride=32, auto=False)
        ref = v5_letterbox(im, IMGSZ, stride=32, auto=False)[0]
        same_shape = mine.shape == ref.shape
        same_px = same_shape and int(np.abs(mine.astype(np.int16) -
                                            ref.astype(np.int16)).max()) <= 0
        dist_max = (int(np.abs(mine.astype(np.int16) - ref.astype(np.int16)).max())
                    if same_shape else -1)
        ok = same_px
        ok_all &= ok
        print(f"  {os.path.basename(p)}: shape {mine.shape} vs {ref.shape} | "
              f"最大像素差={dist_max} | ratio={r1:.6f} pad=({px1},{py1}) | "
              f"{'OK' if ok else '不一致'}")
    print(f"  结论: {'一致' if ok_all else '不一致（需修）'}")
    return ok_all


def load_models():
    torch_model = attempt_load(PT, device="cpu").eval()
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    torch.load = _orig_load
    return torch_model, sess, in_name


def run_one(torch_model, sess, in_name, img_path):
    im0 = cv2.imread(img_path)
    if im0 is None:
        return None

    # ---- 统一预处理：yolov5 官方 letterbox（auto=False：固定方形输入）----
    padded = v5_letterbox(im0, IMGSZ, stride=32, auto=False)[0]
    rgb = padded[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0

    # ---- A. torch 参考真值 ----
    with torch.no_grad():
        pred = torch_model(torch.from_numpy(x))[0]
    det = non_max_suppression(pred, CONF, IOU)[0]
    boxes_a = []
    if det is not None and len(det):
        for *xyxy, conf, cl in det:
            boxes_a.append(([float(v) for v in xyxy], float(conf), int(cl)))

    # ---- B. onnx + 自写解码 ----
    outs = sess.run(None, {in_name: x})
    order = np.argsort([o.shape[1] for o in outs])[::-1]   # 大网格在前
    outs = [outs[i] for i in order]
    b, s, c = yp.decode_nhwc(outs, nc=NC, conf_thres=CONF, iou_thres=IOU)
    boxes_b = [([float(v) for v in b[i]], float(s[i]), int(c[i]))
               for i in range(len(b))]

    # ---- 匹配 ----
    used = set()
    matches, ious = 0, []
    for ba, sa, ca in boxes_a:
        best_j, best_iou = -1, 0.0
        for j, (bb, sb, cb) in enumerate(boxes_b):
            if j in used or cb != ca:
                continue
            v = iou_xyxy(ba, bb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= 0.5:
            used.add(best_j)
            matches += 1
            ious.append(best_iou)

    return {
        "file": os.path.basename(img_path),
        "n_a": len(boxes_a), "n_b": len(boxes_b),
        "matched": matches,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "classes_a": sorted(ca for _, _, ca in boxes_a),
        "classes_b": sorted(cb for _, _, cb in boxes_b),
        "conf_a": [round(sa, 3) for _, sa, _ in boxes_a][:5],
        "conf_b": [round(sb, 3) for _, sb, _ in boxes_b][:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--skip-letterbox", action="store_true")
    args = ap.parse_args()

    lb_ok = True if args.skip_letterbox else check_letterbox()

    print("=" * 70)
    print(f"[测试 2] 解码结果对照（torch 真值 vs onnx+自写解码）  输入 {IMGSZ}")
    torch_model, sess, in_name = load_models()
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))[:args.num]

    total_a = total_b = total_m = 0
    iou_all = []
    for p in imgs:
        r = run_one(torch_model, sess, in_name, p)
        if r is None:
            continue
        total_a += r["n_a"]; total_b += r["n_b"]; total_m += r["matched"]
        if r["mean_iou"]:
            iou_all.append(r["mean_iou"])
        flag = "OK" if (r["n_a"] == r["n_b"] == r["matched"]) else "差异"
        print(f"  {r['file']}: torch={r['n_a']} mine={r['n_b']} "
              f"匹配={r['matched']} 平均IoU={r['mean_iou']:.4f} [{flag}]")
        if flag == "差异":
            print(f"     torch classes={r['classes_a']}")
            print(f"     mine  classes={r['classes_b']}")
            print(f"     torch conf={r['conf_a']}")
            print(f"     mine  conf={r['conf_b']}")

    print("-" * 70)
    print(f"总计: torch 框={total_a}  自写解码框={total_b}  成功匹配={total_m}")
    if iou_all:
        print(f"匹配框平均 IoU = {np.mean(iou_all):.4f}  (越高越好，≥0.95 视为等价)")
    ratio = (total_m / total_a) if total_a else 0.0
    print(f"检出一致性 = {ratio*100:.1f}%")
    ok = (lb_ok and total_a == total_b == total_m)
    print("=" * 70)
    print("结论:", "✅ 后处理与 torch 参考完全一致" if ok
          else "❌ 仍有差异，需排查（见上方明细）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())