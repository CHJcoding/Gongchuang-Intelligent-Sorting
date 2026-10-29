# -*- coding: utf-8 -*-
"""
YOLOv8 BPU 推理（RDK X5，hobot_dnn.pyeasy_dnn）——不依赖 TROS 的 C++ parser。

为什么不用官方 dnn_node_example：
    官方 yolov8 parser 要求 reg 分支输出是 int32（量化后的原始 DFL logits），
    而本项目新导入的模型 reg 输出是 float32，C++ parser 会把它当 int32 读 →
    段错误（实测 exit code -11）。这里自己在 Python 里解码，float32 反而更好解。

模型输出布局（实测 gc_best_1029.bin）：
    out[0] (1,80,80,C) cls   float32   out[1] (1,80,80,64) reg float32
    out[2] (1,40,40,C) cls              out[3] (1,40,40,64) reg
    out[4] (1,20,20,C) cls              out[5] (1,20,20,64) reg
    C = 类别数；reg 每 16 个 bin 一组，共 4 组 (l,t,r,b) → DFL

解码流程：
    softmax(reg 的 16 bin) * arange(16) → 4 个方向的距离（单位=该尺度的 stride）
    box = (grid+0.5)*stride ± distance → 再反向 letterbox 回原图坐标

预处理用 letterbox（等比缩放 + 114 填充），和 ultralytics 训练时一致；
官方 TROS 是直接拉伸，两者都能跑，但 letterbox 的框更贴。
"""
import numpy as np

try:
    import cv2
except ImportError:  # 纯逻辑测试时可以不装 cv2
    cv2 = None

DEFAULT_STRIDES = (8, 16, 32)
REG_BINS = 16
PAD_VALUE = 114


class Detection(object):
    """一个检测结果，坐标是原图（相机图像）像素。"""

    __slots__ = ("x1", "y1", "x2", "y2", "score", "cls_id", "name")

    def __init__(self, x1, y1, x2, y2, score, cls_id, name):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        self.score, self.cls_id, self.name = float(score), int(cls_id), name

    # --- 便捷属性（和 ai_msgs Roi / 业务代码口径一致）---
    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2.0

    def as_roi(self):
        """给业务代码/annotate 用的 dict（x_offset/y_offset/width/height 口径）。"""
        return {
            "x_offset": int(round(self.x1)),
            "y_offset": int(round(self.y1)),
            "width": int(round(self.w)),
            "height": int(round(self.h)),
            "type": self.name,
            "confidence": self.score,
        }

    def clipped(self, w, h):
        return Detection(max(0.0, self.x1), max(0.0, self.y1),
                         min(float(w - 1), self.x2), min(float(h - 1), self.y2),
                         self.score, self.cls_id, self.name)

    def __repr__(self):
        return "<Det %s %.2f (%.0f,%.0f)-(%.0f,%.0f)>" % (
            self.name, self.score, self.x1, self.y1, self.x2, self.y2)


# --------------------------------------------------------------------------
# 纯 numpy 的后处理（可离线单测）
# --------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


def dfl_decode(reg, stride, bins=REG_BINS):
    """reg (h,w,4*bins) → (h,w,4) 的 ltrb 距离，单位是像素（已乘 stride）。

    DFL: 每个方向用 softmax 把 bins 个 logits 变成权重，再对 arange(bins) 求期望。
    """
    reg = np.asarray(reg, dtype=np.float32)
    h, w = reg.shape[0], reg.shape[1]
    r = reg.reshape(h, w, 4, bins)
    r = r - r.max(axis=-1, keepdims=True)          # 数值稳定
    e = np.exp(r)
    p = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
    d = (p * np.arange(bins, dtype=np.float32)).sum(axis=-1)  # (h,w,4)
    return d * float(stride)


def dist2bbox(dist, stride):
    """(h,w,4) ltrb 距离 + 网格中心 → (h,w,4) 的 xyxy（letterbox 坐标系）。"""
    h, w = dist.shape[0], dist.shape[1]
    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    cx = (gx + 0.5) * stride
    cy = (gy + 0.5) * stride
    x1 = cx - dist[..., 0]
    y1 = cy - dist[..., 1]
    x2 = cx + dist[..., 2]
    y2 = cy + dist[..., 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def nms(boxes, scores, iou_thres):
    """标准 NMS，输入 xyxy + score，返回保留下标。"""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thres]
    return keep


def decode_outputs(outputs, input_w, input_h, num_classes, strides=DEFAULT_STRIDES,
                   conf_thres=0.25, iou_thres=0.45, cls_is_logits=None):
    """把 6 个输出张量解成 letterbox 坐标系下的 (boxes, scores, cls_ids)。

    outputs: [(cls_arr, reg_arr)] * 3 或扁平 [cls0, reg0, cls1, reg1, cls2, reg2]
    返回 (boxes Nx4 xyxy, scores N, cls_ids N)
    """
    if len(outputs) == 6:
        pairs = [(outputs[0], outputs[1]), (outputs[2], outputs[3]), (outputs[4], outputs[5])]
    else:  # 允许调用方自己配对
        pairs = outputs

    # 判断 cls 是概率还是 logits：有负数或 >1 就认为是 logits
    if cls_is_logits is None:
        probe = np.asarray(pairs[0][0], dtype=np.float32)
        cls_is_logits = bool(probe.size and (probe.min() < 0.0 or probe.max() > 1.0))

    all_boxes, all_scores, all_cls = [], [], []
    for (cls, reg), stride in zip(pairs, strides):
        cls = np.asarray(cls, dtype=np.float32)
        reg = np.asarray(reg, dtype=np.float32)
        if cls.ndim == 4:
            cls = cls[0]
        if reg.ndim == 4:
            reg = reg[0]
        h, w = cls.shape[0], cls.shape[1]
        if h * w * num_classes != cls.size:
            # 形状对不上就别瞎解，直接报出来（宁可少检也不要错框）
            raise ValueError("cls 形状 %s 与 num_classes=%d 不匹配（stride %d）"
                             % (cls.shape, num_classes, stride))
        scores = sigmoid(cls) if cls_is_logits else cls
        # 每个网格取最大类别
        cls_ids = scores.argmax(axis=-1).astype(np.int32)
        best = scores.max(axis=-1)
        mask = best >= conf_thres
        if not mask.any():
            continue
        dist = dfl_decode(reg, stride)
        boxes = dist2bbox(dist, stride)
        all_boxes.append(boxes[mask])
        all_scores.append(best[mask])
        all_cls.append(cls_ids[mask])

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids = np.concatenate(all_cls, axis=0)

    # 裁剪到输入范围内，再逐类 NMS
    boxes[:, 0] = np.clip(boxes[:, 0], 0, input_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, input_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, input_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, input_h - 1)

    keep = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        k = nms(boxes[idx], scores[idx], iou_thres)
        keep.extend(idx[k].tolist())
    keep.sort(key=lambda i: -scores[i])
    return boxes[keep], scores[keep], cls_ids[keep]


def letterbox_forward(boxes, ratio, dw, dh):
    """letterbox 坐标 → 原图坐标。"""
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / ratio
    return out


def letterbox_params(src_w, src_h, dst_w, dst_h):
    """等比缩放参数，返回 (ratio, dw, dh)。"""
    ratio = min(float(dst_w) / src_w, float(dst_h) / src_h)
    dw = (dst_w - src_w * ratio) / 2.0
    dh = (dst_h - src_h * ratio) / 2.0
    return ratio, dw, dh


# --------------------------------------------------------------------------
# 图像工具（需要 cv2）
# --------------------------------------------------------------------------
def letterbox_image(bgr, dst_w, dst_h, pad=PAD_VALUE):
    """等比缩放到 dst 内并居中填充，返回 (canvas, ratio, dw, dh)。"""
    h, w = bgr.shape[:2]
    ratio, dw, dh = letterbox_params(w, h, dst_w, dst_h)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    if (nw, nh) != (w, h):
        rs = cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (nw, nh), interpolation=rs)
    canvas = np.full((dst_h, dst_w, 3), pad, dtype=bgr.dtype)
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas[top:top + nh, left:left + nw] = bgr
    return canvas, ratio, dw, dh


def bgr_to_nv12(bgr):
    """BGR → NV12，返回**一维** uint8 数组（长度 H*3/2*W），BPU 的 nv12 输入口径。

    重要：喂给 pyeasy_dnn.forward 时必须是 1 维或 (H*3/2, W) 二维数组。
    如果按“官方示例”的形状 reshape 成 (1, H*3/2, W, 1) 再传进去，
    pyeasy_dnn 3.0.4 会静默不更新输入缓冲区 —— 表现为**任何输入得到同一份输出**
    （2026-10-29 在 X5 上实测：二维/一维数组喂进去，输出与 hrt_model_exec
    的 dump 逐字节一致；四维则两种完全不同输入的输出 md5 完全相同）。
    所以这里直接返回一维，宁可调用方自己 reshape。

    NV12 = Y 平面 + UV 交错平面。cv2 的 YUV_I420 是 Y+U+V 三平面，
    所以取出 U/V 后重新交错。
    """
    h, w = bgr.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("nv12 需要偶数宽高，收到 %dx%d" % (w, h))
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    # 别依赖 cv2 的“行打包”细节，直接按平面大小切：Y=h*w, U=h*w/4, V=h*w/4
    flat = yuv.reshape(-1)
    y_size = h * w
    c_size = y_size // 4
    y = flat[:y_size].reshape(h, w)
    u = flat[y_size:y_size + c_size].reshape(h // 2, w // 2)
    v = flat[y_size + c_size:y_size + 2 * c_size].reshape(h // 2, w // 2)
    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h] = y
    uv = nv12[h:].reshape(h // 2, w // 2, 2)
    uv[..., 0] = u
    uv[..., 1] = v
    return nv12.reshape(-1)          # 一维，见函数说明


# --------------------------------------------------------------------------
# 检测器
# --------------------------------------------------------------------------
class Yolov8Bpu(object):
    """板端用：装载 .bin，跑 BPU 推理 + Python 解码。PC 上无法用（没有 hobot_dnn）。"""

    def __init__(self, bin_path, labels, conf_thres=0.25, iou_thres=0.45,
                 strides=DEFAULT_STRIDES, verbose=True):
        from hobot_dnn import pyeasy_dnn as dnn  # 只有板端有
        self.labels = list(labels)
        self.num_classes = len(self.labels)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.strides = tuple(strides)
        self.bin_path = bin_path
        if verbose:
            print("[yolov8] 加载 BPU 模型: %s" % bin_path)
        self.model = dnn.load(bin_path)[0]
        shape = self.model.inputs[0].properties.shape
        # (1,3,H,W)
        self.in_h, self.in_w = int(shape[2]), int(shape[3])
        self.cls_is_logits = None
        if verbose:
            print("[yolov8] 模型输入 %dx%d，输出 %d 个，类别 %d 个"
                  % (self.in_w, self.in_h, len(self.model.outputs), self.num_classes))

    def raw_forward(self, bgr):
        """跑一次推理，返回 6 个 numpy 输出（已 reshape 成 4 维）。"""
        canvas, ratio, dw, dh = letterbox_image(bgr, self.in_w, self.in_h)
        nv12 = bgr_to_nv12(canvas)
        out = self.model.forward(nv12)
        arrs = [np.asarray(o.buffer).reshape(o.buffer.shape) for o in out]
        return arrs, ratio, dw, dh

    def detect(self, bgr, conf_thres=None, iou_thres=None):
        """返回 Detection 列表，坐标已映射回 bgr 的原图。"""
        arrs, ratio, dw, dh = self.raw_forward(bgr)
        boxes, scores, cls_ids = decode_outputs(
            arrs, self.in_w, self.in_h, self.num_classes,
            strides=self.strides,
            conf_thres=self.conf_thres if conf_thres is None else conf_thres,
            iou_thres=self.iou_thres if iou_thres is None else iou_thres,
            cls_is_logits=self.cls_is_logits)
        if len(boxes) == 0:
            return []
        boxes = letterbox_forward(boxes, ratio, dw, dh)
        h0, w0 = bgr.shape[:2]
        out = []
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls_ids):
            name = self.labels[c] if 0 <= c < len(self.labels) else str(c)
            d = Detection(x1, y1, x2, y2, s, c, name).clipped(w0, h0)
            if d.w >= 2 and d.h >= 2:
                out.append(d)
        return out