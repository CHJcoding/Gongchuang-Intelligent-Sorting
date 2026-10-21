# -*- coding: utf-8 -*-
"""
yolo_postprocess.py — YOLOv5 NHWC raw 输出后处理（RDK X5 BPU / ONNX 通用）
================================================================================
背景（2026-10-21 基线）：
    best.onnx 的 3 个输出是 **NHWC raw 特征图**，形状
        (1, 84, 84, 36), (1, 42, 42, 36), (1, 21, 21, 36)      # 672 输入
    36 = 3 anchors × (5 + 7 classes)
    即模型内部已 permute(0,2,3,1)，且 **未做 sigmoid / 未解码**。
    旧 detector.py 按「已解码 (1,N,5+nc)」读取，必然全错。

本模块负责：letterbox → 模型 → NHWC 解码 → NMS → 逆 letterbox → 原图坐标。
anchors 与 best.pt 训练时完全一致，勿改（改了会整体错位）。
"""
import cv2
import numpy as np

# ============ 与 best.pt 完全一致（来自 model.yaml['anchors']） ============
DEFAULT_ANCHORS = np.array([
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
], dtype=np.float32)                      # (3 scale, 3 anchor, 2)
DEFAULT_STRIDES = (8, 16, 32)
NUM_ANCHORS_PER_SCALE = 3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


# ============================================================
# letterbox：与 yolov5 utils.augmentations.letterbox(auto=True) 对齐
# ============================================================
def letterbox(im, new_shape=(672, 672), color=(114, 114, 114),
              stride=32, auto=True, scaleup=True):
    """保持纵横比缩放 + 居中填充。

    返回: (img, ratio, (pad_x, pad_y))
        ratio: 缩放系数（宽高同一个值）
        pad_x/pad_y: 左侧/顶部填充像素
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    shape = im.shape[:2]                    # 当前 (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))   # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2

    if (shape[1], shape[0]) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def unletterbox(boxes, ratio, pad):
    """把 letterbox 坐标系下的 xyxy 框还原到原图坐标。"""
    if not len(boxes):
        return boxes
    out = np.asarray(boxes, dtype=np.float32).copy()
    px, py = pad
    out[:, [0, 2]] = (out[:, [0, 2]] - px) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - py) / ratio
    return out


def preprocess(bgr, input_size, stride=32):
    """BGR 原图 → 模型输入张量 NCHW float32 (1,3,H,W)，归一化到 0~1。

    注意：ONNX / BPU 模型输入尺寸是**固定方形**，必须 auto=False 完整补齐；
    用 auto=True 只会补到 stride 倍数（640×480 的图会变成 512×672），
    非方形输入喂固定输入模型会直接报维度错误。
    """
    img, ratio, pad = letterbox(bgr, (input_size, input_size),
                                stride=stride, auto=False)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(chw), ratio, pad


# ============================================================
# NHWC raw 解码
# ============================================================
def decode_nhwc(outputs, nc=7, anchors=DEFAULT_ANCHORS,
                strides=DEFAULT_STRIDES, conf_thres=0.25, iou_thres=0.45):
    """解码 3 个 NHWC raw 张量。

    Args:
        outputs: [(1, ny, nx, 3*(5+nc)), ...] 或 [(ny, nx, 3*(5+nc)), ...]
    Returns:
        boxes(原模型输入像素坐标系 xyxy), scores, class_ids
    """
    boxes, scores, class_ids = [], [], []

    for si, out in enumerate(outputs):
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 4:
            out = out[0]
        if out.ndim != 3:
            raise ValueError(f'输出 {si} 维度异常: {out.shape}')
        ny, nx, ch = out.shape
        expect = NUM_ANCHORS_PER_SCALE * (5 + nc)
        if ch != expect:
            raise ValueError(f'输出 {si} 通道数 {ch} != 3*(5+{nc})={expect}；'
                             f'检查 nc 是否与模型类别数一致')

        g = out.reshape(ny, nx, NUM_ANCHORS_PER_SCALE, 5 + nc)
        grid_x, grid_y = np.meshgrid(np.arange(nx, dtype=np.float32),
                                     np.arange(ny, dtype=np.float32))
        grid = np.stack((grid_x, grid_y), axis=-1)              # (ny,nx,2)

        # 中心点：(sigmoid*2 - 0.5 + grid) * stride
        xy = (sigmoid(g[..., 0:2]) * 2.0 - 0.5 + grid[..., None, :]) * strides[si]
        # 宽高：(sigmoid*2)^2 * anchor
        wh = (sigmoid(g[..., 2:4]) * 2.0) ** 2 * anchors[si]
        obj = sigmoid(g[..., 4:5])
        cls = sigmoid(g[..., 5:])

        conf = obj * cls.max(axis=-1, keepdims=True)            # (ny,nx,3,1)
        keep = conf[..., 0] > conf_thres
        if not keep.any():
            continue

        c = xy[keep]
        s = wh[keep]
        boxes.append(np.concatenate([c - s / 2, c + s / 2], axis=-1))
        scores.append(conf[keep][:, 0])
        class_ids.append(cls[keep].argmax(axis=-1))

    if not boxes:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32))

    b = np.concatenate(boxes)
    sc = np.concatenate(scores)
    ci = np.concatenate(class_ids).astype(np.int32)
    keep = nms(b, sc, iou_thres)
    return b[keep], sc[keep], ci[keep]


def nms(boxes, scores, iou_thres):
    """按类别无关的贪心 NMS，返回保留索引。"""
    if not len(boxes):
        return np.zeros((0,), np.int64)
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return np.array(keep, dtype=np.int64)


def postprocess(outputs, orig_shape, ratio, pad, nc=7, conf_thres=0.25,
                iou_thres=0.45, anchors=DEFAULT_ANCHORS,
                strides=DEFAULT_STRIDES, clip=True):
    """一步到位：raw 输出 → 原图坐标检测框。

    Returns: list of dict(bbox=(x1,y1,x2,y2), conf, cls_id)
    """
    b, s, c = decode_nhwc(outputs, nc=nc, anchors=anchors, strides=strides,
                          conf_thres=conf_thres, iou_thres=iou_thres)
    b = unletterbox(b, ratio, pad)
    if clip and len(b):
        h, w = orig_shape[:2]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
    return [{'bbox': tuple(map(float, b[i])), 'conf': float(s[i]),
             'cls_id': int(c[i])} for i in range(len(b))]


class YoloV5Postproc:
    """带默认参数的封装，方便节点里直接调用。"""

    def __init__(self, nc=7, input_size=672, conf_thres=0.25, iou_thres=0.45,
                 anchors=DEFAULT_ANCHORS, strides=DEFAULT_STRIDES):
        self.nc = nc
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.strides = tuple(strides)

    def infer_input(self, bgr):
        return preprocess(bgr, self.input_size)

    def __call__(self, outputs, orig_shape, ratio, pad):
        return postprocess(outputs, orig_shape, ratio, pad, nc=self.nc,
                           conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                           anchors=self.anchors, strides=self.strides)