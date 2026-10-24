# -*- coding: utf-8 -*-
"""
YOLOv5n 检测，RDK X5 板端用地平线 BPU 推理 (pyeasy_dnn)。
依赖: hobot_dnn (RDK系统自带)
模型: best.pt -> onnx -> hb_model_transform -> .bin，放到 weights/ 目录
"""
import cv2
import numpy as np
import config as C


class ShapeDetector:
    def __init__(self):
        from hobot_dnn import pyeasy_dnn as dnn
        print(f"[detector] 加载BPU模型: {C.YOLO_BIN_MODEL}")
        self.models = dnn.load(C.YOLO_BIN_MODEL)
        self.input_h = self.models[0].inputs[0].properties.height
        self.input_w = self.models[0].inputs[0].properties.width
        print(f"[detector] 模型输入尺寸: {self.input_w}x{self.input_h}")

    def _preprocess(self, bgr):
        """resize + BGR->RGB，按你导出模型时的预处理对齐"""
        img = cv2.resize(bgr, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def detect(self, bgr_frame):
        """
        输入: BGR图
        返回: list of dict {shape, bbox, cx, cy, conf, roi}
        注意：后处理输出维度要按你导出的onnx微调，下面是YOLOv5n标准格式。
        """
        h0, w0 = bgr_frame.shape[:2]
        img_in = self._preprocess(bgr_frame)
        outputs = self.models[0].forward(img_in)

        # outputs[0].buffer 形状通常是 (1, N, 5+nc)
        out = outputs[0].buffer.reshape(-1, 5 + len(C.SHAPE_NAMES))
        sx = w0 / self.input_w
        sy = h0 / self.input_h

        raw = []
        for row in out:
            obj_conf = row[4]
            if obj_conf < C.YOLO_CONF_THRES:
                continue
            cls_id = int(np.argmax(row[5:]))
            score = float(obj_conf * row[5 + cls_id])
            if score < C.YOLO_CONF_THRES:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = int((cx - bw / 2) * sx)
            y1 = int((cy - bh / 2) * sy)
            x2 = int((cx + bw / 2) * sx)
            y2 = int((cy + bh / 2) * sy)
            raw.append((x1, y1, x2, y2, score, cls_id))

        dets = self._nms(raw, C.YOLO_IOU_THRES)
        result = []
        for x1, y1, x2, y2, score, cls_id in dets:
            roi = bgr_frame[max(0, y1):y2, max(0, x1):x2].copy()
            result.append({
                "shape": C.SHAPE_NAMES[cls_id],
                "bbox": (x1, y1, x2, y2),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "conf": score,
                "roi": roi,
            })
        return result

    @staticmethod
    def _nms(boxes, iou_thres):
        if not boxes:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return [tuple(boxes[i]) for i in keep]


if __name__ == "__main__":
    from camera import USBCamera
    det = ShapeDetector()
    cam = USBCamera()
    cam.open()
    print("按q退出")
    while True:
        f = cam.read()
        if f is None:
            continue
        results = det.detect(f)
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(f, f"{r['shape']} {r['conf']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("detect", f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.close()
    cv2.destroyAllWindows()
