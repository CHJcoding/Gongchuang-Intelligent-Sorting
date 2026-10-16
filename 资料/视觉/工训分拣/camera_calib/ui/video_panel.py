# -*- coding: utf-8 -*-
"""视频预览面板（PyQt5 版）：QLabel 显示 OpenCV 帧，鼠标点击选点 + 拖拽框选 ROI"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


class VideoPanel(QLabel):
    """视频预览组件：显示 OpenCV 帧 + 叠加角点/检测框/ROI，支持点击选点与拖拽框选"""

    point_clicked = pyqtSignal(float, float)  # 原始帧坐标 (x, y)
    roi_changed = pyqtSignal(object)          # 框选范围 (x1, y1, x2, y2)；清除时 None

    def __init__(self, width=640, height=480, parent=None):
        super().__init__(parent)
        self.setMinimumSize(width, height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000000; border: 1px solid #888888;")

        self.current_frame = None
        self.points = []            # 手动选点（原始帧坐标）
        self.point_mode = False
        self.expected_points = 0
        self._scale = 1.0
        self._offset = (0, 0)

        # ROI 框选状态
        self.roi_mode = False       # 是否处于框选模式
        self.roi = None             # (x1, y1, x2, y2) 原始帧坐标；None 表示未框选
        self._dragging = False      # 是否正在拖拽
        self._drag_start = None     # 拖拽起点（原始帧坐标）
        self._last_boxes = None     # 最近一次叠加的检测框，拖拽重绘时复用

    def update_frame(self, frame, extra_points=None, boxes=None):
        """刷新一帧显示；extra_points 为要叠加显示的角点（原始帧坐标），
        boxes 为检测框列表 [(x1, y1, x2, y2, label, conf)]（原始帧坐标）"""
        if frame is None:
            return
        self.current_frame = frame.copy()
        h, w = self.current_frame.shape[:2]

        # 缩放适配控件
        disp_w = self.width() - 4
        disp_h = self.height() - 4
        scale = min(disp_w / w, disp_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        self._scale = scale
        self._offset = ((self.width() - new_w) // 2, (self.height() - new_h) // 2)

        disp = cv2.resize(self.current_frame, (new_w, new_h))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        # 叠加角点
        pts = extra_points if extra_points is not None else self.points
        for p in pts:
            px = int(p[0] * scale)
            py = int(p[1] * scale)
            cv2.circle(rgb, (px, py), 3, (255, 0, 0), -1)
            cv2.circle(rgb, (px, py), 5, (255, 255, 255), 1)

        # 叠加检测框（保存副本供拖拽重绘时复用）
        self._last_boxes = boxes
        if boxes:
            for x1, y1, x2, y2, label, conf in boxes:
                px1, py1, px2, py2 = (int(x1 * scale), int(y1 * scale),
                                      int(x2 * scale), int(y2 * scale))
                cv2.rectangle(rgb, (px1, py1), (px2, py2), (0, 255, 0), 2)
                if label:
                    text = f"{label} {conf:.2f}" if conf is not None else str(label)
                    cv2.putText(rgb, text, (px1, max(0, py1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.point_mode:
            cv2.putText(rgb, f"手动选点: {len(self.points)}/{self.expected_points}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 叠加 ROI：框外半透明压暗 + 黄色矩形框线，直观显示识别范围
        if self.roi is not None:
            rx1 = int(self.roi[0] * scale)
            ry1 = int(self.roi[1] * scale)
            rx2 = int(self.roi[2] * scale)
            ry2 = int(self.roi[3] * scale)
            mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
            dark = (rgb * 0.45).astype(np.uint8)  # 框外区域压暗
            rgb = np.where(mask[..., None] == 255, rgb, dark)
            cv2.rectangle(rgb, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(rgb, "ROI", (rx1, max(0, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.roi_mode:
            cv2.putText(rgb, "左键拖拽框选识别范围，点击空白处清除",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg.copy()))

    def set_point_mode(self, enabled, expected=0):
        self.point_mode = enabled
        self.expected_points = expected
        if not enabled:
            self.points = []

    def clear_points(self):
        self.points = []

    # ---------- ROI 框选 ----------
    def set_roi_mode(self, enabled):
        """开启/关闭拖拽框选模式；开启后鼠标拖拽选择识别范围"""
        self.roi_mode = enabled
        self._dragging = False
        self._drag_start = None

    def set_roi(self, roi):
        """设置识别范围（原始帧坐标 (x1, y1, x2, y2)）；None 表示清除"""
        self.roi = roi
        self._dragging = False
        self._drag_start = None

    def clear_roi(self):
        """清除识别范围"""
        self.set_roi(None)

    def _to_frame_coord(self, ex, ey):
        """控件坐标 -> 原始帧坐标；越界返回 None"""
        if self.current_frame is None:
            return None
        x = (ex - self._offset[0]) / self._scale
        y = (ey - self._offset[1]) / self._scale
        h, w = self.current_frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return float(x), float(y)
        return None

    def mousePressEvent(self, event):
        if self.current_frame is None:
            return
        pos = self._to_frame_coord(event.x(), event.y())
        if pos is None:
            return
        # 框选模式：按下开始拖拽
        if self.roi_mode:
            self._dragging = True
            self._drag_start = pos
            return
        # 选点模式
        if not self.point_mode:
            return
        self.points.append(pos)
        self.point_clicked.emit(pos[0], pos[1])

    def mouseMoveEvent(self, event):
        # 拖拽过程中实时更新 ROI 预览
        if not self.roi_mode or not self._dragging or self._drag_start is None:
            return
        pos = self._to_frame_coord(event.x(), event.y())
        if pos is None or self.current_frame is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = pos
        self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.update_frame(self.current_frame, boxes=self._last_boxes)

    def mouseReleaseEvent(self, event):
        # 松开完成框选；几乎没移动视为点击，清除框选
        if not self.roi_mode or not self._dragging:
            return
        self._dragging = False
        pos = self._to_frame_coord(event.x(), event.y())
        start = self._drag_start
        self._drag_start = None
        if pos is None or start is None:
            return
        x1, y1 = start
        x2, y2 = pos
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.roi = None
            self.roi_changed.emit(None)
        else:
            self.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.roi_changed.emit(self.roi)
        if self.current_frame is not None:
            self.update_frame(self.current_frame, boxes=self._last_boxes)
