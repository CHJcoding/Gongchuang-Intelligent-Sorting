# -*- coding: utf-8 -*-
"""色块识别页（PyQt5 版）：五色（黄/红/绿/蓝/白）实时识别显示 + HSV 阈值微调 + ROI 框选

复用 core/camera.py 的 CameraThread/list_cameras、ui/video_panel.py 的
VideoPanel（update_frame 支持 boxes 叠加）。只做识别显示（框 + 中心点），
不输出坐标；结果表格记录 时间/颜色/中心坐标，供后续联动预留。
支持鼠标拖拽框选识别范围（ROI）：识别只在该范围内运行，框外不产生检测框。
"""
import threading
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QCheckBox, QComboBox,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
)

from core.camera import CameraThread, list_cameras
from core.color_detector import ColorDetector, DEFAULT_THRESHOLDS
from ui.video_panel import VideoPanel

# 每个颜色的可调参数定义: (参数名, 显示名, 最小值, 最大值)
COLOR_PARAMS = {
    "yellow": [("h_low", "H低", 0, 180), ("h_high", "H高", 0, 180),
               ("s_low", "S低", 0, 255), ("v_low", "V低", 0, 255)],
    "red":    [("h_low", "H低", 0, 180), ("h_high", "H高", 0, 180),
               ("h_low2", "H2低", 0, 180), ("h_high2", "H2高", 0, 180),
               ("s_low", "S低", 0, 255), ("v_low", "V低", 0, 255)],
    "green":  [("h_low", "H低", 0, 180), ("h_high", "H高", 0, 180),
               ("s_low", "S低", 0, 255), ("v_low", "V低", 0, 255)],
    "blue":   [("h_low", "H低", 0, 180), ("h_high", "H高", 0, 180),
               ("s_low", "S低", 0, 255), ("v_low", "V低", 0, 255)],
    # "black":  [("v_high", "V高", 0, 255)],  # 黑：暂不识别，注释保留
    "white":  [("v_low", "V低", 0, 255), ("s_high", "S高", 0, 255)],
}

COLOR_LABEL = {"yellow": "黄", "red": "红", "green": "绿", "blue": "蓝",
               # "black": "黑",  # 黑：暂不识别，注释保留
               "white": "白"}


class ColorPage(QWidget):
    """色块识别页：相机预览 + 四色检测叠加 + HSV 阈值微调 + 结果表"""

    camera_list_ready = pyqtSignal(list)
    camera_opened = pyqtSignal(bool, str, object)
    camera_closed = pyqtSignal()

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cam = None
        self.detector = ColorDetector()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.roi = None           # (x1, y1, x2, y2) 识别范围；None 表示全画面
        self._spins = {}          # color_key -> {param: QSpinBox}
        self._last_centers = {}   # color_key -> (cx, cy) 用于去重记录
        self._table_rows = 0

        self.camera_list_ready.connect(self._on_camera_list_ready)
        self.camera_opened.connect(self._on_camera_opened)
        self.camera_closed.connect(self._on_camera_closed)

        self._build_ui()
        self.combo_camera.addItem(f"相机 {self.camera_index}", self.camera_index)

        # ROI 框选信号（VideoPanel 在 _build_ui 中创建）
        self.video.roi_changed.connect(self._on_roi_changed)

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 顶部工具行
        top = QHBoxLayout()

        grp_cam = QGroupBox("相机")
        lay_cam = QHBoxLayout(grp_cam)
        self.combo_camera = QComboBox()
        lay_cam.addWidget(self.combo_camera, stretch=1)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self._refresh_cameras)
        lay_cam.addWidget(self.btn_refresh)
        self.btn_open = QPushButton("打开相机")
        self.btn_close = QPushButton("关闭")
        self.btn_close.setEnabled(False)
        self.btn_open.clicked.connect(self.open_camera)
        self.btn_close.clicked.connect(self.close_camera)
        lay_cam.addWidget(self.btn_open)
        lay_cam.addWidget(self.btn_close)
        top.addWidget(grp_cam)

        grp_det = QGroupBox("识别")
        lay_det = QHBoxLayout(grp_det)
        self.chk_detect = QCheckBox("实时识别")
        self.chk_detect.setChecked(True)
        lay_det.addWidget(self.chk_detect)
        self.btn_reset = QPushButton("恢复默认阈值")
        self.btn_reset.clicked.connect(self._reset_thresholds)
        lay_det.addWidget(self.btn_reset)
        top.addWidget(grp_det)

        grp_roi = QGroupBox("识别范围")
        lay_roi = QHBoxLayout(grp_roi)
        self.btn_roi_mode = QPushButton("框选识别范围")
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.toggled.connect(self._toggle_roi_mode)
        lay_roi.addWidget(self.btn_roi_mode)
        self.btn_roi_clear = QPushButton("清除框选")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        lay_roi.addWidget(self.btn_roi_clear)
        top.addWidget(grp_roi)

        root.addLayout(top)

        # 中间：预览 + 阈值 + 结果表
        mid = QHBoxLayout()
        self.video = VideoPanel(640, 480)
        mid.addWidget(self.video, stretch=3)

        right = QVBoxLayout()

        # HSV 阈值微调
        grp_thr = QGroupBox("HSV 阈值微调")
        lay_thr = QVBoxLayout(grp_thr)
        for key, params in COLOR_PARAMS.items():
            sub = QGroupBox(COLOR_LABEL[key])
            grid = QGridLayout(sub)
            self._spins[key] = {}
            for i, (param, label, lo, hi) in enumerate(params):
                grid.addWidget(QLabel(label), i // 2, (i % 2) * 3)
                spin = QSpinBox()
                spin.setRange(lo, hi)
                spin.setValue(DEFAULT_THRESHOLDS[key][param])
                spin.valueChanged.connect(
                    lambda v, k=key, p=param: self.detector.update_threshold(k, **{p: v})
                )
                grid.addWidget(spin, i // 2, (i % 2) * 3 + 1)
                self._spins[key][param] = spin
            lay_thr.addWidget(sub)
        right.addWidget(grp_thr)

        # 结果表格
        grp_tab = QGroupBox("识别结果")
        lay_tab = QVBoxLayout(grp_tab)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["时间", "颜色", "中心坐标"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay_tab.addWidget(self.table)
        row_btn = QHBoxLayout()
        self.btn_clear = QPushButton("清空记录")
        self.btn_clear.clicked.connect(self._clear_table)
        row_btn.addWidget(self.btn_clear)
        lay_tab.addLayout(row_btn)
        right.addWidget(grp_tab, stretch=1)

        mid.addLayout(right, stretch=2)
        root.addLayout(mid, stretch=1)

        self.lbl_status = QLabel("就绪")
        root.addWidget(self.lbl_status)

    def _reset_thresholds(self):
        for key, params in COLOR_PARAMS.items():
            for param, _label, _lo, _hi in params:
                spin = self._spins[key][param]
                spin.blockSignals(True)
                spin.setValue(DEFAULT_THRESHOLDS[key][param])
                spin.blockSignals(False)
                self.detector.update_threshold(key, **{param: spin.value()})

    # ---------- 相机 ----------
    def _refresh_cameras(self):
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.setText("刷新中...")
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self):
        self.camera_list_ready.emit(list_cameras())

    def _on_camera_list_ready(self, devices):
        self.btn_refresh.setEnabled(True)
        self.btn_refresh.setText("刷新")
        self.combo_camera.clear()
        if not devices:
            self.combo_camera.addItem("未检测到相机", -1)
            return
        for idx, name in devices:
            self.combo_camera.addItem(name, idx)
        index = self.combo_camera.findData(self.camera_index)
        if index >= 0:
            self.combo_camera.setCurrentIndex(index)

    def open_camera(self):
        cam_idx = self.combo_camera.currentData()
        if cam_idx is None or cam_idx < 0:
            QMessageBox.warning(self, "提示", "请先刷新并选择可用相机")
            return
        self.btn_open.setEnabled(False)
        self.btn_open.setText("打开中...")
        threading.Thread(target=self._open_worker, args=(cam_idx,), daemon=True).start()

    def _open_worker(self, cam_idx):
        cam = CameraThread(cam_idx)
        ok = cam.open()
        self.camera_opened.emit(ok, str(cam_idx), cam)

    def _on_camera_opened(self, ok, cam_idx, cam):
        self.btn_open.setEnabled(True)
        self.btn_open.setText("打开相机")
        if not ok:
            QMessageBox.warning(self, "提示", f"无法打开相机 {cam_idx}")
            return
        self.cam = cam
        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)
        self.timer.start(33)

    def close_camera(self):
        self.timer.stop()
        if self.cam is not None:
            threading.Thread(target=self._close_worker, daemon=True).start()
        else:
            self._on_camera_closed()

    def _close_worker(self):
        if self.cam is not None:
            self.cam.stop()
        self.camera_closed.emit()

    def _on_camera_closed(self):
        self.cam = None
        self.btn_open.setEnabled(True)
        self.btn_close.setEnabled(False)

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cam is not None:
            self.cam.stop()
            self.cam = None
        self.btn_open.setEnabled(True)
        self.btn_open.setText("打开相机")
        self.btn_close.setEnabled(False)

    # ---------- 检测 ----------
    def _on_tick(self):
        if self.cam is None:
            return
        frame = self.cam.get_frame()
        if frame is None:
            return
        if not self.chk_detect.isChecked():
            self.video.update_frame(frame)
            return
        dets = self.detector.detect(frame, roi=self.roi)
        # 叠加检测框（颜色名，无置信度）
        boxes = [(d.box[0], d.box[1], d.box[2], d.box[3], d.color_name, None)
                 for d in dets]
        self.video.update_frame(frame, boxes=boxes)
        self._record_detections(dets)
        if dets:
            self.lbl_status.setText(f"检测到 {len(dets)} 个色块")
        else:
            self.lbl_status.setText("识别中...")

    def _record_detections(self, dets):
        """记录识别结果到表格（按颜色去重，中心点变化超过 8px 才追加）"""
        seen = set()
        for d in dets:
            seen.add(d.color_key)
            last = self._last_centers.get(d.color_key)
            if last is not None:
                dist = ((d.center[0] - last[0]) ** 2 + (d.center[1] - last[1]) ** 2) ** 0.5
                if dist < 8.0:
                    continue
            self._last_centers[d.color_key] = d.center
            ts = datetime.now().strftime("%H:%M:%S")
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(ts))
            self.table.setItem(row, 1, QTableWidgetItem(d.color_name))
            self.table.setItem(row, 2,
                               QTableWidgetItem(f"({d.center[0]:.1f}, {d.center[1]:.1f})"))
            self.table.scrollToBottom()
            self._table_rows += 1
        # 清理未再出现的颜色记录
        for k in list(self._last_centers.keys()):
            if k not in seen:
                del self._last_centers[k]

    def _clear_table(self):
        self.table.setRowCount(0)
        self._table_rows = 0
        self._last_centers.clear()

    # ---------- ROI 框选 ----------
    def _toggle_roi_mode(self, checked):
        """切换拖拽框选模式"""
        self.video.set_roi_mode(checked)
        if checked:
            self.lbl_status.setText("请在画面上左键拖拽框选识别范围")
        else:
            self.lbl_status.setText("就绪")

    def _on_roi_changed(self, roi):
        """框选完成/清除后的回调：记录 ROI 并自动退出框选模式"""
        self.roi = roi
        self.btn_roi_mode.setChecked(False)   # 触发 _toggle_roi_mode(False)
        if roi is None:
            self.lbl_status.setText("已清除识别范围，恢复全画面识别")
        else:
            x1, y1, x2, y2 = [int(v) for v in roi]
            self.lbl_status.setText(f"识别范围: ({x1},{y1})-({x2},{y2})，仅框内识别")

    def _clear_roi(self):
        """清除框选范围，恢复全画面识别"""
        self.roi = None
        self.video.clear_roi()
        self.btn_roi_mode.setChecked(False)
        self.lbl_status.setText("已清除识别范围，恢复全画面识别")

    def cleanup(self):
        self.stop_camera()
