# -*- coding: utf-8 -*-
"""数据集采集页（PyQt5 版）：相机预览 + 类别切换 + 单张/连拍采集

复用 capture_dataset.py 的 CLASSES / next_seq / save_frame / draw_osd，
与命令行采集脚本共用同一 dataset 输出目录（<项目根>/dataset），
两套入口续号不覆盖、中文路径 imencode 写盘。
相机复用 core/camera.py 的 CameraThread / list_cameras，采集分辨率固定 640x480。
"""
import os
import threading
import time

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QCheckBox, QComboBox,
    QMessageBox, QButtonGroup,
)

from core.camera import CameraThread, list_cameras
from ui.video_panel import VideoPanel

import capture_dataset as cds


class DatasetPage(QWidget):
    """数据集采集页：相机预览 + 7 类几何体采集（单张/连拍）"""

    camera_list_ready = pyqtSignal(list)
    camera_opened = pyqtSignal(bool, str, object)
    camera_closed = pyqtSignal()

    PREVIEW_W = 640
    PREVIEW_H = 480

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cam = None
        self.cur_idx = 0
        self.counts = {}            # cls_name -> 已拍张数（启动时按目录续算）
        self.out_dir = os.path.join(cds.PROJECT_ROOT, "dataset")
        self._last_saved_t = 0.0
        self._last_saved_txt = ""

        self.timer = QTimer(self)          # 预览刷新
        self.timer.timeout.connect(self._on_tick)
        self.burst_timer = QTimer(self)    # 连拍定时存帧
        self.burst_timer.timeout.connect(self._on_burst_tick)

        self.camera_list_ready.connect(self._on_camera_list_ready)
        self.camera_opened.connect(self._on_camera_opened)
        self.camera_closed.connect(self._on_camera_closed)

        self._build_ui()
        self._init_counts()
        self._update_counts_ui()
        self._select_class(0)

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 顶部：相机控制（枚举/切换/打开/关闭）
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
        root.addWidget(grp_cam)

        # 中间：预览 + 类别切换
        mid = QHBoxLayout()
        self.video = VideoPanel(self.PREVIEW_W, self.PREVIEW_H)
        mid.addWidget(self.video, stretch=3)

        grp_cls = QGroupBox("拍摄类别")
        lay_cls = QVBoxLayout(grp_cls)
        self.class_group = QButtonGroup(self)
        self.class_group.setExclusive(True)
        self.class_buttons = {}
        for i, (name, cn) in enumerate(cds.CLASSES):
            btn = QPushButton("")
            btn.setCheckable(True)
            btn.setMinimumHeight(38)
            self.class_group.addButton(btn, i)
            self.class_buttons[name] = btn
            lay_cls.addWidget(btn)
        self.class_group.buttonClicked.connect(self._on_class_clicked)
        mid.addWidget(grp_cls, stretch=1)

        root.addLayout(mid, stretch=1)

        # 底部：拍摄控制
        grp_ctl = QGroupBox("拍摄控制")
        lay_ctl = QHBoxLayout(grp_ctl)
        self.btn_shot = QPushButton("单张拍摄")
        self.btn_shot.clicked.connect(self._take_shot)
        lay_ctl.addWidget(self.btn_shot)
        self.chk_burst = QCheckBox("连拍")
        self.chk_burst.toggled.connect(self._toggle_burst)
        lay_ctl.addWidget(self.chk_burst)
        lay_ctl.addWidget(QLabel("间隔(s)"))
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.05, 5.0)
        self.spin_interval.setSingleStep(0.05)
        self.spin_interval.setDecimals(2)
        self.spin_interval.setValue(0.5)
        lay_ctl.addWidget(self.spin_interval)
        lay_ctl.addStretch(1)
        self.lbl_out = QLabel("输出: " + self.out_dir)
        lay_ctl.addWidget(self.lbl_out)
        root.addWidget(grp_ctl)

        self.lbl_status = QLabel("就绪")
        root.addWidget(self.lbl_status)

    # ---------- 类别 ----------
    def _init_counts(self):
        """各类已拍数 = 目录现有图片数（续拍不覆盖）"""
        self.counts = {name: max(0, cds.next_seq(self.out_dir, name) - 1)
                       for name, _ in cds.CLASSES}

    def _update_counts_ui(self):
        for i, (name, cn) in enumerate(cds.CLASSES):
            self.class_buttons[name].setText(
                "{} ({})".format(cn, self.counts[name]))

    def _select_class(self, idx):
        self.cur_idx = idx
        name = cds.CLASSES[idx][0]
        self.class_buttons[name].setChecked(True)

    def _on_class_clicked(self, btn):
        idx = self.class_group.id(btn)
        self.cur_idx = idx
        name = cds.CLASSES[idx][0]
        self.lbl_status.setText("当前类别：{}（{}）已拍 {}".format(
            name, cds.CLASSES[idx][1], self.counts[name]))

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
        threading.Thread(target=self._open_worker, args=(cam_idx,),
                         daemon=True).start()

    def _open_worker(self, cam_idx):
        cam = CameraThread(cam_idx, self.PREVIEW_W, self.PREVIEW_H)
        ok = cam.open()
        self.camera_opened.emit(ok, str(cam_idx), cam)

    def _on_camera_opened(self, ok, cam_idx, cam):
        self.btn_open.setEnabled(True)
        self.btn_open.setText("打开相机")
        if not ok:
            QMessageBox.warning(self, "提示", "无法打开相机 {}".format(cam_idx))
            return
        self.cam = cam
        self.btn_open.setEnabled(False)
        self.btn_close.setEnabled(True)
        self.timer.start(33)

    def close_camera(self):
        self.timer.stop()
        if self.chk_burst.isChecked():
            self.chk_burst.setChecked(False)
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
        """切换页签/退出时释放相机（与现有页签一致）"""
        if self.timer.isActive():
            self.timer.stop()
        if self.chk_burst.isChecked():
            self.chk_burst.setChecked(False)
        if self.cam is not None:
            self.cam.stop()
            self.cam = None
        self.btn_open.setEnabled(True)
        self.btn_open.setText("打开相机")
        self.btn_close.setEnabled(False)

    # ---------- 采集 ----------
    def _take_shot(self):
        """拍摄一帧并保存到当前类别目录（单张/连拍共用）"""
        if self.cam is None:
            self.lbl_status.setText("请先打开相机")
            return
        frame = self.cam.get_frame()
        if frame is None:
            self.lbl_status.setText("暂无画面，跳过本次拍摄")
            return
        cls_name = cds.CLASSES[self.cur_idx][0]
        seq = self.counts[cls_name] + 1
        path = cds.save_frame(self.out_dir, cls_name, seq, frame)
        if path:
            self.counts[cls_name] = seq
            self._update_counts_ui()
            self._last_saved_txt = "Saved: " + os.path.basename(path)
            self.lbl_status.setText("已保存 {}（第 {} 张）-> {}".format(
                cls_name, seq, path))
        else:
            self._last_saved_txt = "Save FAILED!"
            self.lbl_status.setText("保存失败！")
        self._last_saved_t = time.time()

    def _on_burst_tick(self):
        self._take_shot()

    def _toggle_burst(self, checked):
        if checked:
            interval = max(0.05, self.spin_interval.value())
            self.burst_timer.start(int(interval * 1000))
            self.lbl_status.setText(
                "连拍开启（间隔 {:.2f}s），转动/移动物体采集多姿态".format(interval))
        else:
            self.burst_timer.stop()
            self.lbl_status.setText("连拍停止")

    # ---------- 预览 ----------
    def _on_tick(self):
        if self.cam is None:
            return
        frame = self.cam.get_frame()
        if frame is None:
            return
        cls_name = cds.CLASSES[self.cur_idx][0]
        disp = frame.copy()
        recording = self.burst_timer.isActive()
        now = time.time()
        saved_text = self._last_saved_txt if (now - self._last_saved_t) < 0.6 else ""
        hint = "Shot:单张  Burst:连拍"
        cds.draw_osd(disp, cls_name, self.counts[cls_name], recording,
                     hint, saved_text)
        self.video.update_frame(disp)

    def cleanup(self):
        self.stop_camera()
