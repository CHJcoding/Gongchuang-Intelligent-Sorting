# -*- coding: utf-8 -*-
"""龙门架标定页（PyQt5 版）

- 复用 core/camera.py 的 CameraThread / list_cameras 与 ui/video_panel.py 的
  VideoPanel（相机控制写法参考 ui/color_page.py：下拉 + 刷新 + 打开/关闭）。
- 复用 core/gantry_calib.py 的 GantryCalib（仿射变换求解/换算/JSON 持久化）
  与 core/gantry_controller.py 的 GantryController（模拟模式 XY 轴 + 夹爪）。
- 工作区 ROI 可框选（VideoPanel.roi_changed），标定配对点仅在框内生效。
- 手动标定：点「采集当前点」后，在画面点选标记物位置，记录
  （像素坐标, 当前机械坐标）配对点；>=3 点后点「计算标定」。
- 自动多点标定：设置网格起始坐标/间距/行列数，控制器按网格逐点移动，
  每点暂停让用户点选画面标记物后自动配对，走完全部点自动计算。
- 实现 stop_camera() / cleanup()，与既有页签一致（受 main_window 统一释放管理）。
"""
import os
import threading

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QRadioButton, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView,
)

from core.camera import CameraThread, list_cameras
from core.gantry_calib import GantryCalib
from core.gantry_controller import GantryController, DEFAULT_JOG_STEP_MM
from ui.video_panel import VideoPanel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "gantry_calib.json")

# 点动步距选项（mm）
JOG_STEP_OPTIONS = [10.0, 50.0, 100.0]


def _point_in_roi(px, py, roi):
    """判断点 (px, py) 是否在 ROI 框 (x1, y1, x2, y2) 内"""
    x1, y1, x2, y2 = roi
    return x1 <= px <= x2 and y1 <= py <= y2


class GantryPage(QWidget):
    """龙门架标定页：相机预览 + 电机点动/回零 + 夹爪 + 手/自动标定"""

    camera_list_ready = pyqtSignal(list)
    camera_opened = pyqtSignal(bool, str, object)
    camera_closed = pyqtSignal()

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cam = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # 标定数学与龙门架控制器（模拟模式）
        self.calib = GantryCalib()
        self.controller = GantryController(parent=self)
        self._sync_controller_limits()
        self.controller.position_changed.connect(self._on_position_changed)
        self.controller.gripper_changed.connect(self._on_gripper_changed)

        # 交互状态
        self.roi = None            # 工作区 ROI (x1, y1, x2, y2)；None 全画面
        self._pick_mode = False    # 采集点选模式（手动/自动共用）
        self._test_mode = False    # 测试换算点选模式
        self._auto_running = False  # 自动标定进行中
        self._auto_queue = []      # 剩余网格点 [(x, y), ...]

        self.camera_list_ready.connect(self._on_camera_list_ready)
        self.camera_opened.connect(self._on_camera_opened)
        self.camera_closed.connect(self._on_camera_closed)

        self._build_ui()
        self.combo_camera.addItem(f"相机 {self.camera_index}", self.camera_index)

        # VideoPanel 信号：ROI 框选 + 画面点选（在 _build_ui 中创建）
        self.video.roi_changed.connect(self._on_roi_changed)
        self.video.point_clicked.connect(self._on_point_clicked)

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 顶部工具行：相机 / 电机控制 / 工作区 ROI
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

        grp_motor = QGroupBox("电机控制（模拟）")
        lay_motor = QGridLayout(grp_motor)
        self.btn_x_minus = QPushButton("X-")
        self.btn_x_plus = QPushButton("X+")
        self.btn_y_minus = QPushButton("Y-")
        self.btn_y_plus = QPushButton("Y+")
        self.btn_x_minus.clicked.connect(lambda: self._jog(-1.0, 0.0))
        self.btn_x_plus.clicked.connect(lambda: self._jog(1.0, 0.0))
        self.btn_y_minus.clicked.connect(lambda: self._jog(0.0, -1.0))
        self.btn_y_plus.clicked.connect(lambda: self._jog(0.0, 1.0))
        lay_motor.addWidget(self.btn_x_minus, 0, 0)
        lay_motor.addWidget(self.btn_x_plus, 0, 1)
        lay_motor.addWidget(self.btn_y_minus, 1, 0)
        lay_motor.addWidget(self.btn_y_plus, 1, 1)

        lay_motor.addWidget(QLabel("步距"), 0, 2)
        self.combo_step = QComboBox()
        for step in JOG_STEP_OPTIONS:
            self.combo_step.addItem(f"{step:g} mm", step)
        self.combo_step.setCurrentIndex(0)
        self.combo_step.currentIndexChanged.connect(self._on_step_changed)
        lay_motor.addWidget(self.combo_step, 0, 3)

        self.btn_home = QPushButton("回零")
        self.btn_home.clicked.connect(self._home)
        lay_motor.addWidget(self.btn_home, 1, 2, 1, 2)

        self.lbl_position = QLabel("X: 0.0 mm    Y: 0.0 mm")
        lay_motor.addWidget(self.lbl_position, 2, 0, 1, 2)

        self.btn_gripper = QPushButton("夹爪张开")
        self.btn_gripper.setCheckable(True)
        self.btn_gripper.clicked.connect(self._toggle_gripper)
        lay_motor.addWidget(self.btn_gripper, 2, 2)
        lay_motor.addWidget(QLabel("角度"), 2, 3)
        self.spin_gripper_angle = QSpinBox()
        self.spin_gripper_angle.setRange(0, 180)
        self.spin_gripper_angle.setValue(90)
        self.spin_gripper_angle.setSuffix("°")
        lay_motor.addWidget(self.spin_gripper_angle, 2, 4)
        top.addWidget(grp_motor)

        grp_roi = QGroupBox("工作区 ROI")
        lay_roi = QVBoxLayout(grp_roi)
        self.btn_roi_mode = QPushButton("框选工作区")
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.toggled.connect(self._toggle_roi_mode)
        lay_roi.addWidget(self.btn_roi_mode)
        self.btn_roi_clear = QPushButton("清除框选")
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        lay_roi.addWidget(self.btn_roi_clear)
        top.addWidget(grp_roi)

        root.addLayout(top)

        # 中间：预览 + 标定控制
        mid = QHBoxLayout()
        self.video = VideoPanel(640, 480)
        mid.addWidget(self.video, stretch=3)

        right = QVBoxLayout()

        # 标定模式切换
        grp_mode = QGroupBox("标定模式")
        lay_mode = QHBoxLayout(grp_mode)
        self.radio_manual = QRadioButton("手动标定")
        self.radio_auto = QRadioButton("自动多点标定")
        self.radio_manual.setChecked(True)
        self.radio_manual.toggled.connect(self._on_mode_changed)
        lay_mode.addWidget(self.radio_manual)
        lay_mode.addWidget(self.radio_auto)
        right.addWidget(grp_mode)

        # 手动标定
        grp_manual = QGroupBox("手动标定")
        lay_manual = QHBoxLayout(grp_manual)
        self.btn_collect = QPushButton("采集当前点")
        self.btn_collect.clicked.connect(self._start_manual_pick)
        lay_manual.addWidget(self.btn_collect)
        self.btn_calc = QPushButton("计算标定")
        self.btn_calc.clicked.connect(self._calc_calib)
        lay_manual.addWidget(self.btn_calc)
        self.btn_clear_pts = QPushButton("清空配对点")
        self.btn_clear_pts.clicked.connect(self._clear_points)
        lay_manual.addWidget(self.btn_clear_pts)
        right.addWidget(grp_manual)

        # 自动标定
        grp_auto = QGroupBox("自动多点标定（网格）")
        lay_auto = QGridLayout(grp_auto)
        lay_auto.addWidget(QLabel("起始 X"), 0, 0)
        self.spin_auto_x0 = QDoubleSpinBox()
        self.spin_auto_x0.setRange(0.0, 5000.0)
        self.spin_auto_x0.setDecimals(1)
        self.spin_auto_x0.setValue(0.0)
        self.spin_auto_x0.setSuffix(" mm")
        lay_auto.addWidget(self.spin_auto_x0, 0, 1)
        lay_auto.addWidget(QLabel("起始 Y"), 0, 2)
        self.spin_auto_y0 = QDoubleSpinBox()
        self.spin_auto_y0.setRange(0.0, 5000.0)
        self.spin_auto_y0.setDecimals(1)
        self.spin_auto_y0.setValue(0.0)
        self.spin_auto_y0.setSuffix(" mm")
        lay_auto.addWidget(self.spin_auto_y0, 0, 3)

        lay_auto.addWidget(QLabel("间距 X"), 1, 0)
        self.spin_auto_dx = QDoubleSpinBox()
        self.spin_auto_dx.setRange(1.0, 1000.0)
        self.spin_auto_dx.setDecimals(1)
        self.spin_auto_dx.setValue(50.0)
        self.spin_auto_dx.setSuffix(" mm")
        lay_auto.addWidget(self.spin_auto_dx, 1, 1)
        lay_auto.addWidget(QLabel("间距 Y"), 1, 2)
        self.spin_auto_dy = QDoubleSpinBox()
        self.spin_auto_dy.setRange(1.0, 1000.0)
        self.spin_auto_dy.setDecimals(1)
        self.spin_auto_dy.setValue(50.0)
        self.spin_auto_dy.setSuffix(" mm")
        lay_auto.addWidget(self.spin_auto_dy, 1, 3)

        lay_auto.addWidget(QLabel("行数"), 2, 0)
        self.spin_auto_rows = QSpinBox()
        self.spin_auto_rows.setRange(1, 20)
        self.spin_auto_rows.setValue(3)
        lay_auto.addWidget(self.spin_auto_rows, 2, 1)
        lay_auto.addWidget(QLabel("列数"), 2, 2)
        self.spin_auto_cols = QSpinBox()
        self.spin_auto_cols.setRange(1, 20)
        self.spin_auto_cols.setValue(3)
        lay_auto.addWidget(self.spin_auto_cols, 2, 3)

        self.btn_auto_start = QPushButton("开始自动标定")
        self.btn_auto_start.clicked.connect(self._start_auto)
        lay_auto.addWidget(self.btn_auto_start, 3, 0, 1, 2)
        self.btn_auto_stop = QPushButton("停止")
        self.btn_auto_stop.setEnabled(False)
        self.btn_auto_stop.clicked.connect(self._stop_auto)
        lay_auto.addWidget(self.btn_auto_stop, 3, 2, 1, 2)
        right.addWidget(grp_auto)

        # 配对点表格
        grp_tab = QGroupBox("配对点（像素坐标 / 机械坐标）")
        lay_tab = QVBoxLayout(grp_tab)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["序号", "像素坐标", "机械坐标", "残差(mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay_tab.addWidget(self.table)
        self.lbl_mean_residual = QLabel("平均残差: -")
        lay_tab.addWidget(self.lbl_mean_residual)
        right.addWidget(grp_tab, stretch=1)

        # 保存/加载
        grp_io = QGroupBox("标定参数")
        lay_io = QHBoxLayout(grp_io)
        self.btn_save = QPushButton("保存参数")
        self.btn_save.clicked.connect(self._save_calib)
        lay_io.addWidget(self.btn_save)
        self.btn_load = QPushButton("加载参数")
        self.btn_load.clicked.connect(self._load_calib)
        lay_io.addWidget(self.btn_load)
        right.addWidget(grp_io)

        # 标定结果与测试
        grp_res = QGroupBox("标定结果 / 测试换算")
        lay_res = QVBoxLayout(grp_res)
        self.lbl_matrix = QLabel("未标定")
        self.lbl_matrix.setWordWrap(True)
        lay_res.addWidget(self.lbl_matrix)
        row_test = QHBoxLayout()
        row_test.addWidget(QLabel("像素 X"))
        self.spin_test_px = QDoubleSpinBox()
        self.spin_test_px.setRange(-9999.0, 9999.0)
        self.spin_test_px.setDecimals(1)
        row_test.addWidget(self.spin_test_px)
        row_test.addWidget(QLabel("Y"))
        self.spin_test_py = QDoubleSpinBox()
        self.spin_test_py.setRange(-9999.0, 9999.0)
        self.spin_test_py.setDecimals(1)
        row_test.addWidget(self.spin_test_py)
        self.btn_test_conv = QPushButton("换算")
        self.btn_test_conv.clicked.connect(self._test_convert)
        row_test.addWidget(self.btn_test_conv)
        self.btn_test_pick = QPushButton("点画面换算")
        self.btn_test_pick.setCheckable(True)
        self.btn_test_pick.toggled.connect(self._toggle_test_pick)
        row_test.addWidget(self.btn_test_pick)
        lay_res.addLayout(row_test)
        self.lbl_test_result = QLabel("机械坐标: -")
        lay_res.addWidget(self.lbl_test_result)
        right.addWidget(grp_res)

        self.lbl_calib_path = QLabel("")
        right.addWidget(self.lbl_calib_path)

        mid.addLayout(right, stretch=2)
        root.addLayout(mid, stretch=1)

        self.lbl_status = QLabel("就绪")
        root.addWidget(self.lbl_status)

    # ---------- 电机控制 ----------
    def _on_step_changed(self, index):
        if 0 <= index < len(JOG_STEP_OPTIONS):
            self.controller.set_jog_step(JOG_STEP_OPTIONS[index])

    def _jog(self, dx, dy):
        self.controller.jog(dx * self.controller.jog_step,
                            dy * self.controller.jog_step)
        self.lbl_status.setText("点动完成")

    def _home(self):
        self.controller.home()
        self.lbl_status.setText("已回零到 (0, 0)")

    def _on_position_changed(self, x, y):
        self.lbl_position.setText(f"X: {x:.1f} mm    Y: {y:.1f} mm")

    def _toggle_gripper(self):
        # 按钮为 checkable：选中 = 张开
        self.controller.set_gripper(self.btn_gripper.isChecked())

    def _on_gripper_changed(self, open_flag):
        self.btn_gripper.setText("夹爪张开" if open_flag else "夹爪闭合")
        # 角度仅作真串口舵机指令参数预留，模拟模式只展示
        self.lbl_status.setText(
            f"夹爪{'张开' if open_flag else '闭合'}（角度 {self.spin_gripper_angle.value()}°）")

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
        self.lbl_status.setText("相机已打开，可开始标定")

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

    # ---------- 预览 ----------
    def _on_tick(self):
        if self.cam is None:
            return
        frame = self.cam.get_frame()
        if frame is None:
            return
        self.video.update_frame(frame)

    # ---------- 工作区 ROI ----------
    def _toggle_roi_mode(self, checked):
        # 框选与点选互斥
        self._exit_pick_mode()
        self._exit_test_mode()
        self.video.set_roi_mode(checked)
        if checked:
            self.lbl_status.setText("请在画面左键拖拽框选工作区 ROI")
        else:
            self.lbl_status.setText("就绪")

    def _on_roi_changed(self, roi):
        self.roi = roi
        self.btn_roi_mode.setChecked(False)
        if roi is None:
            self.lbl_status.setText("已清除工作区 ROI，恢复全画面")
        else:
            x1, y1, x2, y2 = [int(v) for v in roi]
            self.lbl_status.setText(
                f"工作区 ROI: ({x1},{y1})-({x2},{y2})，标定仅在框内生效")

    def _clear_roi(self):
        self.roi = None
        self.video.clear_roi()
        self.btn_roi_mode.setChecked(False)
        self.lbl_status.setText("已清除工作区 ROI，恢复全画面")

    # ---------- 画面点选 ----------
    def _exit_pick_mode(self):
        self._pick_mode = False
        self.video.set_point_mode(False)

    def _exit_test_mode(self):
        self._test_mode = False
        self.btn_test_pick.setChecked(False)
        self.video.set_point_mode(False)

    def _on_point_clicked(self, px, py):
        if self._test_mode:
            self._exit_test_mode()
            self._show_test_result(px, py)
            return
        if not self._pick_mode:
            return
        if self.roi is not None and not _point_in_roi(px, py, self.roi):
            self.lbl_status.setText("点不在工作区 ROI 内，请重新点选")
            return
        # 记录配对点（像素坐标, 当前机械坐标）
        self.calib.add_point(px, py, self.controller.x, self.controller.y)
        self.video.clear_points()
        self._refresh_table()
        if self._auto_running:
            self.lbl_status.setText(
                f"已采集点（机械 {self.controller.x:.1f}, {self.controller.y:.1f}），移动到下一点...")
            self._auto_step()
        else:
            self._exit_pick_mode()
            self.lbl_status.setText(
                f"已采集第 {len(self.calib.points)} 个配对点（机械 "
                f"{self.controller.x:.1f}, {self.controller.y:.1f}）")

    # ---------- 手动标定 ----------
    def _start_manual_pick(self):
        if self.cam is None:
            QMessageBox.warning(self, "提示", "请先打开相机")
            return
        if self._auto_running:
            self._stop_auto()
        self._exit_test_mode()
        self.video.clear_points()
        self.video.set_point_mode(True, 1)
        self._pick_mode = True
        self.lbl_status.setText(
            f"请在画面点选标记物位置（当前机械坐标 "
            f"{self.controller.x:.1f}, {self.controller.y:.1f}）")

    def _calc_calib(self):
        n = len(self.calib.points)
        if n < 3:
            QMessageBox.warning(self, "提示", f"至少需要 3 个配对点，当前 {n} 个")
            return
        ok, _M, residuals, mean = self.calib.solve()
        if not ok:
            QMessageBox.warning(self, "提示", "标定求解失败")
            return
        self._show_matrix()
        self._refresh_table(residuals=residuals)
        self.lbl_mean_residual.setText(f"平均残差: {mean:.3f} mm")
        self.lbl_status.setText(f"标定完成，平均残差 {mean:.3f} mm")

    # ---------- 自动标定 ----------
    def _on_mode_changed(self):
        # 切换模式时停止自动流程，保持状态一致
        if not self.radio_auto.isChecked():
            self._stop_auto()

    def _start_auto(self):
        if self.cam is None:
            QMessageBox.warning(self, "提示", "请先打开相机")
            return
        if self._auto_running:
            return
        x0 = self.spin_auto_x0.value()
        y0 = self.spin_auto_y0.value()
        dx = self.spin_auto_dx.value()
        dy = self.spin_auto_dy.value()
        rows = self.spin_auto_rows.value()
        cols = self.spin_auto_cols.value()
        queue = []
        for r in range(rows):
            for c in range(cols):
                queue.append((x0 + c * dx, y0 + r * dy))
        if not queue:
            return
        if QMessageBox.question(
                self, "确认",
                f"开始自动标定将清空现有配对点（{len(self.calib.points)} 个），"
                f"并依次移动到 {len(queue)} 个网格点采集，是否继续？") != QMessageBox.Yes:
            return
        self._exit_test_mode()
        self._exit_pick_mode()
        self.calib.clear()
        self._refresh_table()
        self.lbl_mean_residual.setText("平均残差: -")
        self.lbl_matrix.setText("标定中...")
        self.lbl_test_result.setText("机械坐标: -")
        self._auto_queue = queue
        self._auto_running = True
        self.btn_auto_start.setEnabled(False)
        self.btn_auto_stop.setEnabled(True)
        self.btn_collect.setEnabled(False)
        self.btn_calc.setEnabled(False)
        self._auto_step()

    def _auto_step(self):
        if not self._auto_running or not self._auto_queue:
            self._auto_finish()
            return
        x, y = self._auto_queue.pop(0)
        self.controller.move_to(x, y)
        self.video.clear_points()
        self.video.set_point_mode(True, 1)
        self._pick_mode = True
        self.lbl_status.setText(
            f"自动标定: 已移动到 ({x:.1f}, {y:.1f}) mm，请在画面点选标记物位置"
            f"（剩余 {len(self._auto_queue)} 点）")

    def _auto_finish(self):
        self._auto_running = False
        self._auto_queue = []
        self._pick_mode = False
        self.video.set_point_mode(False)
        self.btn_auto_start.setEnabled(True)
        self.btn_auto_stop.setEnabled(False)
        self.btn_collect.setEnabled(True)
        self.btn_calc.setEnabled(True)
        n = len(self.calib.points)
        if n < 3:
            self.lbl_status.setText(f"自动标定结束：仅采集 {n} 个点，至少 3 点才可计算")
            return
        ok, _M, residuals, mean = self.calib.solve()
        if not ok:
            self.lbl_status.setText("自动标定结束，但求解失败")
            return
        self._show_matrix()
        self._refresh_table(residuals=residuals)
        self.lbl_mean_residual.setText(f"平均残差: {mean:.3f} mm")
        self.lbl_status.setText(f"自动标定完成，共 {n} 点，平均残差 {mean:.3f} mm")

    def _stop_auto(self):
        was_running = self._auto_running
        self._auto_running = False
        self._auto_queue = []
        self._exit_pick_mode()
        self.btn_auto_start.setEnabled(True)
        self.btn_auto_stop.setEnabled(False)
        self.btn_collect.setEnabled(True)
        self.btn_calc.setEnabled(True)
        if was_running:
            self.lbl_status.setText("已停止自动标定")

    # ---------- 表格 ----------
    def _refresh_table(self, residuals=None):
        pts = self.calib.points
        self.table.setRowCount(len(pts))
        for i, (px, py, mx, my) in enumerate(pts):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"({px:.1f}, {py:.1f})"))
            self.table.setItem(i, 2, QTableWidgetItem(f"({mx:.1f}, {my:.1f})"))
            if residuals is not None and i < len(residuals):
                self.table.setItem(i, 3, QTableWidgetItem(f"{residuals[i]:.3f}"))
            else:
                self.table.setItem(i, 3, QTableWidgetItem("-"))
        self.table.scrollToBottom()

    def _clear_points(self):
        self.calib.clear()
        self._refresh_table()
        self.lbl_mean_residual.setText("平均残差: -")
        self.lbl_matrix.setText("未标定")
        self.lbl_test_result.setText("机械坐标: -")
        self.lbl_status.setText("已清空配对点")

    # ---------- 标定结果与持久化 ----------
    def _show_matrix(self):
        M = self.calib.M
        if M is None:
            self.lbl_matrix.setText("未标定")
            return
        self.lbl_matrix.setText(
            f"a00={M[0, 0]:.6f}  a01={M[0, 1]:.6f}  a02={M[0, 2]:.3f}\n"
            f"a10={M[1, 0]:.6f}  a11={M[1, 1]:.6f}  a12={M[1, 2]:.3f}")

    def _test_convert(self):
        self._show_test_result(self.spin_test_px.value(), self.spin_test_py.value())

    def _show_test_result(self, px, py):
        result = self.calib.pixel_to_machine(px, py)
        if result is None:
            self.lbl_test_result.setText("请先完成标定")
            return
        mx, my = result
        self.lbl_test_result.setText(f"机械坐标: ({mx:.2f}, {my:.2f}) mm")
        self.spin_test_px.setValue(px)
        self.spin_test_py.setValue(py)

    def _toggle_test_pick(self, checked):
        if checked:
            self._exit_pick_mode()
            if self._auto_running:
                self._stop_auto()
            self.video.clear_points()
            self.video.set_point_mode(True, 1)
            self._test_mode = True
            self.lbl_status.setText("点画面任一点，查看换算后的机械坐标")
        else:
            self._exit_test_mode()

    def _save_calib(self):
        if self.calib.M is None:
            QMessageBox.warning(self, "提示", "请先完成标定计算再保存")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存标定参数", DEFAULT_CONFIG_PATH, "JSON (*.json)")
        if not path:
            return
        try:
            self.calib.save_json(path)
        except Exception as e:
            QMessageBox.warning(self, "提示", f"保存失败: {e}")
            return
        self.lbl_calib_path.setText(f"已保存: {path}")
        self.lbl_status.setText("标定参数已保存")

    def _load_calib(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载标定参数", DEFAULT_CONFIG_DIR, "JSON (*.json)")
        if not path:
            return
        if not self.calib.load_json(path):
            QMessageBox.warning(self, "提示", "加载失败或文件格式不正确")
            return
        self._sync_controller_limits()
        self._refresh_table()
        self._show_matrix()
        self.lbl_mean_residual.setText(f"平均残差: {self.calib.mean_residual:.3f} mm")
        self.lbl_calib_path.setText(f"已加载: {path}")
        self.lbl_status.setText("标定参数已加载")

    def _sync_controller_limits(self):
        tl = self.calib.travel_limits
        self.controller.limits = (tl["x_min"], tl["x_max"], tl["y_min"], tl["y_max"])

    # ---------- 清理 ----------
    def cleanup(self):
        self.stop_camera()
        self._stop_auto()
