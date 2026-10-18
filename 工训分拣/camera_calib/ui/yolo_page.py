# -*- coding: utf-8 -*-
"""YOLO 检测页（PyQt5 版）：加载本地 .pt 模型，实时相机/单图检测

- 复用 core/camera.py 的 CameraThread / list_cameras 做相机预览，与既有页签
  保持一致的 stop_camera / cleanup 接口，统一受 main_window 的页签切换释放机制管理。
- 复用 ui/video_panel.py 的 VideoPanel 叠加显示检测框 + 类别 + 置信度。
- 模型加载与推理放在独立后台线程执行（CPU 推理耗时较长，避免阻塞 UI），
  结果通过信号回传主线程显示。
- 模型加载优先使用 ultralytics 的 YOLO 类（可加载 YOLOv5 训练权重）；
  缺少依赖时提示 pip 安装 torch(CPU) + ultralytics。
- 相机不可用时仍可加载模型，并对单张图片文件做离线检测。
"""
import os
import queue
import sys
import threading
from collections import Counter

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox, QFileDialog,
    QMessageBox, QCheckBox,
)

from core.camera import CameraThread, list_cameras
from ui.video_panel import VideoPanel

try:
    from ultralytics import YOLO
    _HAS_ULTRA = True
except Exception:
    _HAS_ULTRA = False

# 官方 YOLOv5 仓库候选路径（用户提供的 E:\yolov5-6.2cpy\yolov5-6.2 可能存在笔误，
# 同时兼容实际存在的 cpoy / 常规命名）
YOLOV5_REPO_CANDIDATES = [
    r"E:\yolov5-6.2cpy\yolov5-6.2",
    r"E:\yolov5-6.2cpoy\yolov5-6.2",
    r"E:\yolov5-6.2\yolov5-6.2",
]
YOLOV5_REQUIREMENTS_HINT = (
    "若为 YOLOv5 官方依赖缺失，请先执行：\n"
    "pip install -r E:\\yolov5-6.2cpy\\yolov5-6.2\\requirements.txt"
)


def _imread_unicode(path):
    """cv2.imread 的 Unicode 路径兼容版（中文路径用 np.fromfile + imdecode）"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class _WorkerSignals(QObject):
    """后台检测线程 -> UI 的信号载体（跨线程 emit，AutoConnection 排队到主线程）"""
    model_loaded = pyqtSignal(bool, str, str, str)  # ok, model_name, loader, info(或错误)
    detect_result = pyqtSignal(object, object, object)  # frame, boxes, stats
    detect_error = pyqtSignal(str)


class _DetectWorker(threading.Thread):
    """后台线程：加载模型 + 逐帧推理，避免 CPU 推理阻塞 UI

    模型只在当前线程内创建与使用（无跨线程访问）；帧队列只保留最新一帧，
    积压的旧帧直接丢弃，保证实时性。
    """

    def __init__(self, signals):
        super().__init__(daemon=True)
        self.signals = signals
        self._running = True
        self._lock = threading.Lock()
        self._pending_load = None          # 待加载的 .pt 路径（单槽）
        self._frame_q = queue.Queue(maxsize=1)  # 最新推理帧 (frame, conf, iou)
        self._model = None
        self._loader = None      # "ultralytics" | "yolov5"
        self._names = None       # YOLOv5 官方模型类别名 dict
        self._device = None      # YOLOv5 官方推理设备

    def request_load(self, path):
        with self._lock:
            self._pending_load = path

    def submit_frame(self, frame, conf, iou):
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_q.put((frame, conf, iou))
        except queue.Full:
            pass

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            # 1) 处理模型加载请求
            with self._lock:
                path = self._pending_load
                self._pending_load = None
            if path:
                self._load_model(path)

            # 2) 取最新帧推理
            try:
                frame, conf, iou = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._detect(frame, conf, iou)

    def _load_model(self, path):
        """优先 ultralytics.YOLO；不兼容（YOLOv5 6.2 权重）或加载失败时 fallback 官方 YOLOv5"""
        name = os.path.basename(path)
        ultra_err = ""
        if _HAS_ULTRA:
            try:
                model = YOLO(path)  # 可加载 YOLOv8 / 兼容版 YOLOv5 权重
                names = getattr(model, "names", None)
                n = len(names) if names else 0
                self._model = model
                self._loader = "ultralytics"
                self._names = None
                info = "{} | 类别数 {}".format(name, n)
                if names and n:
                    info += " | " + ", ".join(str(names[i]) for i in range(min(n, 12)))
                self.signals.model_loaded.emit(True, name, "ultralytics", info)
                return
            except Exception as e:
                ultra_err = str(e)
        else:
            ultra_err = "未安装 ultralytics"
        # fallback：官方 YOLOv5 仓库 attempt_load 加载
        try:
            model, names = self._load_v5(path)
            n = len(names) if names else 0
            info = "{} | 类别数 {}".format(name, n)
            if names and n:
                info += " | " + ", ".join(str(names[i]) for i in range(min(n, 12)))
            self.signals.model_loaded.emit(True, name, "yolov5", info)
        except Exception as e2:
            msg = ("ultralytics 加载失败: {}\n\n"
                   "YOLOv5 官方加载失败: {}\n\n"
                   "{}".format(ultra_err or "未知", e2,
                               YOLOV5_REQUIREMENTS_HINT))
            self.signals.model_loaded.emit(False, name, "", msg)

    def _load_v5(self, path):
        """官方 YOLOv5 仓库加载：sys.path 插入仓库目录 + attempt_load 到 CPU"""
        repo = None
        for cand in YOLOV5_REPO_CANDIDATES:
            if os.path.isdir(cand) and os.path.exists(
                    os.path.join(cand, "models", "experimental.py")):
                repo = cand
                break
        if repo is None:
            raise RuntimeError(
                "未找到官方 YOLOv5 仓库（已尝试: {}）".format(
                    " / ".join(YOLOV5_REPO_CANDIDATES)))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import inspect
        import torch
        from models.experimental import attempt_load

        # torch 2.6+ 默认 torch.load(weights_only=True) 会拒绝 YOLOv5 6.2 老权重
        # （权重 pickle 含不在允许列表内的自定义类），这里临时包装 torch.load
        # 强制 weights_only=False 后加载，加载完恢复原函数，避免污染全局。
        _orig_load = torch.load
        _supports_wo = "weights_only" in inspect.signature(torch.load).parameters

        def _load_compat(*args, **kwargs):
            kwargs.pop("weights_only", None)
            if _supports_wo:
                kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _load_compat
        try:
            model = attempt_load(path, device="cpu")  # YOLOv5 6.2 签名: attempt_load(weights, device=None, ...)
        finally:
            torch.load = _orig_load
        model.eval()
        names = getattr(model, "names", None)
        if not names and hasattr(model, "module"):
            names = getattr(model.module, "names", None)
        if not names:
            names = {}
        # 官方 YOLOv5 6.2 的 model.names 可能是 list，统一转 dict 便于按索引查名
        if isinstance(names, (list, tuple)):
            names = {i: str(n) for i, n in enumerate(names)}
        self._model = model
        self._names = names
        self._loader = "yolov5"
        self._device = torch.device("cpu")
        return model, names

    def _detect(self, frame, conf, iou):
        if self._model is None:
            return
        if self._loader == "yolov5":
            self._detect_v5(frame, conf, iou)
        else:
            self._detect_ultra(frame, conf, iou)

    def _detect_ultra(self, frame, conf, iou):
        try:
            results = self._model(frame, conf=conf, iou=iou, verbose=False)
        except Exception as e:
            self.signals.detect_error.emit(str(e))
            return
        boxes = []
        stats = Counter()
        try:
            if len(results):
                r = results[0]
                names = self._model.names or {}
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
                        label = str(names.get(int(cl), str(cl)))
                        boxes.append((float(x1), float(y1), float(x2),
                                      float(y2), label, float(c)))
                        stats[label] += 1
        except Exception as e:
            self.signals.detect_error.emit(str(e))
            return
        self.signals.detect_result.emit(frame, boxes, dict(stats))

    def _detect_v5(self, frame, conf, iou):
        """官方 YOLOv5 detect 流程：letterbox(640) -> BGR2RGB -> 归一化 ->
        推理 -> NMS -> scale_coords 回原图 -> 画框数据"""
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_coords

        import torch
        model = self._model
        names = self._names or {}
        device = self._device
        img0 = frame
        try:
            img = letterbox(img0, 640, stride=32, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB + HWC2CHW
            img = np.ascontiguousarray(img)
            img_t = torch.from_numpy(img).to(device)
            img_t = img_t.float() / 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)
            with torch.no_grad():
                pred = model(img_t)[0]
            pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)
        except Exception as e:
            self.signals.detect_error.emit(str(e))
            return
        boxes = []
        stats = Counter()
        try:
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        img.shape[1:], det[:, :4], img0.shape).round()
                    for *xyxy, confv, cl in reversed(det):
                        x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]),
                                          float(xyxy[2]), float(xyxy[3]))
                        label = str(names.get(int(cl), str(int(cl))))
                        boxes.append((x1, y1, x2, y2, label, float(confv)))
                        stats[label] += 1
        except Exception as e:
            self.signals.detect_error.emit(str(e))
            return
        self.signals.detect_result.emit(frame, boxes, dict(stats))


class YoloPage(QWidget):
    """YOLO 检测页：模型选择 + 实时相机检测 + 离线单图检测"""

    camera_list_ready = pyqtSignal(list)
    camera_opened = pyqtSignal(bool, str, object)
    camera_closed = pyqtSignal()

    PREVIEW_W = 640
    PREVIEW_H = 480

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cam = None
        self._has_model = False
        self._loading = False
        self._image_mode = False
        self._img_frame = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # 后台检测线程（模型加载 + 推理）
        self._worker_signals = _WorkerSignals()
        self._worker_signals.model_loaded.connect(self._on_model_loaded)
        self._worker_signals.detect_result.connect(self._on_detect_result)
        self._worker_signals.detect_error.connect(self._on_detect_error)
        self._worker = _DetectWorker(self._worker_signals)
        self._worker.start()

        self.camera_list_ready.connect(self._on_camera_list_ready)
        self.camera_opened.connect(self._on_camera_opened)
        self.camera_closed.connect(self._on_camera_closed)

        self._build_ui()
        self._refresh_cameras()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 模型区
        grp_model = QGroupBox("模型")
        lay_model = QHBoxLayout(grp_model)
        self.btn_select_model = QPushButton("选择 .pt 模型")
        self.btn_select_model.clicked.connect(self._select_model)
        lay_model.addWidget(self.btn_select_model)
        self.lbl_model_info = QLabel("未加载模型")
        self.lbl_model_info.setWordWrap(True)
        lay_model.addWidget(self.lbl_model_info, stretch=1)
        root.addWidget(grp_model)

        # 相机区
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

        # 中间：预览 + 检测参数/统计
        mid = QHBoxLayout()
        self.video = VideoPanel(self.PREVIEW_W, self.PREVIEW_H)
        mid.addWidget(self.video, stretch=3)

        grp_side = QVBoxLayout()
        grp_param = QGroupBox("检测参数")
        lay_param = QVBoxLayout(grp_param)
        self.chk_live = QCheckBox("实时检测")
        self.chk_live.setChecked(True)
        lay_param.addWidget(self.chk_live)
        lay_param.addWidget(QLabel("置信度阈值"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(0.25)
        lay_param.addWidget(self.spin_conf)
        lay_param.addWidget(QLabel("NMS IoU 阈值"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 0.99)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setValue(0.45)
        lay_param.addWidget(self.spin_iou)
        grp_side.addWidget(grp_param)

        grp_stat = QGroupBox("帧检测统计")
        lay_stat = QVBoxLayout(grp_stat)
        self.lbl_stats = QLabel("无目标")
        self.lbl_stats.setWordWrap(True)
        lay_stat.addWidget(self.lbl_stats)
        grp_side.addWidget(grp_stat, stretch=1)

        grp_img = QGroupBox("离线检测")
        lay_img = QVBoxLayout(grp_img)
        self.btn_select_image = QPushButton("选择图片检测")
        self.btn_select_image.clicked.connect(self._select_image)
        lay_img.addWidget(self.btn_select_image)
        self.btn_clear_image = QPushButton("清除图片")
        self.btn_clear_image.setEnabled(False)
        self.btn_clear_image.clicked.connect(self._clear_image)
        lay_img.addWidget(self.btn_clear_image)
        grp_side.addWidget(grp_img)

        mid.addLayout(grp_side, stretch=1)
        root.addLayout(mid, stretch=1)

        self.lbl_status = QLabel("就绪")
        root.addWidget(self.lbl_status)

    # ---------- 模型 ----------
    def _select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", "", "PyTorch 模型 (*.pt)")
        if not path:
            return
        self._load_model_path(path)

    def _load_model_path(self, path):
        # 不阻塞：即使未装 ultralytics，后台也会尝试官方 YOLOv5 fallback
        self._loading = True
        self.btn_select_model.setEnabled(False)
        self.btn_select_model.setText("加载中...")
        self.lbl_model_info.setText("正在加载 " + os.path.basename(path) + " ...")
        self._worker.request_load(path)

    def _on_model_loaded(self, ok, name, loader, info):
        self._loading = False
        self.btn_select_model.setEnabled(True)
        self.btn_select_model.setText("选择 .pt 模型")
        if ok:
            self._has_model = True
            loader_txt = "ultralytics" if loader == "ultralytics" else "YOLOv5 官方"
            self.lbl_model_info.setText("已加载({}): {}".format(loader_txt, info))
            self.lbl_status.setText("模型就绪({}): {}".format(loader_txt, name))
        else:
            self._has_model = False
            self.lbl_model_info.setText("加载失败: " + info)
            self.lbl_status.setText("模型加载失败")
            QMessageBox.warning(
                self, "模型加载失败",
                info + "\n\n" + YOLOV5_REQUIREMENTS_HINT +
                "\n\n也可尝试安装 ultralytics：\n"
                "pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu && pip install ultralytics")

    def _on_detect_error(self, msg):
        self.lbl_status.setText("检测出错: " + msg)

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
        if self._image_mode:
            self._clear_image()
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
        self.lbl_status.setText("相机已打开，开始检测")

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
        """切换页签/退出时释放相机（与既有页签一致）"""
        if self.timer.isActive():
            self.timer.stop()
        if self.cam is not None:
            self.cam.stop()
            self.cam = None
        self.btn_open.setEnabled(True)
        self.btn_open.setText("打开相机")
        self.btn_close.setEnabled(False)

    # ---------- 离线图片检测 ----------
    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        img = _imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "提示", "无法读取图片: " + path)
            return
        # 进入图片模式：停止相机预览（相机可能未开，此处仅安全停止）
        if self.timer.isActive():
            self.timer.stop()
        self._image_mode = True
        self._img_frame = img
        self.btn_clear_image.setEnabled(True)
        if self._has_model:
            self._worker.submit_frame(img, self.spin_conf.value(), self.spin_iou.value())
            self.lbl_status.setText("图片检测中: " + os.path.basename(path))
        else:
            self.video.update_frame(img)
            self.lbl_stats.setText("无目标")
            self.lbl_status.setText("图片已加载（未加载模型，仅预览）: "
                                    + os.path.basename(path))

    def _clear_image(self):
        self._image_mode = False
        self._img_frame = None
        self.btn_clear_image.setEnabled(False)
        self.video.setPixmap(QPixmap())
        self.lbl_stats.setText("无目标")
        self.lbl_status.setText("就绪")

    # ---------- 预览与推理 ----------
    def _on_tick(self):
        if self.cam is None:
            return
        frame = self.cam.get_frame()
        if frame is None:
            return
        if self._has_model and self.chk_live.isChecked():
            # 推理在后台线程执行，结果由 _on_detect_result 更新显示
            self._worker.submit_frame(frame, self.spin_conf.value(), self.spin_iou.value())
        else:
            self.video.update_frame(frame)

    def _on_detect_result(self, frame, boxes, stats):
        # 图片模式下仅更新图片结果；相机模式正常更新预览
        if self._image_mode:
            if self._img_frame is None:
                return
            self.video.update_frame(self._img_frame, boxes=boxes)
        else:
            if frame is None:
                return
            self.video.update_frame(frame, boxes=boxes)
        self._show_stats(stats)

    def _show_stats(self, stats):
        if not stats:
            self.lbl_stats.setText("无目标")
            return
        lines = ["{}: {}".format(k, v) for k, v in stats.items()]
        self.lbl_stats.setText("目标数 {}\n".format(sum(stats.values()))
                               + "\n".join(lines))

    def cleanup(self):
        self.stop_camera()
        self._worker.stop()
