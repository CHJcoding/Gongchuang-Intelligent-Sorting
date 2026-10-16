# -*- coding: utf-8 -*-
"""颜色形状识别页（PyQt5 版）：方案 A 融合识别（YOLO 出形状框 + 框内 HSV 取色）

- 复制 ui/yolo_page.py 的整体骨架：CameraThread/list_cameras 相机预览、
  VideoPanel 叠加显示、后台模型加载与逐帧推理线程、conf/iou 参数、
  单图检测、stop_camera/cleanup 接口，与既有页签保持一致。
- 在 YOLO 检测框基础上叠加颜色识别：每个框内用 ColorDetector.classify_roi
  做 HSV 主色投票（中心 60% 区域），检测框标签显示"颜色名·形状名"
  （如 红·圆柱），颜色未判出时显示 "-·圆柱"；结果表格记录
  时间/颜色/形状/置信度/中心坐标，颜色未判出时显示 "-"。
"""
import json
import os
import queue
import sys
import threading
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox, QFileDialog,
    QMessageBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
)

from core.camera import CameraThread, list_cameras
from core.color_detector import ColorDetector
from ui.video_panel import VideoPanel

# 颜色中文名映射（表格颜色列显示中文，与 ui/color_page.py 保持一致）
COLOR_LABEL = {"yellow": "黄", "red": "红", "green": "绿", "blue": "蓝",
               "white": "白", "black": "黑"}
# 颜色英文名映射（画面检测框标签使用，避免 cv2.putText 中文问号）
COLOR_LABEL_EN = {"yellow": "YELLOW", "red": "RED", "green": "GREEN",
                  "blue": "BLUE", "white": "WHITE", "black": "BLACK"}

try:
    from ultralytics import YOLO
    _HAS_ULTRA = True
except Exception:
    _HAS_ULTRA = False

# 官方 YOLOv5 仓库候选路径：优先项目内相对路径（把 yolov5-6.2 仓库和项目放在一起
# 整个文件夹分发，对方电脑即可直接使用），保留本机 E 盘旧路径兜底
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # camera_calib/
YOLOV5_REPO_CANDIDATES = [
    os.path.join(_PROJ_ROOT, "yolov5-6.2"),
    os.path.join(_PROJ_ROOT, "vendor", "yolov5-6.2"),
    os.path.join(_PROJ_ROOT, "yolov5-6.2cpy", "yolov5-6.2"),
    # 测试弹窗：临时注释本机 E 盘路径，验证"找不到仓库自动弹窗选择"流程
    # r"E:\yolov5-6.2cpy\yolov5-6.2",
    # r"E:\yolov5-6.2cpoy\yolov5-6.2",
    # r"E:\yolov5-6.2\yolov5-6.2",
]
# 用户手动选择并记住的 YOLOv5 仓库路径配置文件（放在项目内，跟项目一起分发）
_REPO_CFG = os.path.join(_PROJ_ROOT, "yolov5_repo.json")


def _load_repo_config():
    """读取用户手动指定的 YOLOv5 仓库路径；未配置/损坏/目录失效返回 None"""
    try:
        with open(_REPO_CFG, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("repo_path", "")
        return p if p and os.path.isdir(p) else None
    except Exception:
        return None


def _save_repo_config(repo_path):
    """记住用户手动选择的 YOLOv5 仓库路径，下次加载直接使用"""
    try:
        with open(_REPO_CFG, "w", encoding="utf-8") as f:
            json.dump({"repo_path": repo_path}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _repo_candidates():
    """实际候选顺序：用户手动指定的路径优先，其次内置候选"""
    cands = []
    cfg = _load_repo_config()
    if cfg:
        cands.append(cfg)
    cands.extend(YOLOV5_REPO_CANDIDATES)
    return cands

YOLOV5_REQUIREMENTS_HINT = (
    "若为 YOLOv5 官方依赖缺失，请在 YOLOv5 仓库目录下执行：\n"
    "pip install -r requirements.txt"
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
    repo_needed = pyqtSignal()  # 候选仓库路径都无效，请 UI 弹窗让用户手动选择


class _DetectWorker(threading.Thread):
    """后台线程：加载模型 + 逐帧推理 + 框内 HSV 取色，避免阻塞 UI 主线程

    模型与 ColorDetector 只在当前线程内创建与使用（无跨线程访问）；帧队列
    只保留最新一帧，积压的旧帧直接丢弃，保证实时性。颜色判定在本线程内对
    返回框执行（classify_roi 每框做多次掩码统计，必须与推理一起留在后台，
    否则会占满 UI 主线程导致相机画面卡住），主线程只消费 color_key 结果。
    """

    def __init__(self, signals):
        super().__init__(daemon=True)
        self.signals = signals
        self._running = True
        self._lock = threading.Lock()
        self._pending_load = None          # 待加载的 .pt 路径（单槽）
        self._frame_q = queue.Queue(maxsize=1)  # 最新推理帧 (frame, conf, iou, color_on)
        self._model = None
        self._loader = None      # "ultralytics" | "yolov5"
        self._names = None       # YOLOv5 官方模型类别名 dict
        self._device = None      # YOLOv5 官方推理设备
        # 框内 HSV 主色判定器：只在 run 线程内使用（与推理同线程）
        self.color_detector = ColorDetector()

    def request_load(self, path):
        with self._lock:
            self._pending_load = path

    def submit_frame(self, frame, conf, iou, color_on):
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_q.put((frame, conf, iou, color_on))
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
                frame, conf, iou, color_on = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._detect(frame, conf, iou, color_on)

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
        cands = _repo_candidates()
        for cand in cands:
            if os.path.isdir(cand) and os.path.exists(
                    os.path.join(cand, "models", "experimental.py")):
                repo = cand
                break
        if repo is None:
            # 通知主线程弹窗让用户手动选择仓库目录（后台线程不能直接弹窗）
            try:
                self.signals.repo_needed.emit()
            except Exception:
                pass
            raise RuntimeError(
                "未找到官方 YOLOv5 仓库（已尝试: {}）。\n"
                "已弹出选择窗口，请手动选择 yolov5-6.2 仓库文件夹；"
                "或从 https://github.com/ultralytics/yolov5 下载 v6.2 版本，"
                "将 yolov5-6.2 文件夹放到项目根目录（与 camera_calib 同级或内部均可）。"
                .format(" / ".join(cands)))
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

    def _detect(self, frame, conf, iou, color_on):
        if self._model is None:
            return
        if self._loader == "yolov5":
            boxes, stats = self._detect_v5(frame, conf, iou)
        else:
            boxes, stats = self._detect_ultra(frame, conf, iou)
        if boxes is None:  # 推理出错，错误已通过 detect_error 上报
            return
        # 在推理线程内对每个框做 HSV 取色（classify_roi 掩码统计开销大，
        # 必须留在后台；主线程只消费 color_key）。color_on 关闭时 color_key 为 None。
        colored = []
        for x1, y1, x2, y2, shape, confv in boxes:
            color_key = None
            if color_on:
                color_key = self.color_detector.classify_roi(
                    frame, (x1, y1, x2, y2))
            colored.append((x1, y1, x2, y2, shape, confv, color_key))
        self.signals.detect_result.emit(frame, colored, dict(stats))

    def _detect_ultra(self, frame, conf, iou):
        try:
            results = self._model(frame, conf=conf, iou=iou, verbose=False)
        except Exception as e:
            self.signals.detect_error.emit(str(e))
            return None, None
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
            return None, None
        return boxes, dict(stats)

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
            return None, None
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
            return None, None
        return boxes, dict(stats)


class CombinedPage(QWidget):
    """颜色形状识别页：模型选择 + 实时相机/单图检测 + 框内 HSV 主色融合"""

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
        # 框内 HSV 主色判定已移入后台 _DetectWorker 线程，主线程不再持有取色器
        self._last_centers = {}   # (color_key, shape) -> (cx, cy) 表格去重

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        # 后台检测线程（模型加载 + 推理）
        self._worker_signals = _WorkerSignals()
        self._worker_signals.model_loaded.connect(self._on_model_loaded)
        self._worker_signals.detect_result.connect(self._on_detect_result)
        self._worker_signals.detect_error.connect(self._on_detect_error)
        self._worker_signals.repo_needed.connect(self._on_repo_needed)
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

        # 中间：预览 + 检测参数/统计/结果表/离线检测
        mid = QHBoxLayout()
        self.video = VideoPanel(self.PREVIEW_W, self.PREVIEW_H)
        mid.addWidget(self.video, stretch=3)

        grp_side = QVBoxLayout()
        grp_param = QGroupBox("检测参数")
        lay_param = QVBoxLayout(grp_param)
        self.chk_live = QCheckBox("实时检测")
        self.chk_live.setChecked(True)
        lay_param.addWidget(self.chk_live)
        self.chk_color = QCheckBox("颜色识别")
        self.chk_color.setChecked(True)
        lay_param.addWidget(self.chk_color)
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
        grp_side.addWidget(grp_stat)

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

        # 结果表格：时间 / 颜色 / 形状 / 置信度
        grp_tab = QGroupBox("识别结果")
        lay_tab = QVBoxLayout(grp_tab)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["时间", "颜色", "形状", "置信度"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay_tab.addWidget(self.table)
        row_btn = QHBoxLayout()
        self.btn_clear = QPushButton("清空记录")
        self.btn_clear.clicked.connect(self._clear_table)
        row_btn.addWidget(self.btn_clear)
        lay_tab.addLayout(row_btn)
        grp_side.addWidget(grp_tab, stretch=1)

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
        self._model_path = path  # 记住路径，手动指定仓库后自动重试
        self.btn_select_model.setEnabled(False)
        self.btn_select_model.setText("加载中...")
        self.lbl_model_info.setText("正在加载 " + os.path.basename(path) + " ...")
        self._worker.request_load(path)

    def _on_repo_needed(self):
        """内置候选路径都无效：让用户手动选择 YOLOv5 仓库目录，选完记住并重试"""
        chosen = QFileDialog.getExistingDirectory(
            self, "选择 YOLOv5 官方仓库目录（应包含 models/experimental.py）")
        if chosen:
            _save_repo_config(chosen)
            if getattr(self, "_model_path", None):
                self.lbl_model_info.setText("正在重新加载 ...")
                self._worker.request_load(self._model_path)

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
            self._worker.submit_frame(
                img, self.spin_conf.value(), self.spin_iou.value(),
                self.chk_color.isChecked())
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
            # 推理与框内取色在后台线程执行，结果由 _on_detect_result 更新显示
            self._worker.submit_frame(
                frame, self.spin_conf.value(), self.spin_iou.value(),
                self.chk_color.isChecked())
        else:
            self.video.update_frame(frame)

    def _on_detect_result(self, frame, boxes, stats):
        # 图片模式下仅更新图片结果；相机模式正常更新预览
        if self._image_mode:
            if self._img_frame is None:
                return
            disp_frame = self._img_frame
        else:
            if frame is None:
                return
            disp_frame = frame

        # 方案 A 融合：YOLO 出形状框，框内 HSV 主色已在后台推理线程完成取色，
        # 主线程直接用 color_key 生成英文画面标签（RED·YZ）与表格中文颜色名，
        # 不在主线程执行任何 classify_roi 掩码统计，避免占用 UI 主线程。
        # chk_color 关闭时后台传入 color_key 为 None，逻辑与原来一致。
        disp_boxes = []
        records = []
        for x1, y1, x2, y2, shape, conf, color_key in boxes:
            label_en = COLOR_LABEL_EN.get(color_key, "-") if color_key else "-"
            color_name = COLOR_LABEL.get(color_key, "") if color_key else ""
            label = f"{label_en}·{shape}"
            disp_boxes.append((x1, y1, x2, y2, label, conf))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            records.append((color_key, shape, color_name or "-", conf, (cx, cy)))

        self.video.update_frame(disp_frame, boxes=disp_boxes)
        self._show_stats(stats)
        self._record_results(records)

    def _record_results(self, records):
        """记录识别结果到表格（按 颜色+形状 去重，中心点变化超过 25px 才追加）

        中心坐标仅用于内部位移去重判断，不写入表格展示。
        行数上限 500：超出后删除最旧行，避免 QTableWidget 无限增长拖慢 UI。
        """
        seen = set()
        for color_key, shape, color_name, conf, center in records:
            key = (color_key, shape)
            seen.add(key)
            last = self._last_centers.get(key)
            if last is not None:
                dist = ((center[0] - last[0]) ** 2 + (center[1] - last[1]) ** 2) ** 0.5
                if dist < 25.0:
                    continue
            self._last_centers[key] = center
            ts = datetime.now().strftime("%H:%M:%S")
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(ts))
            self.table.setItem(row, 1, QTableWidgetItem(color_name))
            self.table.setItem(row, 2, QTableWidgetItem(shape))
            self.table.setItem(row, 3, QTableWidgetItem(f"{conf:.2f}"))
            self.table.scrollToBottom()
        # 行数上限 500：保留最近 500 行，超出删除最旧行
        while self.table.rowCount() > 500:
            self.table.removeRow(0)
        # 清理未再出现的 颜色+形状 记录
        for k in list(self._last_centers.keys()):
            if k not in seen:
                del self._last_centers[k]

    def _clear_table(self):
        self.table.setRowCount(0)
        self._last_centers.clear()

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
