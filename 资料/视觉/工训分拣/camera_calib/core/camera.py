# -*- coding: utf-8 -*-
"""USB 相机采集线程：独立线程抓帧，通过队列/锁传递最新帧，避免阻塞 UI"""
import queue
import subprocess
import threading

import cv2


def _open_capture(index, width=640, height=480):
    """按后端优先级打开相机：DSHOW 优先 -> 默认(MSMF) 回退

    MSMF 在后台线程打开部分 USB 相机（如 RGB Camera）会永久阻塞，
    而 DSHOW 秒开；故 DSHOW 优先，失败才回退默认。
    """
    if hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap
    cap.release()
    return None


_sys_count_cache = {"ts": 0.0, "val": 0}


def _system_camera_count():
    """通过 Windows 设备管理器获取相机设备数量（不实际打开摄像头）

    结果缓存 30 秒，避免每次刷新都跑一遍 PowerShell（约 2s）拖慢体验。
    """
    import time as _time

    now = _time.time()
    if now - _sys_count_cache["ts"] < 30.0:
        return _sys_count_cache["val"]
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "(Get-PnpDevice -Class Camera,Image -Status OK -ErrorAction SilentlyContinue | Measure-Object).Count",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        n = int(out.stdout.strip())
        n = max(n, 0)
    except Exception:
        n = 0
    _sys_count_cache["ts"] = now
    _sys_count_cache["val"] = n
    return n


def _probe_capture(index, timeout=3.0):
    """带超时探测单个相机编号：独立线程打开，避免 MSMF 卡死 UI

    与 _open_capture 保持同一后端顺序（DSHOW 优先 -> 默认回退），
    避免"枚举不到但实际能打开"的漏检。
    """
    result = []

    def worker():
        try:
            # DSHOW 优先：对 MSMF 会阻塞的 USB 相机可秒开
            if hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap is not None:
                    if cap.isOpened():
                        cap.release()
                        result.append(True)
                        return
                    cap.release()
            cap = cv2.VideoCapture(index)
            if cap is not None:
                if cap.isOpened():
                    cap.release()
                    result.append(True)
                    return
                cap.release()
        except Exception:
            pass
        result.append(False)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return len(result) > 0 and bool(result[0])


# 摄像头枚举互斥锁 + 结果缓存：主窗口启动时 5 个页面并发枚举，锁保证同一时刻
# 只有一条线程真正探测摄像头，其余复用缓存，避免多线程同时打开同一路 USB 相机
# 导致驱动争抢、界面卡死
_enum_lock = threading.Lock()
_enum_cache = {"ts": 0.0, "val": None}


def list_cameras(max_index=10):
    """枚举本机可用相机设备号（后台探测 + 超时保护，DSHOW 优先避免卡死）

    线程安全：并发调用时第一个线程执行完整枚举并将结果缓存 30 秒，
    其余线程直接复用缓存，不重复探测。

    Args:
        max_index: 最大探测编号上限

    Returns:
        [(index, name), ...]：可用相机列表
    """
    import time as _time

    now = _time.time()
    with _enum_lock:
        if _enum_cache["val"] is not None and now - _enum_cache["ts"] < 30.0:
            return list(_enum_cache["val"])

        # 先查系统设备数，据此收缩探测范围；查不到才按 max_index 全探
        sys_count = _system_camera_count()
        if sys_count > 0:
            limit = max(2, min(max_index, sys_count + 2))
        else:
            limit = max_index

        devices = []
        for i in range(limit):
            if _probe_capture(i):
                devices.append((i, f"相机 {i}"))

        _enum_cache["ts"] = now
        _enum_cache["val"] = devices
        return list(devices)


class CameraThread:
    """相机采集线程封装"""

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self._frame = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        """打开相机并启动采集线程"""
        self.cap = _open_capture(self.camera_index, self.width, self.height)
        if self.cap is None:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret = False
            if not ret:
                if not self.running:
                    break
                continue
            with self._lock:
                self._frame = frame
            # 队列只保留最新帧，满了丢旧帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def get_frame(self):
        """获取最新一帧（拷贝，线程安全）"""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        """先释放设备（read 立即失败退出），再等线程，避免卡满超时"""
        self.running = False
        cap = self.cap
        self.cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        with self._lock:
            self._frame = None
