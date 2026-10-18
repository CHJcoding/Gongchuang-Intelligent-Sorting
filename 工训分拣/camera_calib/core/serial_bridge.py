# -*- coding: utf-8 -*-
"""串口收发框架（第一版）

协议细节留空待填。内置模拟数据源模式：用 QTimer 定时依次触发
「动作完成」「已复位」事件，用于无硬件联调；可通过 set_mode("serial")
切换真串口模式（真串口代码留 TODO）。

模拟时序：每 SIM_INTERVAL_MS 触发一次事件，相位偶数发 action_done、
奇数发 reset_done，与分拣状态机配合（非对应状态的事件会被状态机忽略）。
"""
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

SIM_INTERVAL_MS = 1500  # 模拟模式下事件触发间隔（毫秒）


class SerialBridge(QObject):
    """串口收发框架：模拟数据源 / 真串口（TODO）双模式"""

    action_done = pyqtSignal()        # 机械臂动作完成事件
    reset_done = pyqtSignal()         # 机械臂复位完成事件
    data_received = pyqtSignal(bytes)  # 串口收到数据（协议解析留待填充）
    error = pyqtSignal(str)           # 错误/提示

    def __init__(self, mode: str = "simulate", parent=None):
        super().__init__(parent)
        self.mode = mode
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._sim_tick)
        self._sim_phase = 0
        self._serial = None           # TODO(真串口): 串口对象

    def start(self) -> None:
        """启动数据源：simulate 模式启动 QTimer；serial 模式 TODO"""
        if self.mode == "simulate":
            self._sim_phase = 0
            self._sim_timer.start(SIM_INTERVAL_MS)
        else:
            # TODO(真串口): 打开串口、配置波特率、启动读取线程
            # self._serial = serial.Serial(port=..., baudrate=115200, timeout=0.1)
            # 收到协议帧后解析并 emit action_done / reset_done
            self.error.emit("真串口模式尚未实现，请使用 simulate 模式联调")

    def stop(self) -> None:
        """停止数据源"""
        self._sim_timer.stop()
        # TODO(真串口): if self._serial: self._serial.close()

    def set_mode(self, mode: str) -> None:
        """切换模式：simulate（模拟数据源） / serial（真串口，TODO）"""
        self.mode = mode

    def send(self, data: bytes) -> None:
        """向机械臂下发指令（协议细节待填）"""
        if self.mode == "simulate":
            # 模拟模式不下发真实数据，仅作占位
            return
        # TODO(真串口): if self._serial: self._serial.write(data)

    def _sim_tick(self) -> None:
        """模拟时序：动作完成 → 复位完成 → 动作完成 → ..."""
        if self._sim_phase % 2 == 0:
            self.action_done.emit()
        else:
            self.reset_done.emit()
        self._sim_phase += 1
