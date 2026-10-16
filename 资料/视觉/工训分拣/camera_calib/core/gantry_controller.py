# -*- coding: utf-8 -*-
"""龙门架 XY 轴步进电机 + 舵机夹爪控制（模拟模式）

参考 core/serial_bridge.py 的风格：先内置模拟数据源（直接改内部坐标并发信号），
真串口协议待定，留 TODO(真串口) 占位；切换真串口后再按协议帧下发指令。

模拟模式下 jog / move_to / home 立即更新 self.x / self.y 并发出
position_changed；夹爪只记录开合状态并发出 gripper_changed。
"""
from PyQt5.QtCore import QObject, pyqtSignal

DEFAULT_JOG_STEP_MM = 10.0  # 默认点动步距（毫米）


class GantryController(QObject):
    """龙门架控制器：XY 轴定位 + 舵机夹爪开合（模拟模式）

    Signals:
        position_changed(float, float): 位置更新（x, y，单位 mm）
        jog_done: 点动完成
        homing_done: 回零完成
        gripper_changed(bool): 夹爪开合状态变化（True=张开）
    """

    position_changed = pyqtSignal(float, float)
    jog_done = pyqtSignal()
    homing_done = pyqtSignal()
    gripper_changed = pyqtSignal(bool)

    def __init__(self, x=0.0, y=0.0, jog_step=DEFAULT_JOG_STEP_MM,
                 limits=None, parent=None):
        super().__init__(parent)
        self.x = float(x)
        self.y = float(y)
        self.jog_step = float(jog_step)
        self.gripper_open = False
        # 机械行程边界 (x_min, x_max, y_min, y_max)；None 表示不限制
        self.limits = limits

    def _apply_limits(self):
        """把当前位置限制在机械行程边界内"""
        if self.limits is None:
            return
        x_min, x_max, y_min, y_max = self.limits
        self.x = min(max(self.x, x_min), x_max)
        self.y = min(max(self.y, y_min), y_max)

    def set_jog_step(self, step):
        """设置点动步距（mm）"""
        self.jog_step = float(step)

    def jog(self, dx, dy):
        """点动：按增量移动 (dx, dy) mm

        模拟模式直接改坐标并发 position_changed / jog_done；
        真串口模式 TODO：按协议下发增量移动指令并等待到位应答。
        """
        self.x += float(dx)
        self.y += float(dy)
        self._apply_limits()
        self.position_changed.emit(self.x, self.y)
        self.jog_done.emit()
        # TODO(真串口): 协议待定。按协议帧发送增量移动指令
        # self._serial.write(...)

    def move_to(self, x, y):
        """绝对定位到 (x, y) mm（模拟模式同步生效）"""
        self.x = float(x)
        self.y = float(y)
        self._apply_limits()
        self.position_changed.emit(self.x, self.y)
        # TODO(真串口): 协议待定。发送绝对定位指令并等待到位事件

    def home(self):
        """回零到 (0, 0)，完成后发 homing_done"""
        self.x = 0.0
        self.y = 0.0
        self.position_changed.emit(self.x, self.y)
        self.homing_done.emit()
        # TODO(真串口): 协议待定。发送回零指令，等待回零完成事件

    def set_gripper(self, open_flag):
        """舵机夹爪开合：True=张开 / False=闭合；模拟模式只记状态"""
        self.gripper_open = bool(open_flag)
        self.gripper_changed.emit(self.gripper_open)
        # TODO(真串口): 协议待定。发送舵机角度指令（角度由 UI 侧角度 SpinBox 提供）
