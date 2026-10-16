# -*- coding: utf-8 -*-
"""分拣统计页：序号 / 货物名称 / 图片 / 分拣成功总数量

- 表格每行一件货物；图片列显示货物配置图片或纯色占位缩略图
- 「清零」按钮手动清零所有计数；计数仅存内存，重启自动清零
- 当前状态显示：空闲/已锁定/等待动作/等待复位/校验中
- 模拟模式提供「模拟一次分拣」按钮：无硬件时演示完整状态流转与计数
"""
import os

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from core.goods_config import GoodsItem, get_image_path
from core.sorter import Sorter, SorterState, STATE_TEXT

COLUMN_HEADERS = ["序号", "货物名称", "图片", "分拣成功总数量"]
THUMB_SIZE = 100         # 图片缩略图尺寸（像素）
SIM_CENTER = (320.0, 240.0)  # 模拟锁定时的虚拟中心像素坐标
AUTO_INTERVAL_MS = 2000  # 一键启动自动分拣间隔（毫秒），模拟模式周期触发一次分拣


class StatisticsPage(QWidget):
    """分拣统计表格页面"""

    def __init__(self, sorter: Sorter, goods, serial_bridge=None, parent=None):
        super().__init__(parent)
        self.sorter = sorter
        self.goods = list(goods)
        self.serial_bridge = serial_bridge   # 可选：真串口模式下用于判断/预留启动接口
        self._sim_index = 0
        self._auto_running = False
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._on_simulate)
        self._build_ui()
        self._refresh_table()
        # 注册状态机回调（运行在 Qt 主线程，直接更新 UI）
        self.sorter.on_state_changed(self._on_state_changed)
        self.sorter.on_count_changed(self._on_count_changed)
        self._update_state_label()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # ========== 顶部：一键启动大按钮 ==========
        self.btn_auto = QPushButton("▶ 一键启动")
        self.btn_auto.setMinimumHeight(56)
        _font = self.btn_auto.font()
        _font.setPointSize(16)
        _font.setBold(True)
        self.btn_auto.setFont(_font)
        self.btn_auto.clicked.connect(self._on_toggle_auto)
        root.addWidget(self.btn_auto)

        # ========== 系统运行状态（独立一行，与单次状态文本区分） ==========
        self.lbl_system_status = QLabel("系统状态：空闲")
        self.lbl_system_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #666666;")
        root.addWidget(self.lbl_system_status)

        # 顶部：状态显示 + 操作按钮
        top = QHBoxLayout()

        grp_status = QGroupBox("当前状态")
        lay_status = QHBoxLayout(grp_status)
        self.lbl_state = QLabel("空闲")
        self.lbl_state.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a73e8;")
        lay_status.addWidget(self.lbl_state)
        top.addWidget(grp_status, stretch=1)

        grp_op = QGroupBox("操作")
        lay_op = QHBoxLayout(grp_op)
        self.btn_sim = QPushButton("模拟一次分拣")
        self.btn_sim.clicked.connect(self._on_simulate)
        lay_op.addWidget(self.btn_sim)
        self.btn_clear = QPushButton("清零")
        self.btn_clear.clicked.connect(self._on_clear)
        lay_op.addWidget(self.btn_clear)
        top.addWidget(grp_op)
        root.addLayout(top)

        # 统计表格
        self.table = QTableWidget(len(self.goods), len(COLUMN_HEADERS))
        self.table.setHorizontalHeaderLabels(COLUMN_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(THUMB_SIZE + 12)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        root.addWidget(self.table, stretch=1)

        self.lbl_hint = QLabel(
            "提示：分拣成功 = 机械臂动作完成 + 复位后视觉校验通过（容差圈内无同色残留）。"
            "模拟模式无需相机，点击「模拟一次分拣」可观察完整状态流转与计数变化。"
        )
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

    # ---------- 表格 ----------
    def _refresh_table(self):
        for row, g in enumerate(self.goods):
            # 序号
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.item(row, 0).setTextAlignment(Qt.AlignCenter)
            # 货物名称
            self.table.setItem(row, 1, QTableWidgetItem(g.name))
            # 图片（配置图或纯色占位缩略图）
            img_path = get_image_path(g)
            item = QTableWidgetItem()
            if os.path.isfile(img_path):
                item.setIcon(QIcon(QPixmap(img_path)))
            self.table.setItem(row, 2, item)
            # 分拣成功总数量
            self.table.setItem(row, 3, QTableWidgetItem(
                str(self.sorter.counts.get(g.goods_id, 0))))
            self.table.item(row, 3).setTextAlignment(Qt.AlignCenter)

    def _on_count_changed(self, goods_id: str, new_count: int):
        """计数变化回调：更新对应行数量列"""
        for row, g in enumerate(self.goods):
            if g.goods_id == goods_id:
                self.table.setItem(row, 3, QTableWidgetItem(str(new_count)))
                self.table.item(row, 3).setTextAlignment(Qt.AlignCenter)
                return

    def _on_state_changed(self, new_state, old_state):
        self._update_state_label()

    def _update_state_label(self):
        self.lbl_state.setText(STATE_TEXT.get(self.sorter.state, "未知"))

    # ---------- 一键启动自动循环 ----------
    def _on_toggle_auto(self):
        """一键启动/停止：切换自动分拣循环"""
        if self._auto_running:
            self._stop_auto()
        else:
            self._start_auto()

    def _start_auto(self):
        """启动自动分拣循环"""
        self._auto_running = True
        self.btn_auto.setText("■ 停止")
        self.lbl_system_status.setText("系统状态：运行中")
        self.lbl_system_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a73e8;")
        if self.serial_bridge is not None and self.serial_bridge.mode != "simulate":
            # TODO(真机): 启动相机识别线程，检测到货物后锁定并下发
            # TODO(真机): 打开串口通道（serial_bridge.start()），接收 action_done / reset_done
            pass
        # 模拟模式：QTimer 周期触发"模拟一次分拣"，先立即触发一轮
        self._auto_timer.start(AUTO_INTERVAL_MS)
        self._on_simulate()

    def _stop_auto(self):
        """停止自动分拣循环（暂停触发，当前已下发的一轮由模拟事件流自然收尾）"""
        self._auto_timer.stop()
        self._auto_running = False
        self.btn_auto.setText("▶ 一键启动")
        self.lbl_system_status.setText("系统状态：空闲")
        self.lbl_system_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #666666;")
        if self.serial_bridge is not None and self.serial_bridge.mode != "simulate":
            # TODO(真机): 停止相机识别线程，关闭串口通道（serial_bridge.stop()）
            pass

    # ---------- 操作 ----------
    def _on_simulate(self):
        """模拟一次分拣：顺序选择一件货物 → 锁定 → 下发，等待模拟事件流完成校验"""
        if self.sorter.state is not SorterState.IDLE:
            # 自动循环中忙碌则静默跳过本轮，等待下个周期；手动点击时给出提示
            if not self._auto_running:
                self.lbl_hint.setText(
                    f"当前状态「{STATE_TEXT.get(self.sorter.state)}」，请等待回到空闲后再模拟"
                )
            return
        g = self.goods[self._sim_index % len(self.goods)]
        self._sim_index += 1
        if self.sorter.lock(g.goods_id, SIM_CENTER):
            self.sorter.dispatch()
            self.lbl_hint.setText(f"已模拟锁定「{g.name}」并下发，等待动作完成/复位事件...")

    def _on_clear(self):
        """清零所有计数"""
        self.sorter.reset_counts()
        self._refresh_table()

    def stop_camera(self):
        """与其它页签接口保持一致；统计页不直接持有相机，无需处理"""
        pass

    def cleanup(self):
        """关闭页签时的清理钩子"""
        pass
