# -*- coding: utf-8 -*-
"""主窗口：QTabWidget 承载色块识别 + 分拣统计（PyQt5 版）"""
from PyQt5.QtWidgets import QMainWindow, QTabWidget

from core.color_detector import ColorDetector
from core.goods_config import get_default_goods
from core.serial_bridge import SerialBridge
from core.sorter import Sorter
from ui.color_page import ColorPage
from ui.combined_page import CombinedPage
from ui.dataset_page import DatasetPage
from ui.gantry_page import GantryPage
from ui.statistics_page import StatisticsPage
from ui.yolo_page import YoloPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相机标定上位机")

        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.pages = {}
        self._build_pages()

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_pages(self):
        # 分拣状态机与串口框架随主窗口启动（默认模拟模式，无硬件联调）
        self.detector = ColorDetector()
        self.goods = get_default_goods()
        self.sorter = Sorter(self.detector, self.goods)
        self.serial_bridge = SerialBridge(mode="simulate", parent=self)
        self.serial_bridge.action_done.connect(self.sorter.on_action_done)
        self.serial_bridge.reset_done.connect(self.sorter.on_reset_done)
        self.serial_bridge.start()

        self.pages["color"] = ColorPage(camera_index=0)
        self.pages["statistics"] = StatisticsPage(self.sorter, self.goods, serial_bridge=self.serial_bridge)
        self.pages["dataset"] = DatasetPage(camera_index=0)
        self.pages["yolo"] = YoloPage(camera_index=0)
        self.pages["combined"] = CombinedPage(camera_index=0)
        self.pages["gantry"] = GantryPage(camera_index=0)

        self.tabs.addTab(self.pages["color"], "色块识别")
        self.tabs.addTab(self.pages["statistics"], "分拣统计")
        self.tabs.addTab(self.pages["dataset"], "数据采集")
        self.tabs.addTab(self.pages["yolo"], "YOLO检测")
        self.tabs.addTab(self.pages["combined"], "颜色形状识别")
        self.tabs.addTab(self.pages["gantry"], "龙门架标定")

    def _on_tab_changed(self, index):
        # 切换页签时释放所有相机，避免占用
        for page in self.pages.values():
            if hasattr(page, "stop_camera"):
                page.stop_camera()

    def closeEvent(self, event):
        self.serial_bridge.stop()
        for page in self.pages.values():
            if hasattr(page, "cleanup"):
                page.cleanup()
        super().closeEvent(event)
