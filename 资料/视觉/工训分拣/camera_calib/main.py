# -*- coding: utf-8 -*-
"""相机标定上位机入口（PyQt5 版）"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
