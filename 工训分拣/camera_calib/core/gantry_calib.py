# -*- coding: utf-8 -*-
"""龙门架标定数学（纯 Python，不依赖 UI）

相机俯视工作区（约 15cm 高），通过若干「像素坐标 <-> 机械坐标」配对点
求解仿射变换矩阵 M(2x3)：

    [mx]   [a00 a01 a02] [px]
    [my] = [a10 a11 a12] [py]
                          [1 ]

机械坐标单位为 mm（以龙门架回零位为原点，X/Y 两轴步进电机）。
配对点 >= 3 个时做最小二乘（与 cv2.estimateAffine2D 最小二乘模式等价，
这里直接使用 numpy.linalg.lstsq 便于控制残差计算）；恰好 3 点时用方阵精确解。

标定矩阵、配对点与机械行程边界可持久化到 JSON：
    {"matrix": [...], "points": [...], "travel_limits": {...}, "mean_residual": ...}
"""
import json
import os

import numpy as np

# 默认机械行程边界（mm），加载/保存时与 GantryController.limits 同步
DEFAULT_TRAVEL_LIMITS = {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0, "y_max": 400.0}


class GantryCalib:
    """龙门架手眼标定：像素坐标 <-> 机械坐标 仿射变换"""

    def __init__(self):
        self.points = []          # [(px, py, mx, my), ...]
        self.M = None             # 2x3 numpy 矩阵；未求解时为 None
        self.mean_residual = 0.0  # 平均残差（mm）
        self.travel_limits = dict(DEFAULT_TRAVEL_LIMITS)  # 机械行程边界

    # ---------- 配对点管理 ----------
    def add_point(self, px, py, mx, my):
        """添加一对配对点：像素坐标 (px, py) 与 机械坐标 (mx, my)"""
        self.points.append((float(px), float(py), float(mx), float(my)))

    def clear(self):
        """清空配对点与求解结果"""
        self.points = []
        self.M = None
        self.mean_residual = 0.0

    def set_points(self, points):
        """整体设置配对点列表 [(px, py, mx, my), ...]"""
        self.points = [(float(p[0]), float(p[1]), float(p[2]), float(p[3]))
                       for p in points]
        self.M = None
        self.mean_residual = 0.0

    # ---------- 求解 ----------
    def solve(self):
        """计算仿射变换矩阵（配对点 >= 3 才可求解）

        Returns:
            (ok, M, residuals, mean_residual)
                ok: 是否求解成功（配对点数量不足返回 False）
                M: 2x3 numpy 矩阵（成功时）
                residuals: 各配对点的残差列表（机械坐标 mm）
                mean_residual: 平均残差（mm）
        """
        n = len(self.points)
        if n < 3:
            return False, None, [], 0.0
        pts = np.array(self.points, dtype=np.float64)
        # A: 每行 [px, py, 1]；b: (mx, my)
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(n)])
        b = pts[:, 2:4]
        if n == 3:
            # 恰好 3 点：3 未知数 3 方程，方阵精确解
            coeff = np.linalg.solve(A, b)  # 3x2
        else:
            # 多于 3 点：最小二乘（等价 cv2.estimateAffine2D 的最小二乘模式）
            coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
        M = coeff.T  # 2x3
        # 各点重投影残差（机械坐标 mm）
        pred = A @ coeff
        residuals = [float(np.hypot(pred[i, 0] - pts[i, 2],
                                    pred[i, 1] - pts[i, 3]))
                     for i in range(n)]
        self.M = M
        self.mean_residual = float(np.mean(residuals))
        return True, M, residuals, self.mean_residual

    # ---------- 坐标换算 ----------
    def pixel_to_machine(self, px, py):
        """像素坐标 -> 机械坐标 (mx, my)；未完成标定时返回 None"""
        if self.M is None:
            return None
        v = self.M @ np.array([float(px), float(py), 1.0])
        return float(v[0]), float(v[1])

    # ---------- 持久化 ----------
    def save_json(self, path):
        """保存标定参数到 JSON（矩阵 + 配对点 + 行程边界）

        目录不存在时自动创建；返回写入路径。
        """
        data = {
            "matrix": None if self.M is None else self.M.reshape(-1).tolist(),
            "points": [[float(p[0]), float(p[1]), float(p[2]), float(p[3])]
                       for p in self.points],
            "travel_limits": self.travel_limits,
            "mean_residual": self.mean_residual,
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def load_json(self, path):
        """从 JSON 加载标定参数；文件不存在 / 解析失败返回 False"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        pts = data.get("points", [])
        if pts:
            self.set_points(pts)
        limits = data.get("travel_limits")
        if isinstance(limits, dict):
            self.travel_limits = {
                k: float(limits.get(k, DEFAULT_TRAVEL_LIMITS[k]))
                for k in ("x_min", "x_max", "y_min", "y_max")
            }
        matrix = data.get("matrix")
        if matrix and len(matrix) == 6:
            self.M = np.array(matrix, dtype=np.float64).reshape(2, 3)
        self.mean_residual = float(data.get("mean_residual", 0.0))
        return True
