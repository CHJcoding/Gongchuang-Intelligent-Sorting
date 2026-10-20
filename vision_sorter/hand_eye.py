# -*- coding: utf-8 -*-
"""手眼标定结果应用：像素坐标 -> 机械臂基坐标(mm)。"""
import numpy as np
import config as C


def pixel_to_robot(u, v, z_mm=None):
    """
    u,v: 像素坐标
    z_mm: 物料高度，None用默认Z_OFFSET
    返回: (x_mm, y_mm, z_mm)
    """
    M = np.array(C.HAND_EYE_M, dtype=np.float32)
    pt = np.array([u, v, 1.0], dtype=np.float32)
    xy = M @ pt
    z = C.Z_OFFSET if z_mm is None else z_mm
    return round(float(xy[0]), 2), round(float(xy[1]), 2), float(z)


# ============ 标定工具：跑一次，把测到的矩阵填回 config.py ============
def calibrate():
    """
    现场标定步骤：
    1. 在托盘上放5-9个已知机械臂坐标的标记点（比如用机械臂走到那点记下(x,y)）
    2. 对每个点，用相机拍图，点出该点的像素(u,v)
    3. 跑这个函数，得到2x3仿射矩阵，打印后填回 config.HAND_EYE_M
    """
    # 示例数据：把下面换成你实测的点
    pixel_pts = np.array([
        [320, 240], [960, 240], [320, 480], [960, 480], [640, 360]
    ], dtype=np.float32)
    robot_pts = np.array([
        [100, 200], [300, 200], [100, 350], [300, 350], [200, 275]
    ], dtype=np.float32)

    # 求 [x;y] = M @ [u;v;1]
    A = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
    M, *_ = np.linalg.lstsq(A, robot_pts, rcond=None)
    M = M.T  # 2x3
    print("把下面这段填到 config.py 的 HAND_EYE_M：")
    print(repr(M.tolist()))

    # 标定误差自检
    pred = (M @ A.T).T
    err = np.sqrt(((pred - robot_pts) ** 2).sum(axis=1))
    print(f"各点重投影误差(mm): {np.round(err, 2)}")
    print(f"平均误差: {err.mean():.2f} mm, 最大: {err.max():.2f} mm")


if __name__ == "__main__":
    calibrate()
