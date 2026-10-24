# -*- coding: utf-8 -*-
"""手眼标定结果应用：像素坐标 -> 机械臂基坐标(mm)。

矩阵来源优先级（先找到先用）：
    1. 环境变量 GC_HAND_EYE_JSON
    2. config.HAND_EYE_JSON（非空时）
    3. 本文件同目录 hand_eye.json
    4. 当前工作目录 hand_eye.json
    5. /root/vision_ws/hand_eye.json（板端约定位置）
    6. 以上都没有 -> 用 config.HAND_EYE_M 的占位值，CALIBRATED = False

  ⚠ CALIBRATED=False 时算出来的坐标是【假的】，千万别拿去驱动龙门实动。

JSON 格式（由 tools/planar_calib.py 生成，也可以手写）::

    {
      "matrix": [[a, b, c], [d, e, f]],     # 也接受 6 个数的扁平数组
      "mean_residual": 1.23,
      "max_residual": 2.45,
      "n_points": 54,
      "square_mm": 25.0,
      "rot": 0,
      "mirror": "none",
      "calibrated_at": "2026-10-24 15:00:00"
    }

标定现场流程见 rdk/vision_sorter/BOARD_README.md 第九节。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

import config as C

# Windows 控制台默认 GBK，打印 ⚠/✅ 会抛 UnicodeEncodeError；放宽错误处理即可。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_load_warnings = []          # 读取 JSON 失败的原因，留给 warn_if_uncalibrated 一起报


# ============================ 矩阵来源解析 ============================
def candidate_paths():
    """按优先级列出所有可能的标定文件路径（去重保序）。"""
    out = []

    def _add(p):
        p = (p or "").strip()
        if p and p not in out:
            out.append(p)

    _add(os.environ.get("GC_HAND_EYE_JSON", ""))
    _add(getattr(C, "HAND_EYE_JSON", ""))
    _add(os.path.join(_THIS_DIR, "hand_eye.json"))
    _add(os.path.join(os.getcwd(), "hand_eye.json"))
    _add("/root/vision_ws/hand_eye.json")
    return out


def _matrix_from_json(path):
    """读一个 JSON，返回 (2x3 ndarray, 原始 dict)。格式不对抛 ValueError。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("顶层不是对象")
    raw = data.get("matrix")
    if raw is None:
        raise ValueError("缺少 matrix 字段")
    arr = np.array(raw, dtype=np.float64)
    if arr.size != 6:
        raise ValueError("matrix 需要 6 个数，实际 %d 个" % arr.size)
    arr = arr.reshape(2, 3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix 含 nan/inf")
    return arr, data


def _resolve():
    """返回 (matrix, source, meta, calibrated)。"""
    for path in candidate_paths():
        if not os.path.isfile(path):
            continue
        try:
            m, meta = _matrix_from_json(path)
            return m, path, meta, True
        except Exception as e:                       # 坏文件不该让节点起不来
            _load_warnings.append("%s 解析失败(%s: %s)，已跳过"
                                  % (path, type(e).__name__, e))
    return (np.array(C.HAND_EYE_M, dtype=np.float64),
            "config.HAND_EYE_M(占位值)", {}, False)


MATRIX, SOURCE, META, CALIBRATED = _resolve()


def reload():
    """重新按优先级加载矩阵（标定工具写完 JSON 后调用）。"""
    global MATRIX, SOURCE, META, CALIBRATED, _load_warnings
    _load_warnings = []
    MATRIX, SOURCE, META, CALIBRATED = _resolve()
    return MATRIX


def is_calibrated():
    return bool(CALIBRATED)


def describe():
    """一行描述当前矩阵来源，用于启动日志。"""
    tag = "已标定" if CALIBRATED else "⚠ 未标定(占位值)"
    s = "%s | 来源=%s" % (tag, SOURCE)
    if CALIBRATED:
        mean = META.get("mean_residual")
        mx = META.get("max_residual")
        n = META.get("n_points")
        if mean is not None:
            s += " | 残差 平均%.2fmm 最大%.2fmm" % (
                float(mean), float(mx if mx is not None else mean))
        if n:
            s += " | 点数=%s" % n
    return s


def warn_if_uncalibrated(logger=None):
    """启动时提醒"坐标还是假的"。logger 可以是 rclpy logger，也可以是 None（打印）。"""
    def _emit(level, msg):
        if logger is None:
            print(msg)
        else:
            getattr(logger, level)(msg)

    for w in _load_warnings:
        _emit("warn", "[hand_eye] " + w)

    if CALIBRATED:
        _emit("info", "[hand_eye] " + describe())
        return True

    _emit("error",
          "[hand_eye] ⚠⚠ 手眼标定未完成，当前用的是 config.HAND_EYE_M 占位值，"
          "像素→毫米 的结果是假的，禁止用来驱动龙门实动！")
    _emit("error",
          "[hand_eye] 标定：python3 tools/planar_calib.py --image 棋盘格照片.jpg "
          "（详见 BOARD_README.md 第八节）")
    _emit("error",
          "[hand_eye] 试过的路径: %s" % ", ".join(candidate_paths()))
    return False


# ============================ 坐标变换 ============================
def pixel_to_robot(u, v, z_mm=None):
    """u,v: 像素坐标；z_mm: 物料高度，None 用 Z_OFFSET。

    返回 (x_mm, y_mm, z_mm)。
    """
    pt = np.array([float(u), float(v), 1.0], dtype=np.float64)
    xy = MATRIX @ pt
    z = C.Z_OFFSET if z_mm is None else z_mm
    return round(float(xy[0]), 2), round(float(xy[1]), 2), float(z)


# ============================ 求解 / 持久化 ============================
def solve(pixel_pts, robot_pts):
    """最小二乘求 2x3 仿射矩阵。

    pixel_pts: N×2 (u,v)  ;  robot_pts: N×2 (x_mm, y_mm)
    返回 (M, residuals, mean, max)

    ⚠ 2x3 有 6 个未知数：N=3 时残差【恒为 0】，看不出好坏；
      至少 4 个点才有误差读数，工程上建议 6~9 个以上。
    """
    P = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    R = np.asarray(robot_pts, dtype=np.float64).reshape(-1, 2)
    if len(P) != len(R):
        raise ValueError("像素点 %d 个，机械坐标点 %d 个，数量必须一致"
                         % (len(P), len(R)))
    if len(P) < 3:
        raise ValueError("至少需要 3 组点对才能解 2x3 仿射（建议 ≥6）")

    A = np.hstack([P, np.ones((len(P), 1))])        # N×3
    M, *_ = np.linalg.lstsq(A, R, rcond=None)       # 3×2
    M = M.T                                         # 2×3

    pred = (M @ A.T).T
    residuals = np.sqrt(((pred - R) ** 2).sum(axis=1))
    return M, residuals, float(residuals.mean()), float(residuals.max())


def residual_report(mean, mx, n):
    """把残差翻译成人话 + 是否达标（平均 ≤5mm 且 最大 ≤8mm）。"""
    lines = ["重投影残差: 平均 %.2f mm / 最大 %.2f mm / 点数 %d"
             % (mean, mx, n)]
    if n < 4:
        lines.append("⚠ 点数 < 4，残差恒接近 0，无法说明标定质量")
    if mean <= 5.0 and mx <= 8.0:
        lines.append("✅ 达标（平均 ≤5mm，最大 ≤8mm）")
    else:
        lines.append("❌ 未达标：检查网格是否铺平、角点是否漏检、"
                     "标定平面是否就是物料高度")
    return "\n".join(lines)


def save_json(path, M, pixel_pts=None, robot_pts=None, mean=None,
              mx=None, extra=None):
    """把矩阵写成 hand_eye.json（原子写：先写临时文件再替换）。"""
    data = {
        "matrix": np.asarray(M, dtype=np.float64).reshape(2, 3).tolist(),
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if mean is not None:
        data["mean_residual"] = round(float(mean), 4)
    if mx is not None:
        data["max_residual"] = round(float(mx), 4)
    if pixel_pts is not None:
        data["n_points"] = len(pixel_pts)
    if extra:
        data.update(extra)

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)                    # Windows/Linux 都能原子替换
    return path


# ============================ 点对文件读取 ============================
def load_points(path):
    """读点对文件，返回 (pixel_pts, robot_pts)。

    支持两种格式：
      * CSV  : 每行  u,v,x_mm,y_mm  （可带表头，自动跳过非数字行）
      * JSON : {"points": [[u,v,x,y], ...]}
    """
    ext = os.path.splitext(path)[1].lower()
    rows = []

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points") if isinstance(data, dict) else data
        for p in (pts or []):
            if len(p) >= 4:
                rows.append([float(v) for v in p[:4]])
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for raw in csv.reader(f):
                try:
                    vals = [float(x) for x in raw[:4] if str(x).strip() != ""]
                except ValueError:
                    continue                      # 表头或注释行
                if len(vals) >= 4:
                    rows.append(vals[:4])

    if len(rows) < 3:
        raise ValueError("%s 里只读到 %d 组有效点对，至少 3 组（建议 ≥6）"
                         % (path, len(rows)))
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0:2], arr[:, 2:4]


# ============================ 命令行工具 ============================
def calibrate(points_path=None):
    """从点对文件求解矩阵，打印结果并写 hand_eye.json。

    points_path 为 None 时用 <本目录>/points.csv。
    """
    if points_path is None:
        points_path = os.path.join(_THIS_DIR, "points.csv")
    pixel_pts, robot_pts = load_points(points_path)
    M, _res, mean, mx = solve(pixel_pts, robot_pts)
    print(residual_report(mean, mx, len(pixel_pts)))
    print("矩阵(填进 config.HAND_EYE_M 或保存 JSON)：")
    print("%s" % (M.tolist(),))
    out = os.path.join(_THIS_DIR, "hand_eye.json")
    save_json(out, M, pixel_pts, robot_pts, mean, mx)
    print("已写入 %s" % out)
    return M


def apply_probe(dx, dy, path=None):
    """实抓偏差补偿：把 (dx, dy) 加到矩阵第 3 列（平移项），写回 JSON。

    用法场景：矩阵规模/方向都对，但抓取落点整体偏了 —— 量出偏差直接补。
    第 3 列就是原点平移，改它不影响比例和方向。
    """
    if path is None:
        for p in candidate_paths():
            if os.path.isfile(p):
                path = p
                break
    if path and os.path.isfile(path):
        M, meta = _matrix_from_json(path)
    else:
        M = np.array(C.HAND_EYE_M, dtype=np.float64)
        meta = {}
        path = os.path.join(_THIS_DIR, "hand_eye.json")

    M = M.copy()
    M[0, 2] += float(dx)
    M[1, 2] += float(dy)
    extra = {k: meta[k] for k in ("square_mm", "rot", "mirror", "n_points")
             if k in meta}
    extra["probe_mm"] = [round(float(dx), 3), round(float(dy), 3)]
    save_json(path, M, mean=meta.get("mean_residual"),
              mx=meta.get("max_residual"), extra=extra)
    print("已把偏差 (%.2f, %.2f) mm 补偿进平移项，写入 %s" % (dx, dy, path))
    print("新矩阵: %s" % (M.tolist(),))
    return M


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="手眼标定：从点对文件求 2x3 矩阵，或做实抓偏差补偿")
    ap.add_argument("--points", help="点对文件（CSV: u,v,x_mm,y_mm 或 JSON）")
    ap.add_argument("--probe", nargs=2, type=float, metavar=("DX", "DY"),
                    help="实抓偏差补偿(mm)，加到矩阵平移项")
    ap.add_argument("--out", help="输出 JSON 路径（默认 hand_eye.json）")
    ap.add_argument("--show", action="store_true", help="只显示当前生效的矩阵来源")
    args = ap.parse_args(argv)

    if args.show or (not args.points and not args.probe):
        print("当前状态: " + describe())
        print("尝试过的路径:")
        for p in candidate_paths():
            print("  %s %s" % ("[存在]" if os.path.isfile(p) else "[     ]", p))
        print("矩阵:\n%s" % np.array2string(MATRIX, precision=6))
        if not args.points and not args.probe:
            print("\n用法: --points 点对文件  或  --probe DX DY")
        return 0

    if args.probe:
        apply_probe(args.probe[0], args.probe[1], args.out)
        return 0

    pixel_pts, robot_pts = load_points(args.points)
    M, _res, mean, mx = solve(pixel_pts, robot_pts)
    print(residual_report(mean, mx, len(pixel_pts)))
    print("矩阵: %s" % (M.tolist(),))
    out = args.out or os.path.join(_THIS_DIR, "hand_eye.json")
    save_json(out, M, pixel_pts, robot_pts, mean, mx)
    print("已写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())