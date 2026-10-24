# -*- coding: utf-8 -*-
"""平面标定：用一张棋盘格照片求「像素 -> 毫米」的 2x3 仿射矩阵。

为什么用棋盘格
--------------
夹爪/针尖在画面里只有 1~2 像素宽，人眼从屏幕读数误差就有几毫米；
而棋盘格角点是黑白锐利交点，`cornerSubPix` 能到 0.1px 量级，
而且角点的物理坐标是【已知的】（格子边长），所以：

    一次拍摄 -> 54 组点对 -> 直接解出比例和方向
    全程不需要知道龙门坐标，也不需要用眼睛去看夹爪。

典型用法
--------
1) 打印 tools/chessboard_9x6.png（A4，100% 比例，不许缩放）
2) 平铺在物料所在平面，背面贴死或用亚克力压平（翘 2mm 就是几毫米误差）
3) 相机拍一张存成 board.jpg，然后：

    python planar_calib.py --image board.jpg

   棋盘没正放（龙门 X/Y 和经验方向不一致）时再加上对齐参数：

    python planar_calib.py --image board.jpg --rot 0 --mirror y
    python planar_calib.py --image board.jpg --anchor-index 0 --anchor-xy 120.5 60.0

4) 实抓一次看偏差，直接补偿（不用重拍）：

    python planar_calib.py --probe 12 -8

输出 hand_eye.json，`sort_vision_node` 启动时自动优先读取。

自检（不需要相机、不需要照片）
------------------------------
    python planar_calib.py --self-test

会用合成棋盘图验证「检测 + 求解 + 合成」整条链路，并校验已知矩阵能被还原。
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
# 兼容两种布局：仓库里 tools/ + rdk/vision_sorter/；板端 /root/vision_ws/tools + vision_sorter
for _p in (os.path.join(_PARENT, "rdk", "vision_sorter"),
           os.path.join(_PARENT, "vision_sorter"),
           "/root/vision_ws/vision_sorter", _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import hand_eye                                     # noqa: E402  复用求解/写盘逻辑

# Windows 控制台默认 GBK，打印 ✅/⚠ 这类字符会直接抛 UnicodeEncodeError；
# 这里只放宽错误处理（不换编码），中文照常显示，个别符号变 "?"，不会崩。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

DEFAULT_COLS = 9        # 内角点列数
DEFAULT_ROWS = 6        # 内角点行数
DEFAULT_SQUARE = 25.0   # 格子边长 mm
DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), "hand_eye.json")


# ============================ 基础工具 ============================
def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 np.fromfile 绕过去。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_intrinsic(path):
    """读 camera_intrinsic.json，返回 (mtx, dist) 或 (None, None)。"""
    if not path or not os.path.isfile(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mtx = np.array(data["mtx"], dtype=np.float64)
    dist = np.array(data.get("dist", []), dtype=np.float64).reshape(-1)
    return mtx, dist


def board_object_points(cols, rows, square_mm):
    """棋盘格所有内角点的物理坐标（mm，Z=0），与 findChessboardCorners 的顺序一致。"""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] *= float(square_mm)
    return objp[:, :2].astype(np.float64)


# ============================ 角点检测 ============================
def detect_corners(gray, cols, rows):
    """检测棋盘内角点，返回 N×2 像素坐标；失败返回 None。"""
    ok, corners = False, None

    # 优先用 SB 版：更快更稳，带亚像素
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = 0
        for name in ("CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
            flags |= int(getattr(cv2, name, 0))
        try:
            ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
        except cv2.error:
            ok, corners = False, None

    # 回退到经典版 + cornerSubPix
    if not ok:
        flags = 0
        for name in ("CALIB_CB_ADAPTIVE_THRESH", "CALIB_CB_NORMALIZE_IMAGE"):
            flags |= int(getattr(cv2, name, 0))
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)

    if not ok or corners is None:
        return None
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


# ============================ 坐标系对齐 ============================
def align_matrix(rot_deg=0, mirror="none"):
    """棋盘毫米坐标系 -> 龙门毫米坐标系 的 3x3 齐次矩阵。

    rot_deg : 逆时针旋转角度（0/90/180/270 最常用）
    mirror  : none / x / y / xy —— 龙门某根轴跟棋盘方向相反时用
    """
    th = np.deg2rad(float(rot_deg))
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th), np.cos(th)]], dtype=np.float64)
    m = (mirror or "none").lower()
    if m == "x":
        R = R @ np.diag([-1.0, 1.0])
    elif m == "y":
        R = R @ np.diag([1.0, -1.0])
    elif m in ("xy", "yx", "both"):
        R = R @ np.diag([-1.0, -1.0])
    A = np.eye(3, dtype=np.float64)
    A[:2, :2] = R
    return A


def compose(outer3, inner23):
    """先 inner23（像素->棋盘），再 outer3（棋盘->龙门）。"""
    inner = np.vstack([np.asarray(inner23, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (np.asarray(outer3, dtype=np.float64) @ inner)[:2, :]


def shift_translation(M, du, dv):
    """在已解出的矩阵上平移（就是改第 3 列）。"""
    M = np.array(M, dtype=np.float64, copy=True)
    M[0, 2] += float(du)
    M[1, 2] += float(dv)
    return M


# ============================ 主流程 ============================
def calibrate_from_image(image_path, cols=DEFAULT_COLS, rows=DEFAULT_ROWS,
                         square_mm=DEFAULT_SQUARE, intrinsic_path=None,
                         rot=0, mirror="none", anchor=None,
                         anchor_index=None, anchor_xy=None, show=False):
    """从一张棋盘格照片标定。返回 dict（含 matrix / 残差 / 预览信息）。"""
    img = imread_unicode(image_path)
    if img is None:
        raise RuntimeError("读不到图片: %s" % image_path)

    undistorted = False
    mtx, dist = load_intrinsic(intrinsic_path)
    if mtx is not None:
        img = cv2.undistort(img, mtx, dist)
        undistorted = True
    else:
        print("[提示] 没提供/没找到内参 JSON，跳过去畸变。"
              "画面边缘可能有几 mm 系统误差，建议先跑 camera_intrinsic_calib.py")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = detect_corners(gray, cols, rows)
    need = cols * rows
    if pixels is None or len(pixels) != need:
        raise RuntimeError(
            "没检测到 %d×%d=%d 个内角点。检查：\n"
            "  - 棋盘是否完整入镜（不能出画面）\n"
            "  - --cols/--rows 是否与打印的图案一致（是【内角点】，不是格子数）\n"
            "  - 光线是否过曝/过暗、纸是否翘曲" % (cols, rows, need))

    obj = board_object_points(cols, rows, square_mm)

    # 尺度自检：格子在画面里太小 = 相机太远/棋盘太小，精度会崩
    grid = pixels.reshape(rows, cols, 2)
    step = np.concatenate([np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                           np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()])
    cell_px = float(np.median(step))
    if cell_px < 6:
        raise RuntimeError(
            "格子在画面里只有 %.1f 像素，太小了，标定精度不可用：\n"
            "  - 相机靠近一些，或换更大的棋盘/更大的格子\n"
            "  - 目标：每格 >= 15 像素" % cell_px)
    elif cell_px < 12:
        print("[警告] 格子只有 %.1f 像素，偏小；建议靠近或换大棋盘（>=15px 更稳）"
              % cell_px)
    else:
        print("[检测] 格子约 %.1f 像素，尺度合适" % cell_px)

    # 2x3 仿射只有 6 个未知数，54 个点是超定的，直接最小二乘。
    # ⚠ 注意：检测器给出的角点起止顺序存在 180° 歧义，而残差【无法区分】——
    #   棋盘格对 180° 旋转是对称的，两种顺序的残差完全一样。
    #   所以方向只能靠 --rot / --mirror 校正，平移靠实抓 --probe 校正。
    B, _res, mean, mx = hand_eye.solve(pixels, obj)
    paired = pixels

    M = compose(align_matrix(rot, mirror), B)

    # 锚点：可以用序号（配对表里的角点）或直接给像素
    if anchor is None and anchor_index is not None and anchor_xy is not None:
        if not (0 <= anchor_index < len(paired)):
            raise RuntimeError("--anchor-index %s 越界，检测到 %d 个角点（0~%d）"
                               % (anchor_index, len(paired), len(paired) - 1))
        u, v = paired[anchor_index]
        anchor = (float(u), float(v), float(anchor_xy[0]), float(anchor_xy[1]))
        print("[对齐] 锚点序号 %d -> 像素(%.1f,%.1f)" % (anchor_index, u, v))

    if anchor is not None:
        u, v, X, Y = anchor
        pred = M @ np.array([u, v, 1.0])
        M = shift_translation(M, X - pred[0], Y - pred[1])
        print("[对齐] 用锚点 (%.1f,%.1f)px -> (%.1f,%.1f)mm 修正平移" % (u, v, X, Y))

    # 残差必须用 B（像素->棋盘毫米）对 obj 算。
    # ⚠ 不能用对齐后的 M 去比 obj —— M 输出的是龙门坐标，obj 是棋盘坐标，
    #   一旦加了 --rot/--mirror/--anchor 两者就不是同一个坐标系，残差会变成垃圾。
    #   对齐是刚体变换，不会改变标定质量，所以残差沿用 solve() 的结果。
    info_res = (mean, mx)

    info = {
        "matrix": M.tolist(),
        "mean_residual": round(mean, 4),
        "max_residual": round(mx, 4),
        "n_points": int(len(paired)),
        "square_mm": float(square_mm),
        "cell_px": round(cell_px, 2),
        "rot": int(rot),
        "mirror": str(mirror),
        "image": os.path.abspath(image_path),
        "undistorted": undistorted,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if show:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (cols, rows),
                                  paired.reshape(-1, 1, 2), True)
        out_png = os.path.splitext(image_path)[0] + "_corners.png"
        try:
            cv2.imwrite(out_png, vis)
            print("[显示] 角点检测结果已保存: %s" % out_png)
        except cv2.error:
            pass

    return info, paired, obj, pixels


def print_table(pixels, obj, M, limit=6):
    """打印前几个角点的像素/棋盘毫米/最终龙门毫米，方便人工核对方向。"""
    print("\n角点抽样（核对方向用）：")
    print("  %-5s %-18s %-16s %s" % ("序号", "像素(u,v)", "棋盘(mm)", "最终坐标(mm)"))
    idx = list(range(min(limit, len(pixels)))) + \
          ([len(pixels) - 1] if len(pixels) > limit else [])
    for i in idx:
        g = M @ np.array([pixels[i][0], pixels[i][1], 1.0])
        print("  %-5d (%7.1f,%7.1f)  (%6.1f,%6.1f)  (%8.1f,%8.1f)"
              % (i, pixels[i][0], pixels[i][1], obj[i][0], obj[i][1], g[0], g[1]))
    print("  若「最终坐标」的 X 随棋盘横移而反向、或 Y 反向，"
          "用 --mirror x / --mirror y 修正。")


def write_points_csv(path, pixels, M):
    """把检测到的点对写成 CSV（u,v,x_mm,y_mm，最终龙门坐标），可直接喂 hand_eye。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("u,v,x_mm,y_mm\n")
        for u, v in pixels:
            g = M @ np.array([u, v, 1.0])
            f.write("%.3f,%.3f,%.3f,%.3f\n" % (u, v, g[0], g[1]))
    return path


# ============================ 自检 ============================
def _render_board(cols, rows, square_px):
    """画一张理想棋盘格（0/255），返回灰度图。格数 = (cols+1)×(rows+1)。"""
    sq = int(square_px)
    w, h = (cols + 1) * sq, (rows + 1) * sq
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
    return img


def self_test():
    """合成棋盘图，验证「检测 -> 求解 -> 对齐」能还原已知矩阵。"""
    cols, rows, square = DEFAULT_COLS, DEFAULT_ROWS, DEFAULT_SQUARE
    sq_px = 24
    board = _render_board(cols, rows, sq_px)

    # 已知变换 G：棋盘上以【左上角为原点】的毫米坐标 -> 画布像素。
    # 缩放 + 旋转 + 轻微剪切 + 平移，模拟相机斜看。
    th = np.deg2rad(7.0)
    k = 1.15                                   # px/mm
    G = np.array([
        [k * np.cos(th), -k * np.sin(th) + 0.06, 80.0],
        [k * np.sin(th),  k * np.cos(th),        60.0],
        [0.0,             0.0,                    1.0],
    ])
    # 棋盘像素 -> mm（左上角原点）-> 画布
    mm_per_px = square / sq_px
    W = G @ np.diag([mm_per_px, mm_per_px, 1.0])
    canvas = cv2.warpPerspective(board, W, (760, 620),
                                 flags=cv2.INTER_LINEAR, borderValue=255)

    gray = cv2.GaussianBlur(canvas, (3, 3), 0)
    pixels = detect_corners(gray, cols, rows)
    ok = True
    if pixels is None or len(pixels) != cols * rows:
        print("❌ 自检失败：合成棋盘检出 %s 个角点，期望 %d"
              % ("0" if pixels is None else len(pixels), cols * rows))
        return 1
    print("✅ 角点检测：%d/%d" % (len(pixels), cols * rows))

    obj = board_object_points(cols, rows, square)
    # 真值：objp 的原点在【第一个内角点】，而 G 的原点在棋盘左上角，差一个格子
    M_gt = np.linalg.inv(G)[:2, :].copy()
    M_gt[:, 2] -= square

    best = None
    for tag, pts in (("正序", pixels), ("反向", pixels[::-1])):
        M, _res, mean, mx = hand_eye.solve(pts, obj)
        err = np.abs(M - M_gt).max()
        print("   %s求解: 残差 平均%.2fmm 最大%.2fmm | 与真值最大偏差 %.5f"
              % (tag, mean, mx, err))
        if best is None or err < best[0]:
            best = (err, tag, M, pts, mean, mx)
    if best[0] > 1.0:
        print("❌ 自检失败：矩阵还原偏差 %.3f mm 过大" % best[0])
        ok = False
    else:
        print("✅ 矩阵还原（%s）偏差 %.3f mm < 1.0"
              "（合成图有半像素约定偏置，属正常；绝对偏差靠 --probe 补）"
              % (best[1], best[0]))
    print("   注：两种顺序残差相同（棋盘对 180° 旋转对称），"
          "这正是方向必须靠 --rot/--mirror 校正的原因")

    # 坐标回代：五个已知毫米点 -> 像素 -> 反算毫米
    M = best[2]
    probe_mm = np.array([[0, 0], [200, 0], [0, 150], [200, 150], [100, 75]],
                        dtype=np.float64)
    A = np.hstack([probe_mm + square, np.ones((len(probe_mm), 1))])
    px = (G @ A.T).T[:, :2]                    # 棋盘左上角原点 -> 画布像素
    back = (M @ np.hstack([px, np.ones((len(px), 1))]).T).T
    err = np.sqrt(((back - probe_mm) ** 2).sum(axis=1))
    if err.max() < 1.0:
        print("✅ 坐标回代：5 个点最大误差 %.3f mm < 1.0" % err.max())
    else:
        print("❌ 坐标回代误差 %.3f mm 过大" % err.max())
        ok = False

    # 平移补偿：加 (10,-5) 后预测应整体平移
    M2 = shift_translation(M, 10.0, -5.0)
    back2 = (M2 @ np.hstack([px, np.ones((len(px), 1))]).T).T
    d = back2 - back
    if np.allclose(d[:, 0], 10.0, atol=1e-9) and np.allclose(d[:, 1], -5.0, atol=1e-9):
        print("✅ 平移补偿：所有点一致偏移 (10,-5) mm")
    else:
        print("❌ 平移补偿不对: %s" % (d,))
        ok = False

    # 对齐矩阵自身
    A_y = align_matrix(0, "y")
    v = A_y @ np.array([1.0, 2.0, 1.0])
    if abs(v[0] - 1.0) < 1e-9 and abs(v[1] + 2.0) < 1e-9:
        print("✅ mirror=y 只翻 Y 轴")
    else:
        print("❌ mirror=y 结果不对: %s" % (v,))
        ok = False
    A_r90 = align_matrix(90, "none")
    v = A_r90 @ np.array([1.0, 0.0, 1.0])
    if abs(v[0]) < 1e-9 and abs(v[1] - 1.0) < 1e-9:
        print("✅ rot=90 把 +X 转到 +Y")
    else:
        print("❌ rot=90 结果不对: %s" % (v,))
        ok = False

    # ---- 端到端：把合成图当"照片"跑完整流程（读图/次级检测/锚点/JSON 读写）----
    try:
        tmpdir = tempfile.mkdtemp(prefix="planar_calib_")
        img_path = os.path.join(tmpdir, "board.png")
        cv2.imencode(".png", canvas)[1].tofile(img_path)   # 支持中文路径
        info, pts2, obj2, _raw = calibrate_from_image(img_path)

        info2, pts3, _obj3, _ = calibrate_from_image(
            img_path,
            anchor=(float(pts2[0][0]), float(pts2[0][1]), 500.0, 300.0))
        g = np.array(info2["matrix"]) @ np.array([pts3[0][0], pts3[0][1], 1.0])
        if abs(g[0] - 500.0) < 1e-6 and abs(g[1] - 300.0) < 1e-6:
            print("✅ 锚点对齐：角点 0 被钉到 (500,300)")
        else:
            print("❌ 锚点对齐失败: %s" % (g,))
            ok = False

        out_json = os.path.join(tmpdir, "hand_eye.json")
        hand_eye.save_json(out_json, np.array(info["matrix"]),
                           pts2, obj2, info["mean_residual"],
                           info["max_residual"],
                           extra={"n_points": info["n_points"]})
        m2, _meta2 = hand_eye._matrix_from_json(out_json)
        if np.allclose(m2, np.array(info["matrix"])):
            print("✅ 端到端：照片 -> 检测 -> 求解 -> JSON 读写一致")
        else:
            print("❌ 端到端 JSON 往返不一致")
            ok = False
    except Exception as e:
        print("❌ 端到端流程异常: %s: %s" % (type(e).__name__, e))
        ok = False

    print("\n%s" % ("全部自检通过 ✅" if ok else "有自检项失败 ❌"))
    return 0 if ok else 1


# ============================ 命令行 ============================
def find_default_intrinsic():
    """在常见位置找 camera_intrinsic.json。"""
    for pat in (os.path.join(_HERE, "camera_intrinsic.json"),
                os.path.join(os.getcwd(), "camera_intrinsic.json")):
        if os.path.isfile(pat):
            return pat
    hits = glob.glob(os.path.join(os.getcwd(), "**", "camera_intrinsic.json"),
                     recursive=True)
    return hits[0] if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="棋盘格平面标定：像素 -> 毫米 2x3 仿射矩阵")
    ap.add_argument("--image", help="棋盘格照片路径")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS, help="内角点列数")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="内角点行数")
    ap.add_argument("--square", type=float, default=DEFAULT_SQUARE, help="格子边长 mm")
    ap.add_argument("--intrinsic", help="camera_intrinsic.json 路径（自动找）")
    ap.add_argument("--rot", type=float, default=0, help="棋盘相对龙门的旋转角(度)")
    ap.add_argument("--mirror", default="none", choices=["none", "x", "y", "xy"],
                    help="龙门轴与棋盘轴相反时翻转")
    ap.add_argument("--anchor-index", type=int, default=None,
                    help="锚点角点序号（见打印的角点抽样表）")
    ap.add_argument("--anchor-xy", nargs=2, type=float, metavar=("X", "Y"),
                    help="该锚点在龙门坐标系里的实际毫米坐标")
    ap.add_argument("--anchor", nargs=4, type=float,
                    metavar=("U", "V", "X", "Y"),
                    help="直接给像素+毫米的锚点")
    ap.add_argument("--probe", nargs=2, type=float, metavar=("DX", "DY"),
                    help="实抓偏差补偿(mm)，不改文件里的其他参数")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 hand_eye.json 路径")
    ap.add_argument("--points-out", help="额外导出点对 CSV（u,v,x_mm,y_mm）")
    ap.add_argument("--show", action="store_true", help="保存角点可视化图")
    ap.add_argument("--dry-run", action="store_true",
                    help="只检测+预览，不写 hand_eye.json（方向没定前先这么跑）")
    ap.add_argument("--self-test", action="store_true", help="用合成图自检，无需相机")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.probe:
        hand_eye.apply_probe(args.probe[0], args.probe[1], args.out)
        return 0

    if not args.image:
        ap.print_help()
        print("\n提示：想先验证算法是否正常，跑 --self-test")
        return 2

    anchor = None
    if args.anchor:
        anchor = tuple(args.anchor)
    if args.anchor_index is not None and not args.anchor_xy:
        print("[提示] --anchor-index 需要配合 --anchor-xy X Y 一起用")
        return 2

    intrinsic = args.intrinsic or find_default_intrinsic()
    info, pixels, obj, _raw = calibrate_from_image(
        args.image, cols=args.cols, rows=args.rows, square_mm=args.square,
        intrinsic_path=intrinsic, rot=args.rot, mirror=args.mirror,
        anchor=anchor, anchor_index=args.anchor_index,
        anchor_xy=args.anchor_xy, show=args.show)

    M = np.array(info["matrix"], dtype=np.float64)
    print_table(pixels, obj, M)
    print()
    print(hand_eye.residual_report(info["mean_residual"],
                                   info["max_residual"], info["n_points"]))
    print("矩阵: %s" % (info["matrix"],))

    if args.dry_run:
        print("\n[dry-run] 没有写 hand_eye.json。确认方向和尺度没问题后，去掉 --dry-run 再跑一次。")
        return 0

    hand_eye.save_json(args.out, M, pixels, obj,
                       info["mean_residual"], info["max_residual"],
                       extra={"square_mm": info["square_mm"],
                              "cell_px": info["cell_px"],
                              "rot": info["rot"], "mirror": info["mirror"],
                              "image": info["image"],
                              "undistorted": info["undistorted"],
                              "n_points": info["n_points"]})
    print("已写入 %s" % args.out)

    if args.points_out:
        write_points_csv(args.points_out, pixels, M)
        print("点对已导出 %s" % args.points_out)

    print("\n下一步：让龙门实抓一次，看落点偏多少，然后")
    print("  python planar_calib.py --probe DX DY")
    return 0


if __name__ == "__main__":
    sys.exit(main())