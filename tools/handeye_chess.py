# -*- coding: utf-8 -*-
"""手眼标定（棋盘格 + 夹爪点两次）。

【当前状态：可选实验方案，非生产主流程】
生产流程是 4 锚点方案 tools/collect_anchors.py + tools/planar_calib.py --anchors-file，
说明见 /root/vision_ws/HANDEYE.md。本方案保留备选。

为什么当初设计成这样（供参考；当前已不作为生产流程依据）
------------------
旧流程要人把夹爪精确对到 4 个点上（还要避开死区），每次目测偏差直接变成标定
误差（σ/2 传进去）。这里把「精度来源」从人眼挪到棋盘格：

    棋盘格平放在料盘（= 抓取平面）上，格子间距已知 → 相机自己就能把像素
    换算成料盘坐标，精度 ~0.1px；人只需要把夹爪移到**两个指定内角点**上方
    点两次，求出「料盘坐标系 → 龙门坐标系」的 4 自由度相似变换。

2 点严格确定 4 个自由度（平移2 + 旋转1 + 尺度1），所以**不需要更多点**。
但 2 点没有任何冗余，点错了不会自己暴露 —— 所以强烈建议再点 1 个
「验算点」（不参与解算），solve 时报告预测误差；超过阈值拒绝写文件。

整条链路
--------
    pixel --(棋盘格已知尺寸)--> board_mm --(两次点触)--> gantry_mm

用法
----
    # 0) 棋盘格摆到料盘上（9x6 内角点，别贴边，四角都进画面）
    python3 tools/handeye_chess.py snap --device /dev/cam0 --square 24
    python3 tools/handeye_chess.py snap --image board.jpg --square 24   # 用已有照片

    # 1) 夹爪点两次（对角两点，离得越远角度越准）
    python3 tools/handeye_chess.py point --corner 10 --port /dev/ttyACM0
    python3 tools/handeye_chess.py point --corner 43 --port /dev/ttyACM0

    # 2) 解算 + 验收 + 写 /root/vision_ws/hand_eye.json
    python3 tools/handeye_chess.py solve

    # 3)（建议）验算点：不参与解算，只用来暴露点错/镜像
    python3 tools/handeye_chess.py point --corner 16 --verify
    python3 tools/handeye_chess.py solve

    # 4) 离线自测（不需要相机、不需要 32）
    python3 tools/handeye_chess.py selftest

角点编号（9 列 x 6 行，index = row*9 + col）::

    #0  #1  ...               #8
    #9 ...                   #17
    ...
    #45 ...                  #53

    左上=#0，右上=#8，左下=#45，右下=#53；内缩一格的四角是 #10 #16 #37 #43。

产物
----
    <work>/board.json    棋盘格照片的角点像素 + 每点对应的料盘毫米坐标
    <work>/points.json   夹爪点触记录（anchor=参与解算 / verify=只验算）
    <out>                hand_eye.json（格式与 rdk/vision_sorter/hand_eye.py 一致）
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

PATTERN = (9, 6)                  # 内角点列数, 行数
DEFAULT_SQUARE = 24.0             # 格边长 mm（量一次即可，只影响尺度自检）

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


# ================================================================= 基础工具
def log(msg=""):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def board_mm(pattern=PATTERN, square=DEFAULT_SQUARE):
    """棋盘格内角点 -> 料盘毫米坐标（#0 为原点，x 向右、y 向下）。"""
    cols, rows = pattern
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((c * float(square), r * float(square)))
    return np.array(out, dtype=np.float64)


def corner_rc(idx, pattern=PATTERN):
    cols, rows = pattern
    if not (0 <= idx < cols * rows):
        raise ValueError("角点编号 %s 越界（0~%d）" % (idx, cols * rows - 1))
    return idx // cols, idx % cols


def fit_similarity(src, dst):
    """4 自由度相似变换 dst = s*R*src + t（不含镜像），最小二乘。

    用复数表示求解：w = A*(z - z_mean) + w_mean，A = a + ib。
    返回 (M 2x3, info)。2 点时残差恒为 0，这是数学性质不是代码问题。
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape:
        raise ValueError("点数不一致: %d vs %d" % (len(src), len(dst)))
    if len(src) < 2:
        raise ValueError("至少需要 2 个点")
    zs = src[:, 0] + 1j * src[:, 1]
    zd = dst[:, 0] + 1j * dst[:, 1]
    zc = zs.mean()
    dc = zd.mean()
    dz = zs - zc
    dw = zd - dc
    den = np.sum(np.abs(dz) ** 2)
    if den < 1e-12:
        raise ValueError("所有点重合，无法解算")
    A = np.sum(dw * np.conj(dz)) / den
    a, b = float(A.real), float(A.imag)
    M = np.array([[a, -b, 0.0], [b, a, 0.0]], dtype=np.float64)
    M[:, 2] = np.array([dc.real, dc.imag]) - M[:, :2] @ np.array([zc.real, zc.imag])
    pred = apply_M(M, src)
    res = np.linalg.norm(pred - dst, axis=1)
    info = {
        "scale": float(np.hypot(a, b)),
        "rot_deg": float(np.degrees(np.arctan2(b, a))),
        "n": int(len(src)),
        "mean": float(res.mean()),
        "max": float(res.max()),
        "residuals": [float(x) for x in res],
    }
    return M, info


def apply_M(M, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ np.asarray(M, dtype=np.float64)[:, :2].T + \
        np.asarray(M, dtype=np.float64)[:, 2]


def compose(M_a_b, M_b_c):
    """gantry = A(board), board = B(pixel) -> gantry = A∘B(pixel)。"""
    A = np.asarray(M_a_b, dtype=np.float64)
    B = np.asarray(M_b_c, dtype=np.float64)
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = A[:, :2] @ B[:, :2]
    out[:, 2] = A[:, :2] @ B[:, 2] + A[:, 2]
    return out


# ================================================================= 内参
def find_intrinsic(explicit=""):
    """找 camera_intrinsic.json（与 camera_model.py 的搜索顺序保持近似）。"""
    cand = []
    if explicit:
        cand.append(explicit)
    cand.append(os.environ.get("GC_CAMERA_INTRINSIC_JSON", ""))
    here = os.path.dirname(os.path.abspath(__file__))
    cand += [
        os.path.join(here, "camera_intrinsic.json"),
        os.path.join(here, "..", "camera_intrinsic.json"),
        os.path.join(os.getcwd(), "camera_intrinsic.json"),
        os.path.join(here, "..", "vision_sorter", "camera_intrinsic.json"),
        "/root/vision_ws/camera_intrinsic.json",
    ]
    for p in cand:
        p = os.path.abspath(p) if p else ""
        if p and os.path.isfile(p):
            try:
                with io.open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                K = np.array(d.get("mtx") or d.get("camera_matrix"), dtype=np.float64)
                D = np.array(d.get("dist") or d.get("dist_coeffs"),
                             dtype=np.float64).reshape(-1)
                if K.shape == (3, 3) and D.size >= 4:
                    return K, D, p
            except Exception:
                continue
    return None, None, ""


def undistort_pts(pts, K, D):
    """像素点去畸变（保持同一像素坐标系，P=K）。"""
    if K is None:
        return np.asarray(pts, dtype=np.float64)
    import cv2
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, K, D, P=K)
    return out.reshape(-1, 2)


# ================================================================= 采图/检测
def grab_frame(device, width, height, tries=12):
    import cv2
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if os.name != "nt" else \
        cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError("打不开相机 %s（多半是检测节点正占着相机：先 "
                       "bash /root/vision_ws/stop_all.sh，或用 --image 传一张照片）"
                       % device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(tries):                 # 前几帧常是黑/半帧
        ok, f = cap.read()
        if ok and f is not None and f.size:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("相机 %s 读不到帧" % device)
    return frame


def detect_corners(bgr, pattern=PATTERN):
    """返回 (ok, corners_px[N,2], 说明)。经典法 -> 快速法兜底。"""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + \
        cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    method = "findChessboardCorners"
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        method = "findChessboardCornersSB"
    if not ok or corners is None:
        return False, None, "两种检测都失败：检查九列六行的内角点是否全部在画面内、别贴边"
    corners = corners.reshape(-1, 2).astype(np.float64)
    if method != "findChessboardCornersSB":
        corners = cv2.cornerSubPix(
            gray, corners.reshape(-1, 1, 2), (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
        ).reshape(-1, 2)
    return True, corners, method


# ================================================================= 文件
def _dump(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load(path, default=None):
    if not os.path.isfile(path):
        return default
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paths(work, out):
    return (os.path.join(work, "board.json"),
            os.path.join(work, "points.json"),
            out)


# ================================================================= 子命令
def cmd_snap(args):
    """拍/读一张棋盘格照，检测角点，写 board.json。"""
    board_p, _, _ = paths(args.work, args.out)
    if args.image:
        import cv2
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise RuntimeError("读不到图片 %s" % args.image)
        src = args.image
    else:
        bgr = grab_frame(args.device, args.width, args.height)
        src = "camera:%s" % args.device

    K, D, ipath = find_intrinsic(args.intrinsic)
    ok, corners, how = detect_corners(bgr, tuple(args.pattern))
    log("检测方法: %s" % how)
    if not ok:
        log("✗ %s" % how)
        return 2
    log("✓ 检出 %d 个内角点，图像 %dx%d" % (len(corners), bgr.shape[1], bgr.shape[0]))

    if K is not None and not args.no_undistort:
        corners_u = undistort_pts(corners, K, D)
        log("去畸变: 已启用 内参=%s" % ipath)
        moved = np.linalg.norm(corners_u - corners, axis=1)
        log("          修正量 平均%.2fpx 最大%.2fpx" % (moved.mean(), moved.max()))
        undist = True
    else:
        corners_u = corners
        undist = False
        log("⚠ 去畸变: 未启用（%s）—— 边缘角点会有几个像素偏差"
            % ("明确 --no-undistort" if args.no_undistort else "找不到 camera_intrinsic.json"))

    bmm = board_mm(tuple(args.pattern), args.square)
    M_pb, info = fit_similarity(corners_u, bmm)
    log("像素->料盘 拟合: 1mm = %.3f px（尺度 %.4f mm/px）· 残差 平均%.3fpx 最大%.3fpx"
        % (1.0 / max(info["scale"], 1e-9), info["scale"], info["mean"], info["max"]))
    if info["max"] > 1.5:
        log("⚠ 残差偏大（>1.5px）：棋盘格可能不平/有反光/角点检测错位，先别往下做")

    _dump(board_p, {
        "image": src,
        "image_size": [int(bgr.shape[1]), int(bgr.shape[0])],
        "pattern": list(args.pattern),
        "square_mm": float(args.square),
        "undistorted": bool(undist),
        "intrinsic": ipath,
        "corners_px": [[round(float(u), 3), round(float(v), 3)] for u, v in corners_u],
        "board_mm": [[round(float(x), 3), round(float(y), 3)] for x, y in bmm],
        "fit_px_to_board": {
            "matrix": M_pb.tolist(),
            "px_per_mm": float(info["scale"]),
            "residual_mean_px": info["mean"],
            "residual_max_px": info["max"],
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log("已写 %s" % board_p)
    log("")
    log("各角点像素坐标（挑两个离得远的内角点去点夹爪，建议 #10 与 #43）:")
    for idx in (0, 10, 16, 37, 43, len(corners_u) - 1):
        u, v = corners_u[idx]
        bx, by = bmm[idx]
        log("  #%-2d 像素(%7.1f,%7.1f)  料盘(%6.1f,%6.1f)mm" % (idx, u, v, bx, by))
    return 0


def _read_gantry(port, baud, window, tol):
    """从 32 读一次稳定坐标：采样 window 秒，要求点数够且波动 < tol。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                    "vision_sorter"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "vision_sorter"))
    import comm                                   # 复用与节点同一套协议解析
    from serial.tools import list_ports           # noqa: F401  确认 pyserial 在
    c = comm.Comm()
    c.open()
    log("串口已开: %s  采样 %.1fs（夹爪保持不动）..." % (c.port, window))
    samples = []
    t_end = time.time() + window
    last = None
    try:
        while time.time() < t_end:
            pos = c.poll_position_f()
            if pos is not None:
                samples.append(pos)
                last = pos
            time.sleep(0.01)
    finally:
        c.close()
    if len(samples) < 5:
        return None, "只收到 %d 帧坐标（32 没在回传？看 serial 是否通、valid 是否为 0）" \
            % len(samples)
    arr = np.array(samples, dtype=np.float64)
    spread = float(np.max(np.ptp(arr, axis=0)))
    med = np.median(arr, axis=0)
    log("收到 %d 帧，中位 (%.2f, %.2f)，波动 %.2fmm" % (len(samples), med[0], med[1], spread))
    if spread > tol:
        return None, "夹爪还在动（波动 %.2fmm > %.2fmm），停稳了再来一次" % (spread, tol)
    return (float(med[0]), float(med[1])), ""


def cmd_point(args):
    """记录一次夹爪点触（anchor 参与解算 / verify 只验算）。"""
    board_p, points_p, _ = paths(args.work, args.out)
    board = _load(board_p)
    if not board:
        log("✗ 还没有 %s，先跑 snap" % board_p)
        return 2
    idx = int(args.corner)
    if not (0 <= idx < len(board["corners_px"])):
        log("✗ 角点编号 %d 越界" % idx)
        return 2

    if args.gantry:
        gx, gy = float(args.gantry[0]), float(args.gantry[1])
        log("手输坐标: (%.2f, %.2f)" % (gx, gy))
    else:
        g, err = _read_gantry(args.port, args.baud, args.window, args.tol)
        if g is None:
            log("✗ %s" % err)
            return 3
        gx, gy = g

    pts = _load(points_p, default={"anchors": [], "verify": [], "square_mm": None})
    pts["square_mm"] = board.get("square_mm")
    role = "verify" if args.verify else "anchor"
    item = {"corner": idx, "gantry": [round(gx, 3), round(gy, 3)],
            "role": role, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if role == "verify":
        pts["verify"] = [p for p in pts["verify"] if p["corner"] != idx]
        pts["verify"].append(item)
    else:
        pts["anchors"] = [p for p in pts["anchors"] if p["corner"] != idx]
        pts["anchors"].append(item)
    _dump(points_p, pts)

    bx, by = board["board_mm"][idx]
    log("已记录 %s: 角点 #%d 料盘(%.1f,%.1f)mm -> 龙门(%.2f,%.2f)mm"
        % (role, idx, bx, by, gx, gy))
    log("当前 anchor %d 个 / verify %d 个"
        % (len(pts["anchors"]), len(pts["verify"])))
    if len(pts["anchors"]) < 2:
        log("还需要再点 1 个 anchor（换一个离得远的内角点，比如对角那个）")
    elif not pts["verify"]:
        log("已有 2 个 anchor，可以 solve 了；建议再加 1 个 verify 点暴露点错")
    return 0


def cmd_solve(args):
    """解算 + 验收 + 写 hand_eye.json。"""
    board_p, points_p, out_p = paths(args.work, args.out)
    board = _load(board_p)
    pts = _load(points_p)
    if not board:
        log("✗ 没有 %s" % board_p)
        return 2
    if not pts or len(pts.get("anchors", [])) < 2:
        log("✗ 需要 2 个 anchor 点（当前 %d）"
            % len(pts.get("anchors", []) if pts else []))
        return 2

    bmm = np.array(board["board_mm"], dtype=np.float64)
    anchors = pts["anchors"]
    idxs = [int(a["corner"]) for a in anchors]
    src = bmm[idxs]
    dst = np.array([a["gantry"] for a in anchors], dtype=np.float64)

    # 两点离得太近 -> 角度误差被放大
    sep = float(np.linalg.norm(src[0] - src[1]))
    log("两 anchor 点相距 %.1fmm（建议 >=100mm，越远角度越准）" % sep)
    warn_sep = sep < 60.0

    M_gb, info_gb = fit_similarity(src, dst)
    scale, rot = info_gb["scale"], info_gb["rot_deg"]
    log("料盘->龙门: 尺度 %.6f  旋转 %+.3f°  （尺度应≈1.000000）" % (scale, rot))

    M_pb = np.array(board["fit_px_to_board"]["matrix"], dtype=np.float64)
    M_pg = compose(M_gb, M_pb)

    checks = []
    if abs(scale - 1.0) > 0.02:
        checks.append("尺度偏差 %.2f%%：square_mm 与 32 的单位不一致（打印缩放没量准 / "
                      "格边长填错）" % ((scale - 1.0) * 100))
    if warn_sep:
        checks.append("两个 anchor 相距只有 %.1fmm，角度误差会被放大" % sep)

    # 验算点（不参与解算）
    verr = None
    vlist = pts.get("verify", [])
    if vlist:
        v = vlist[0]
        vidx = int(v["corner"])
        pred = apply_M(M_pg, np.array([board["corners_px"][vidx]]))[0]
        verr = float(np.linalg.norm(pred - np.array(v["gantry"], dtype=np.float64)))
        log("验算点 #%d: 预测(%.2f,%.2f) 实测(%.2f,%.2f) 误差 %.2fmm"
            % (vidx, pred[0], pred[1], v["gantry"][0], v["gantry"][1], verr))
        if verr > args.max_verify:
            checks.append("验算点误差 %.2fmm > %.2fmm：两次点触有一次点错角点了，"
                          "或者棋盘格不在抓取平面高度上" % (verr, args.max_verify))
    else:
        log("⚠ 没有验算点：2 点解算没有冗余，点错了不会自己暴露。"
            "建议再 point 一个 --verify 角点。")

    # 覆盖范围自检：棋盘格四角预测出的龙门坐标是否合理
    corners4 = [0, 8, 45, 53]
    log("")
    log("棋盘格四角 -> 龙门坐标预测（人工核对一下方向/正负号）:")
    for idx in corners4:
        p = apply_M(M_pg, np.array([board["corners_px"][idx]]))[0]
        log("  #%-2d 料盘(%6.1f,%6.1f)mm -> 龙门(%8.2f,%8.2f)mm"
            % (idx, bmm[idx][0], bmm[idx][1], p[0], p[1]))

    if checks:
        log("")
        for c in checks:
            log("✗ %s" % c)
        if not args.force:
            log("")
            log("拒绝写入 %s（确认无误可加 --force）" % out_p)
            return 4
        log("⚠ --force：仍然写入")

    matrix = [[round(float(M_pg[0][0]), 9), round(float(M_pg[0][1]), 9),
               round(float(M_pg[0][2]), 6)],
              [round(float(M_pg[1][0]), 9), round(float(M_pg[1][1]), 9),
               round(float(M_pg[1][2]), 6)]]
    doc = {
        "matrix": matrix,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mean_residual": (round(verr, 3) if verr is not None else 0.0),
        "max_residual": (round(verr, 3) if verr is not None else 0.0),
        "n_points": len(anchors) + (1 if verr is not None else 0),
        "square_mm": float(board.get("square_mm") or 0.0),
        "rot": 0,
        "mirror": "none",
        "undistorted": bool(board.get("undistorted")),
        "method": "chess+2point",
        "scale_board_to_gantry": round(scale, 9),
        "rot_deg": round(rot, 4),
        "verified": verr is not None,
        "verify_error_mm": (round(verr, 3) if verr is not None else None),
        "px_per_mm": round(float(board["fit_px_to_board"]["px_per_mm"]), 6),
        "anchors": anchors,
        "image": board.get("image", ""),
        "board_px_residual_max": round(
            float(board["fit_px_to_board"]["residual_max_px"]), 3),
    }
    _dump(out_p, doc)
    log("")
    log("✓ 已写 %s" % out_p)
    log("  matrix = %s" % json.dumps(matrix))
    log("  单位：像素 -> 毫米；去畸变=%s；验收误差 %s"
        % ("是" if doc["undistorted"] else "否",
           ("%.2fmm" % verr) if verr is not None else "无（未验算）"))
    log("")
    log("下一步：重启业务节点，启动日志应显示 [hand_eye] 已标定 ...")
    return 0


# ================================================================= 自测
def _synth(scale=1.0, rot=17.0, t=(120.0, -80.0), noise_px=0.0, px_per_mm=13.0,
           pattern=PATTERN, square=DEFAULT_SQUARE, seed=0):
    """造一组「真值」：料盘 -> 龙门 的相似变换 + 像素坐标。"""
    rng = np.random.RandomState(seed)
    bmm = board_mm(pattern, square)
    th = np.radians(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    gantry = scale * (bmm @ R.T) + np.array(t)
    # 像素：随便一个相似变换 + 噪声
    th2 = np.radians(-23.0)
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])
    px = (bmm @ R2.T) * px_per_mm + np.array([320.0, 240.0])
    if noise_px:
        px = px + rng.randn(*px.shape) * noise_px
    return bmm, gantry, px


def cmd_selftest(args):
    fails = []

    def check(name, cond, extra=""):
        log(("  ✓ " if cond else "  ✗ ") + name + (("  " + extra) if extra else ""))
        if not cond:
            fails.append(name)

    log("[1] 无噪声理想解算")
    bmm, gantry, px = _synth()
    M_pb, _ = fit_similarity(px, bmm)
    for idxs in ([10, 43], [0, 53], [16, 37]):
        M_gb, _ = fit_similarity(bmm[idxs], gantry[idxs])
        M_pg = compose(M_gb, M_pb)
        err = np.max(np.linalg.norm(apply_M(M_pg, px) - gantry, axis=1))
        check("两点 %s 全场最大误差 < 1e-6mm" % idxs, err < 1e-6, "实测 %.2e" % err)

    log("[2] 尺度/角度还原")
    bmm, gantry, px = _synth(scale=1.0, rot=17.0)
    M_pb, _ = fit_similarity(px, bmm)
    M_gb, info = fit_similarity(bmm[[10, 43]], gantry[[10, 43]])
    check("scale≈1.0", abs(info["scale"] - 1.0) < 1e-9, "%.9f" % info["scale"])
    check("rot≈17°", abs(info["rot_deg"] - 17.0) < 1e-6, "%.6f" % info["rot_deg"])

    log("[3] 像素噪声 0.3px -> 误差仍是毫米级以下")
    bmm, gantry, px = _synth(noise_px=0.3, seed=7)
    M_pb, _ = fit_similarity(px, bmm)
    M_gb, _ = fit_similarity(bmm[[10, 43]], gantry[[10, 43]])
    M_pg = compose(M_gb, M_pb)
    err = np.max(np.linalg.norm(apply_M(M_pg, px) - gantry, axis=1))
    check("全场最大误差 < 1.5mm", err < 1.5, "实测 %.3fmm" % err)

    log("[4] 尺度异常能被抓到（square_mm 填大 4%）")
    bmm_true, gantry, px = _synth(square=24.0)          # 物理真实 24mm
    bmm_wrong = board_mm(PATTERN, 25.0)                 # 人却按 25mm 算
    M_pb, _ = fit_similarity(px, bmm_wrong)
    M_gb, info = fit_similarity(bmm_wrong[[10, 43]], gantry[[10, 43]])
    check("scale 偏离 1 超过 2%", abs(info["scale"] - 1.0) > 0.02,
          "实测 %.4f" % info["scale"])
    check("尺度正好是 24/25", abs(info["scale"] - 24.0 / 25.0) < 1e-9,
          "%.9f" % info["scale"])

    log("[5] 点错一个角点 -> 验算点误差变大（可被拒写）")
    bmm, gantry, px = _synth()
    bad = gantry.copy()
    bad[43, 1] += 6.0                          # 第二次点触偏了 6mm
    M_gb, _ = fit_similarity(bmm[[10, 43]], bad[[10, 43]])
    M_pg = compose(M_gb, M_pb)
    pred = apply_M(M_pg, np.array([px[16]]))[0]
    verr = float(np.linalg.norm(pred - gantry[16]))
    check("验算误差 > 3mm", verr > 3.0, "实测 %.2fmm" % verr)

    log("[6] 尺度自检：两点相距过近会被提示")
    bmm, gantry, px = _synth()
    sep = float(np.linalg.norm(bmm[10] - bmm[11]))
    check("#10 与 #11 相距 < 60mm", sep < 60.0, "%.1fmm" % sep)

    log("[7] apply/compose 与直接相乘一致")
    M1 = np.array([[0.5, 0.1, 3.0], [-0.1, 0.5, -2.0]])
    M2 = np.array([[1.2, -0.3, 7.0], [0.3, 1.2, 1.0]])
    Mc = compose(M1, M2)
    p = np.array([[11.0, 22.0]])
    check("compose == M1·M2·p", np.allclose(apply_M(Mc, p),
                                            apply_M(M1, apply_M(M2, p)), atol=1e-12))

    log("")
    if fails:
        log("✗ 自测失败 %d 项: %s" % (len(fails), ", ".join(fails)))
        return 1
    log("✓ 自测全部通过")
    return 0


# ================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="手眼标定：棋盘格 + 夹爪点两次",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--work", default="/root/vision_ws/handeye",
                    help="中间产物目录（board.json/points.json）")
    ap.add_argument("--out", default="/root/vision_ws/hand_eye.json",
                    help="输出的标定文件")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("snap", help="检测棋盘格角点，写 board.json")
    s.add_argument("--image", default="", help="用已有照片（不给则现场拍）")
    s.add_argument("--device", default="/dev/cam0", help="相机设备（现场拍时用）")
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--square", type=float, default=DEFAULT_SQUARE, help="格边长 mm")
    s.add_argument("--pattern", type=int, nargs=2, default=list(PATTERN))
    s.add_argument("--intrinsic", default="", help="camera_intrinsic.json 路径")
    s.add_argument("--no-undistort", action="store_true")
    s.set_defaults(func=cmd_snap)

    p = sub.add_parser("point", help="记录一次夹爪点触")
    p.add_argument("--corner", type=int, required=True, help="角点编号（如 10/43）")
    p.add_argument("--verify", action="store_true", help="只做验算，不参与解算")
    p.add_argument("--gantry", type=float, nargs=2, default=None,
                   help="手输龙门坐标 X Y（不给则从 32 读）")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--window", type=float, default=3.0, help="采样秒数")
    p.add_argument("--tol", type=float, default=0.8, help="允许波动 mm")
    p.set_defaults(func=cmd_point)

    v = sub.add_parser("solve", help="解算 + 验收 + 写 hand_eye.json")
    v.add_argument("--max-verify", type=float, default=5.0,
                   help="验算点允许误差 mm，超过则拒写")
    v.add_argument("--force", action="store_true", help="检查不过也写")
    v.set_defaults(func=cmd_solve)

    t = sub.add_parser("selftest", help="离线自测（不需要相机/32）")
    t.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())