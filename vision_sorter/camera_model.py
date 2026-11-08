# -*- coding: utf-8 -*-
"""相机内参 / 去畸变（C100 + camera_intrinsic.json）。

为什么需要：C100 是广角镜头，中心几乎没畸变，但 r=0.64（料盘边缘那一圈）
修正量已经到 14 px，按 0.15mm/px 折算就是 2mm 级；越往外越夸张。
手眼矩阵是仿射的，没法吸收这种非线性畸变，
所以【像素→毫米之前】必须先把像素去畸变，否则边缘物料会偏十几毫米。

当前内参（2026-10-26 重标，见 camera_intrinsic.json）：
    fx=314.9 fy=317.0 cx=320.4 cy=255.4   纯径向模型 k1=0.2394 k2=-0.3811 k3=0.2395
    p1=p2=0（切向项固定为 0，见下面 model_note 的理由）
    单应残余 1.315px（旧内参 2.498px，干脆不去畸变 1.871px → 旧内参是坏的）

valid_radius：样本角点只覆盖到归一化半径 0.635，超过它是外推。
    工作区（料盘）在 r<0.5，够用；画面四角属于外推区，别信那里的毫米值。

关键一致性要求：
  tools/planar_calib.py 标定时已经把整张图 cv2.undistort 过（得到的就是"去畸变图像"），
  所以它算出来的矩阵是【去畸变像素 -> 毫米】。
  推理时检测框来自原始（带畸变）图像，因此必须用同一套内参做 undistortPoints，
  两边才是同一个像素坐标系。改这里之前先确认 planar_calib 的行为。

  ⚠ 两边必须同时开关。手眼标定用哪套像素，推理就必须用哪套。
     hand_eye.json 里记了 "undistorted": true/false，节点启动会比对，不一致直接报错。

文件格式（tools/camera_intrinsic_calib.py 的输出）:
    {"mtx": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist": [k1,k2,p1,p2,k3],
     "reproj_error": 0.289, "size": [640,480], ...}

查找顺序：
  1. 环境变量 GC_CAMERA_INTRINSIC_JSON
  2. config.CAMERA_INTRINSIC_JSON
  3. 本文件同目录 camera_intrinsic.json
  4. 当前工作目录 camera_intrinsic.json
  5. /root/vision_ws/camera_intrinsic.json（板端约定位置）

找不到 / 关闭 / cv2 缺失 时：全模块退化为"直通"，不改动任何像素，并 warn 一次。
"""
import io
import json
import os

import numpy as np

import config as C

try:
    import cv2
    HAS_CV2 = True
except ImportError:                      # pragma: no cover
    cv2 = None
    HAS_CV2 = False


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_W = int(getattr(C, "IMAGE_WIDTH", 640) or 640)
IMAGE_H = int(getattr(C, "IMAGE_HEIGHT", 480) or 480)

K = None
D = None
ACTIVE = False            # 是否真的在做去畸变
SOURCE = ""               # 内参来源路径（空 = 没找到）
META = {}                 # 原始 json 内容（含 reproj_error / size 等）
NOTE = ""                 # 为什么没生效
MODEL = ""                # 畸变模型标签（no_tangent / free / radial_only ...）
VALID_R = None            # 样本覆盖到的归一化半径（None = json 没写，不判断）
COVERAGE = {}             # 覆盖统计 {p50,p90,max}

_WARNED = False


def candidate_paths():
    out = []
    env = os.environ.get("GC_CAMERA_INTRINSIC_JSON", "").strip()
    if env:
        out.append(env)
    cfg = str(getattr(C, "CAMERA_INTRINSIC_JSON", "") or "").strip()
    if cfg:
        out.append(cfg)
    out.append(os.path.join(_THIS_DIR, "camera_intrinsic.json"))
    out.append(os.path.join(os.getcwd(), "camera_intrinsic.json"))
    out.append("/root/vision_ws/camera_intrinsic.json")
    return out


def _parse_meta(obj):
    """从 json 里取 K / D，容忍几种常见字段名。"""
    def pick(*keys):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return None

    m = pick("mtx", "matrix", "K", "camera_matrix", "cameraMatrix")
    d = pick("dist", "D", "dist_coeffs", "distCoeffs")
    if m is None:
        raise ValueError("缺少 mtx / camera_matrix 字段")
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError("mtx 必须是 3x3，现在是 %s" % (m.shape,))
    if d is None:
        d = np.zeros(5)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if d.size < 4:
        raise ValueError("dist 至少要有 4 个系数（k1,k2,p1,p2），现在 %d 个" % d.size)
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(d)):
        raise ValueError("内参里有 nan/inf")
    if abs(m[0, 0]) < 1e-6 or abs(m[1, 1]) < 1e-6:
        raise ValueError("fx/fy 为 0，内参无效")
    return m, d


def _load(path):
    with io.open(path, encoding="utf-8-sig") as f:
        obj = json.load(f)
    m, d = _parse_meta(obj)
    return m, d, (obj if isinstance(obj, dict) else {})


def _as_float(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


_STARTUP_NOTES = []          # 启用但不致命的问题，启动时和 describe() 一起打出来


def _frame_radius_norm():
    """画面四角里最远的那个点的归一化半径。"""
    if K is None:
        return 1.0
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    corners = [(0, 0), (IMAGE_W - 1, 0), (0, IMAGE_H - 1), (IMAGE_W - 1, IMAGE_H - 1)]
    return max((((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2) ** 0.5 for u, v in corners)


def radius_norm(u, v):
    """像素点相对主点的归一化半径（1.0 ≈ 画面半对角线）。"""
    if K is None:
        return 0.0
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return float((((float(u) - cx) / fx) ** 2 + ((float(v) - cy) / fy) ** 2) ** 0.5)


def is_covered(u, v):
    """该像素位置是否落在标定样本覆盖过的范围内（valid_radius 之内）。

    超出 = 外推区，去畸变出来的坐标是模型瞎猜的，不该拿它算毫米。
    """
    if not ACTIVE or VALID_R is None:
        return True
    return radius_norm(u, v) <= float(VALID_R) + 1e-9


def check_point(u, v):
    """给单个像素点定级，返回 (level, text)。

    level: "ok" 覆盖内 / "edge" 外推区 / "far" 明显超出（修正量很大）
    """
    r = radius_norm(u, v)
    if VALID_R is None:
        return "ok", "r=%.2f" % r
    if r <= float(VALID_R):
        return "ok", "r=%.2f" % r
    if r <= float(VALID_R) * 1.6:
        return "edge", "r=%.2f > 覆盖 %.2f（外推）" % (r, float(VALID_R))
    return "far", "r=%.2f 远超覆盖 %.2f（外推，别信毫米值）" % (r, float(VALID_R))


def _distort_points(pts_px):
    """去畸变像素 -> 带畸变像素（正畸变模型，和 cv2.projectPoints 同口径）。

    只用径向 + 切向的 Brown-Conrady 模型，就是 cv2.undistortPoints 的反方向，
    所以拿来做"正逆一致"自检。
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    k1, k2, p1, p2 = (list(D) + [0.0] * 8)[:4]
    k3 = float(D[4]) if len(D) > 4 else 0.0
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([fx * xd + cx, fy * yd + cy])


def _roundtrip_worst(points):
    """去畸变再正畸变，返回最大落点误差（px）。"""
    src = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ideal = np.asarray(cv2.undistortPoints(src.reshape(-1, 1, 2), K, D, P=K),
                       dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(ideal)):
        return float("inf")
    back = _distort_points(ideal)
    return float(np.max(np.linalg.norm(back - src, axis=1)))


def worst_corner_roundtrip_px():
    """画面四角的往返误差。

    注意：这个值大不代表内参坏 —— cv2.undistortPoints 是 5 次固定点迭代，
    在极端边缘（归一化半径 >1.1）会不收敛，牛顿法解同一模型的误差是 0。
    只在工作区之外变差，所以仅作提示，不作为停用去畸变的理由。
    """
    if not ACTIVE:
        return 0.0
    w, h = int(IMAGE_W), int(IMAGE_H)
    return _roundtrip_worst([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])


def is_model_sane(rt_tol_px=0.5, r_limit=None):
    """模型自检：(1) 画面内径向单调；(2) 工作区内正逆一致误差 < rt_tol_px。

    正逆一致的测法：拿【画面里真实存在的像素】去畸变，再正畸变回去，
    必须落回原处。反过来（先造理想点再正畸变）会造出画面外的像素，
    OpenCV 的迭代在那种点上本来就不收敛，测出来的误差没有意义。

    只查工作区（valid_radius，默认至少 0.9）：画面最角上 cv2 自己就不收敛，
    那是 cv2 的迭代次数问题，不是内参坏了，没必要因此关掉去畸变。

    旧内参就是栽在这上面：整片区域都折叠，r=1.0 处正逆差 379px，
    undistortPoints 直接吐垃圾，所以那种必须拦下来。
    """
    if not ACTIVE:
        return True, ""
    if not radial_is_monotonic():
        return False, "径向畸变在画面范围内不单调（边缘折叠），去畸变会返回垃圾坐标"

    r_lim = float(r_limit) if r_limit is not None else max(0.9, float(VALID_R or 0.0))
    w, h = int(IMAGE_W), int(IMAGE_H)
    pts = []
    for i in range(1, 17):
        for j in range(1, 13):
            u, v = i * (w - 1) / 17.0, j * (h - 1) / 13.0
            if radius_norm(u, v) <= r_lim:
                pts.append((u, v))
    if not pts:
        pts = [(w / 2.0, h / 2.0)]
    worst = _roundtrip_worst(pts)
    if not np.isfinite(worst):
        return False, "去畸变写出了 nan/inf"
    if worst > float(rt_tol_px):
        return False, "工作区内正逆一致性最差 %.1fpx（>%.1fpx），undistortPoints 结果不可信" % (
            worst, rt_tol_px)
    return True, ""


def _resolve():
    """解析一次，填充模块级状态。返回是否生效。"""
    global K, D, ACTIVE, SOURCE, META, NOTE, MODEL, VALID_R, COVERAGE
    K = D = None
    ACTIVE = False
    SOURCE = ""
    META = {}
    NOTE = ""
    MODEL = ""
    VALID_R = None
    COVERAGE = {}
    del _STARTUP_NOTES[:]

    if not bool(getattr(C, "UNDISTORT_ENABLE", True)):
        NOTE = "config.UNDISTORT_ENABLE = False"
        return False
    if not HAS_CV2:
        NOTE = "没装 opencv（import cv2 失败）"
        return False

    for p in candidate_paths():
        if not p or not os.path.isfile(p):
            continue
        try:
            m, d, meta = _load(p)
        except Exception as e:                       # noqa: BLE001
            NOTE = "%s 读取失败：%s: %s" % (p, type(e).__name__, e)
            continue
        K, D, META, SOURCE = m, d, meta, os.path.abspath(p)
        MODEL = str(meta.get("model", "") or "")
        VALID_R = _as_float(meta.get("valid_radius"))
        cov = meta.get("radial_coverage")
        COVERAGE = cov if isinstance(cov, dict) else {}
        ACTIVE = True
        NOTE = ""

        # 分辨率是否和标定时一致：不一致的话 fx 会整体错一个比例，必须喊出来
        size = meta.get("size")
        try:
            sw, sh = int(size[0]), int(size[1])
        except (TypeError, ValueError, IndexError):
            sw = sh = 0
        if sw and sh and (sw != IMAGE_W or sh != IMAGE_H):
            _STARTUP_NOTES.append(
                "内参按 %dx%d 标定，但节点配置是 %dx%d —— 去畸变结果不可信"
                % (sw, sh, IMAGE_W, IMAGE_H))

        # 合理性闸门：模型在画面内必须单调且可逆，否则去畸变会把像素揉乱
        ok, why = is_model_sane()
        if not ok:
            ACTIVE = False
            NOTE = "%s 的畸变模型不合理：%s" % (os.path.basename(p), why)
            return False
        cq = worst_corner_roundtrip_px()
        if cq > 1.0:
            _STARTUP_NOTES.append(
                "画面四角往返误差 %.0fpx：那里归一化半径 >1.1，cv2 迭代不收敛 + 内参外推，"
                "别信画面角上的毫米值" % cq)
        return True

    NOTE = "没找到 camera_intrinsic.json（找过：%s）" % ", ".join(candidate_paths())
    return False


def reload():
    """重新解析（换了内参文件后调用）。"""
    return _resolve()


def is_active():
    return ACTIVE


def describe():
    if not ACTIVE:
        return "去畸变未启用（%s）" % (NOTE or "未知原因")
    k = np.array(K).ravel()
    try:
        reproj = float(META.get("reproj_error", -1))
    except (TypeError, ValueError):
        reproj = -1.0
    parts = ["去畸变已启用",
             "内参=%s" % SOURCE,
             "fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (k[0], k[4], k[2], k[5])]
    if MODEL:
        parts.append("模型=%s" % MODEL)
    if reproj >= 0:
        parts.append("标定重投影误差=%.3fpx" % reproj)
    if VALID_R is not None:
        parts.append("可信半径=%.2f(画面归一化)" % float(VALID_R))
    parts.append("边缘最大修正=%.1fpx" % max_correction_px())
    text = " | ".join(parts)
    for note in _STARTUP_NOTES:
        text += "\n  ⚠ " + note
    return text


def undistort_points(pts):
    """(N,2) 像素 -> 去畸变后的 (N,2) 像素（同一像素坐标系，newCameraMatrix=K）。

    未启用时原样返回，调用方不需要判断开关。
    """
    arr = np.asarray(pts, dtype=np.float64)
    single = (arr.ndim == 1)
    if single:
        arr = arr.reshape(1, 2)
    if not ACTIVE or arr.size == 0:
        return arr[0] if single else arr
    src = arr.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.undistortPoints(src, K, D, P=K)
    out = np.asarray(out, dtype=np.float64).reshape(-1, 2)
    return out[0] if single else out


def undistort_point(u, v):
    """单点版本，返回 (u', v')。"""
    p = undistort_points(np.array([[float(u), float(v)]]))
    return float(p[0][0]), float(p[0][1])


def max_correction_px(w=None, h=None):
    """画面四角的最大修正量（px），用来判断内参是否离谱。未启用返回 0。"""
    if not ACTIVE:
        return 0.0
    w = int(w or IMAGE_W)
    h = int(h or IMAGE_H)
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]],
                       dtype=np.float64)
    fixed = undistort_points(corners)
    return float(np.max(np.linalg.norm(fixed - corners, axis=1)))


def radial_is_monotonic():
    """检查径向畸变函数在画面范围内是否单调递增。

    不单调 = 模型在边缘出现"折叠"，undistortPoints 会把不同像素映射到同一处，
    去畸变结果不可信（通常说明内参过拟合或标定样本没覆盖边缘）。
    """
    if not ACTIVE:
        return True
    k1, k2, _p1, _p2 = (list(D) + [0.0] * 8)[:4]
    k3 = float(D[4]) if len(D) > 4 else 0.0
    r_max = _frame_radius_norm()
    r = np.linspace(1e-6, r_max, 400)
    r_u = r * (1.0 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    return bool(np.all(np.diff(r_u) > 0))


def warn_if_disabled(logger=None):
    """启动时调用一次：没启用就喊一声，启用了就报状态。"""
    global _WARNED
    log = logger
    msg = "[camera] " + describe()
    if not ACTIVE and not _WARNED:
        _WARNED = True
        txt = (msg + "\n  → 广角镜头边缘会偏十几毫米。先跑 tools/board_calib_server.py"
                      " 采集 → tools/eval_intrinsic.py --recalib-flags no_tangent 解算"
                      " 生成 camera_intrinsic.json 放到板子上。")
        if log:
            log.warn(txt)
        else:
            print(txt)
        return
    ok, why = is_model_sane()
    if ACTIVE and not ok:                       # reload 换了内参后的兜底
        txt = msg + "\n  ⚠ 内参模型不可用：%s → 去畸变已关闭" % why
        if log:
            log.warn(txt)
        else:
            print(txt)
        return
    if log:
        log.info(msg)
    else:
        print(msg)


def check_hand_eye_consistency(hand_eye_meta, logger=None):
    """比对 hand_eye.json 的 undistorted 标记与当前开关，不一致就报。

    标定用带畸变像素、推理却去畸变（或反过来）会引入几十毫米系统误差，
    而且表现是"稳定地偏"，极难现场定位，所以这里强制喊出来。
    """
    if not isinstance(hand_eye_meta, dict):
        return None
    flag = hand_eye_meta.get("undistorted", None)
    if flag is None:
        return None                     # 旧版 json 没记，无法比对
    flag = bool(flag)
    if flag == bool(ACTIVE):
        return True
    txt = ("[camera] ⚠⚠ 手眼标定与去畸变设置不一致！\n"
           "  hand_eye.json 标定时 %s 去畸变，当前节点 %s 去畸变。\n"
           "  两者必须一致，否则坐标会稳定地偏十几毫米。\n"
           "  → 要么重新标定（tools/planar_calib.py --image ...），"
           "要么把 config.UNDISTORT_ENABLE 改成 %s。"
           % ("有" if flag else "没有",
              "有" if ACTIVE else "没有",
              "True" if flag else "False"))
    if logger:
        logger.error(txt)
    else:
        print(txt)
    return False


_resolve()