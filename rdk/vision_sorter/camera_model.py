# -*- coding: utf-8 -*-
"""相机内参 / 去畸变（C100 + camera_intrinsic.json）。

为什么需要：C100 是广角镜头，实测畸变在画面中心只有 0.5px，
但到四角达到 130~200px。手眼矩阵是仿射的，没法吸收这种非线性畸变，
所以【像素→毫米之前】必须先把像素去畸变，否则边缘物料会偏十几毫米。

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


def _resolve():
    """解析一次，填充模块级状态。返回是否生效。"""
    global K, D, ACTIVE, SOURCE, META, NOTE
    K = D = None
    ACTIVE = False
    SOURCE = ""
    META = {}
    NOTE = ""

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
        ACTIVE = True
        NOTE = ""
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
    if reproj >= 0:
        parts.append("标定重投影误差=%.3fpx" % reproj)
    parts.append("边缘最大修正=%.1fpx" % max_correction_px())
    return " | ".join(parts)


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
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    corners = [(0, 0), (IMAGE_W - 1, 0), (0, IMAGE_H - 1), (IMAGE_W - 1, IMAGE_H - 1)]
    r_max = max(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 for u, v in corners) ** 0.5
    k1, k2, _p1, _p2 = (list(D) + [0.0] * 8)[:4]
    k3 = float(D[4]) if len(D) > 4 else 0.0
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
        txt = (msg + "\n  → 广角镜头边缘会偏十几毫米。先跑 tools/camera_intrinsic_calib.py"
                      " 生成 camera_intrinsic.json 放到板子上。")
        if log:
            log.warn(txt)
        else:
            print(txt)
        return
    if ACTIVE and not radial_is_monotonic():
        txt = (msg + "\n  ⚠ 内参模型在画面边缘【不单调】，去畸变可能反而更差："
                     "重新标内参，样本要覆盖画面四角。")
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