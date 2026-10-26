# -*- coding: utf-8 -*-
"""离线单测：串口协议 + 手眼标定链路（不需要 RDK、不需要 STM32、不需要相机）。

跑法：
    python tools/test_protocol.py

覆盖：
  1. 下行 10 字节帧（含 CRC 范围、颜色编号、绿帧字节级不变）
  2. 上行 0x07 坐标帧的组帧/解析（分包、CRC 错、杂字节、与 0x06 混流）
  3. 抓取完成 1 字节 / 2 字节（GRAB_STATUS_ENABLE）两条路径
  4. hand_eye：JSON 优先加载、坏 JSON 回退占位、--probe 平移补偿
  5. planar_calib：align_matrix / compose / solve 的数学正确性
"""
import os
import sys
import json
import tempfile

import numpy as np

try:
    import cv2
except ImportError:          # 没装 opencv 时跳过"合成棋盘"相关回归项
    cv2 = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
# 兼容两种布局：仓库里 tools/ + rdk/vision_sorter/；板端 /root/vision_ws/tools + vision_sorter
for _p in (os.path.join(_PARENT, "rdk", "vision_sorter"),
           os.path.join(_PARENT, "vision_sorter"),
           "/root/vision_ws/vision_sorter", _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import config as C            # noqa: E402
import comm                  # noqa: E402
import hand_eye              # noqa: E402
import camera_model          # noqa: E402
import planar_calib          # noqa: E402
import safety                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("OK   %s" % name)
    else:
        FAIL.append(name)
        print("FAIL %s %s" % (name, detail))


# ============================ 1. 下行帧 ============================
def test_downlink():
    fr = comm.build_frame("YZHU", "green", -12, 407)
    hexs = " ".join("%02X" % b for b in fr)
    check("下行帧长度=10", len(fr) == 10, hexs)
    check("下行帧字节完全一致", hexs == "AA 55 06 03 F4 FF 97 01 FE 0D", hexs)
    check("CRC 覆盖 0~7 字节", comm._crc8(fr[0:8]) == fr[8])
    check("帧头/帧尾", fr[0] == 0xAA and fr[1] == 0x55 and fr[9] == 0x0D)

    # 坐标小端
    x = int.from_bytes(fr[4:6], "little", signed=True)
    y = int.from_bytes(fr[6:8], "little", signed=True)
    check("x/y 小端且值正确", (x, y) == (-12, 407), "%s" % ((x, y),))

    # 颜色编号固定，不随 HSV 字典增删漂移
    check("green=3", comm.COLOR_ID["green"] == 3)
    check("black=8（未前移）", comm.COLOR_ID["black"] == 8)
    check("white=9（未前移）", comm.COLOR_ID["white"] == 9)
    check("pink=7（2026-10-26 启用）", comm.COLOR_ID["pink"] == 7)
    check("颜色编号全表无重复",
          len(set(comm.COLOR_ID.values())) == len(comm.COLOR_ID),
          str(comm.COLOR_ID))
    check("颜色编号与 config.COLOR_IDS 完全一致",
          comm.COLOR_ID == C.COLOR_IDS, str(comm.COLOR_ID))
    check("每个 HSV 区间都有编号（不许漏配）",
          set(C.COLOR_RANGES_HSV) <= set(C.COLOR_IDS),
          str(set(C.COLOR_RANGES_HSV) - set(C.COLOR_IDS)))
    check("magenta 已删除", "magenta" not in comm.COLOR_ID
          and "magenta" not in C.COLOR_RANGES_HSV)
    check("未知颜色 -> 255",
          comm.build_frame("YZHU", "nope", 0, 0)[3] == 255)
    check("未知形状 -> 255",
          comm.build_frame("nope", "green", 0, 0)[2] == 255)

    # int16 越界截断而不是抛异常
    big = comm.build_frame("YZHU", "green", 99999, -99999)
    check("坐标越界截断到 int16",
          int.from_bytes(big[4:6], "little", signed=True) == 32767 and
          int.from_bytes(big[6:8], "little", signed=True) == -32768)


# ============================ 2. 坐标帧 ============================
def test_pos_frame():
    fr = comm.build_pos_frame(123, -456)
    check("坐标帧长度=6", len(fr) == 6, len(fr))
    check("坐标帧帧头=0x07", fr[0] == 0x07)
    check("坐标帧 CRC 覆盖前 5 字节", comm._crc8(fr[0:5]) == fr[5])

    c = comm.Comm()
    c._rx = bytearray(fr)
    c._parse_rx()
    check("解析出坐标", c.latest_position == (123, -456), c.latest_position)
    check("position_seq=1", c.position_seq == 1)

    # poll_position 取走即清空
    check("poll_position 返回坐标", c.poll_position() == (123, -456))
    check("poll_position 取走后为空", c.poll_position() is None)

    # 分包：先给前 3 字节
    c = comm.Comm()
    c._rx = bytearray(fr[:3])
    c._parse_rx()
    check("半包不误判", c.latest_position is None)
    c._rx.extend(fr[3:])
    c._parse_rx()
    check("分包到齐后解析成功", c.latest_position == (123, -456))

    # CRC 错
    chk = comm.Comm()
    bad = bytearray(comm.build_pos_frame(10, 20))
    bad[-1] ^= 0xFF
    chk._rx = bytearray(bad)
    chk._parse_rx()
    check("坏 CRC 被丢弃", chk.latest_position is None and chk.rx_bad >= 1,
          "bad=%d" % chk.rx_bad)

    # 与完成字节混流 + 杂字节
    mix = b"\x00\xFF" + comm.build_pos_frame(5, 6) + b"\x06" + \
        comm.build_pos_frame(-7, -8) + b"\xAB"
    cm = comm.Comm()
    cm._rx = bytearray(mix)
    cm._parse_rx()
    check("混流后坐标取最新", cm.latest_position == (-7, -8), cm.latest_position)
    check("混流后完成字节被识别", cm.done_seen is True)
    check("混流后 position_seq=2", cm.position_seq == 2, cm.position_seq)

    # x 值里含 0x06/0x07 也不能误判
    tricky = comm.build_pos_frame(0x0706, 0x0006)
    ct = comm.Comm()
    ct._rx = bytearray(tricky)
    ct._parse_rx()
    check("payload 含 0x06/0x07 仍能正确解析",
          ct.latest_position == (0x0706, 0x0006), ct.latest_position)


# ============================ 3. 完成字节 1/2 字节 ============================
def test_done_paths():
    old = C.GRAB_STATUS_ENABLE
    try:
        C.GRAB_STATUS_ENABLE = False
        c = comm.Comm()
        c._rx = bytearray(b"\x06")
        c._parse_rx()
        seen, st = c.poll_done_status()
        check("1 字节模式：poll_done_status 只给 seen", (seen, st) == (True, None))

        c = comm.Comm()
        c._rx = bytearray(b"\x06")
        c._parse_rx()
        check("1 字节模式：poll_done 为 True", c.poll_done() is True)
        check("完成字节消费后不重复触发", c.poll_done() is False)

        C.GRAB_STATUS_ENABLE = True
        for status, label in ((0x00, "成功"), (0x01, "空抓"),
                              (0x02, "急停"), (0x03, "其它")):
            c = comm.Comm()
            c._rx = bytearray(bytes([0x06, status]))
            c._parse_rx()
            seen, st = c.poll_done_status()
            check("2 字节模式 status=0x%02X(%s)" % (status, label),
                  seen is True and st == status, (seen, st))

        # 非法 status（0x07 是坐标帧头）不能吞掉下一帧
        c = comm.Comm()
        c._rx = bytearray(bytes([0x06]) + comm.build_pos_frame(1, 2))
        c._parse_rx()
        seen, st = c.poll_done_status()
        check("2 字节模式遇到非法 status 只当 1 字节完成",
              seen is True and st is None, (seen, st))
        check("非法 status 后坐标帧仍被解析",
              c.poll_position() == (1, 2), c.latest_position)
    finally:
        C.GRAB_STATUS_ENABLE = old


# ============================ 4. hand_eye 加载/回退/补偿 ============================
def test_hand_eye():
    tmp = tempfile.mkdtemp(prefix="he_test_")
    good = os.path.join(tmp, "good.json")
    hand_eye.save_json(good, [[0.05, 0.0, 10.0], [0.0, -0.05, 20.0]],
                       mean=1.5, mx=2.5, extra={"n_points": 54})
    with open(good, "r", encoding="utf-8") as f:
        data = json.load(f)
    check("JSON 里 matrix 是 2x3", np.array(data["matrix"]).shape == (2, 3))
    check("JSON 含残差与点数",
          data["mean_residual"] == 1.5 and data["n_points"] == 54)

    old_env = os.environ.get("GC_HAND_EYE_JSON")
    try:
        os.environ["GC_HAND_EYE_JSON"] = good
        hand_eye.reload()
        check("env 指定 JSON 后被识别为已标定", hand_eye.CALIBRATED is True)
        check("来源就是该 JSON", hand_eye.SOURCE == good, hand_eye.SOURCE)
        x, y, z = hand_eye.pixel_to_robot(100, 200)
        check("坐标按新矩阵计算", (x, y) == (15.0, 10.0), (x, y))

        # 平移补偿
        hand_eye.apply_probe(2.0, -3.0, good)
        hand_eye.reload()
        x, y, _ = hand_eye.pixel_to_robot(100, 200)
        check("--probe 只改平移", (x, y) == (17.0, 7.0), (x, y))

        # 坏 JSON 回退
        broken = os.path.join(tmp, "broken.json")
        with open(broken, "w", encoding="utf-8") as f:
            f.write('{"matrix": [[1,2]]}')
        os.environ["GC_HAND_EYE_JSON"] = broken
        hand_eye.reload()
        check("坏 JSON 不会崩，回退占位值", hand_eye.CALIBRATED is False)
        check("回退后矩阵等于 config 占位值",
              np.allclose(hand_eye.MATRIX, np.array(C.HAND_EYE_M)))
    finally:
        if old_env is None:
            os.environ.pop("GC_HAND_EYE_JSON", None)
        else:
            os.environ["GC_HAND_EYE_JSON"] = old_env
        hand_eye.reload()


# ============================ 5. 标定数学 ============================
def test_math():
    # 已知仿射：px -> mm
    M_true = np.array([[0.25, 0.02, -30.0],
                       [-0.01, -0.3, 150.0]])
    px = np.array([[10, 20], [300, 20], [10, 400], [300, 400],
                   [150, 210], [60, 330], [250, 90]], dtype=np.float64)
    A = np.hstack([px, np.ones((len(px), 1))])
    mm = (M_true @ A.T).T
    M, res, mean, mx = hand_eye.solve(px, mm)
    check("solve 精确还原已知矩阵", np.allclose(M, M_true, atol=1e-9),
          "%s" % (M - M_true,))
    check("solve 残差为 0", mx < 1e-9, mx)

    # 3 点时残差恒 0（这正是要点：看不出质量）
    M3, _r3, _m3, mx3 = hand_eye.solve(px[:3], mm[:3])
    check("3 点残差恒为 0（说明点数少时残差无意义）", mx3 < 1e-9)
    check(">=4 点才有读数", mx >= 0.0)

    # 手写点对文件求解
    tmp = tempfile.mkdtemp(prefix="pts_")
    csv_path = os.path.join(tmp, "points.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("u,v,x_mm,y_mm\n")
        for (u, v), (x, y) in zip(px, mm):
            f.write("%g,%g,%.6f,%.6f\n" % (u, v, x, y))
    p2, r2 = hand_eye.load_points(csv_path)
    check("CSV 读点数一致", len(p2) == len(px))
    M4, _r, _m, mx4 = hand_eye.solve(p2, r2)
    check("CSV 求解与内存求解一致", np.allclose(M4, M_true, atol=1e-6), mx4)

    # 对齐矩阵
    v = planar_calib.align_matrix(0, "none") @ np.array([1.0, 2.0, 1.0])
    check("align 恒等", np.allclose(v, [1, 2, 1]))
    v = planar_calib.align_matrix(0, "x") @ np.array([1.0, 2.0, 1.0])
    check("mirror=x 只翻 X", np.allclose(v, [-1, 2, 1]))
    v = planar_calib.align_matrix(180, "none") @ np.array([1.0, 2.0, 1.0])
    check("rot=180 两轴同时反向", np.allclose(v, [-1, -2, 1]))
    v = planar_calib.align_matrix(0, "xy") @ np.array([1.0, 2.0, 1.0])
    check("mirror=xy 等效 rot180", np.allclose(v, [-1, -2, 1]))

    # compose：先像素->棋盘，再棋盘->龙门
    inner = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    outer = planar_calib.align_matrix(0, "y")
    Mc = planar_calib.compose(outer, inner)
    p = Mc @ np.array([100.0, 50.0, 1.0])
    check("compose 结果 = 先inner后outer",
          np.allclose(p, [10.0, -5.0]), p)

    # 平移补偿不改比例
    M5 = planar_calib.shift_translation(M_true, 7.0, -9.0)
    check("shift 只动第 3 列",
          np.allclose(M5[:, :2], M_true[:, :2]) and
          np.allclose(M5[:, 2], M_true[:, 2] + [7.0, -9.0]))

    # 残差报告
    rep = hand_eye.residual_report(1.0, 2.0, 54)
    check("残差报告判定达标", "达标" in rep and "未达标" not in rep, rep)
    rep = hand_eye.residual_report(9.0, 20.0, 54)
    check("残差报告判定未达标", "未达标" in rep, rep)
    rep = hand_eye.residual_report(0.0, 0.0, 3)
    check("3 点时报告提示点数不足", "点数" in rep, rep)


def test_aligned_residual():
    """回归：加 --mirror/--anchor 后残差不能被算成垃圾。

    历史 bug：残差用「对齐后的 M」去比「棋盘 obj」。M 输出龙门坐标、
    obj 是棋盘坐标，一旦加了对齐两者不是同一坐标系，残差会变成 125mm 这种假数据。
    """
    if cv2 is None:
        print("   (跳过：没装 opencv)")
        return
    cols, rows, square, sq = 9, 6, 25.0, 24
    board = planar_calib._render_board(cols, rows, sq)
    G = np.array([[1.15, 0.0, 80.0], [0.0, 1.15, 60.0], [0.0, 0.0, 1.0]])
    W = G @ np.diag([square / sq, square / sq, 1.0])
    canvas = cv2.warpPerspective(board, W, (760, 620),
                                 flags=cv2.INTER_LINEAR, borderValue=255)
    tmp = tempfile.mkdtemp(prefix="cal_")
    path = os.path.join(tmp, "board.png")
    cv2.imencode(".png", canvas)[1].tofile(path)

    plain, px, obj, _ = planar_calib.calibrate_from_image(path)
    check("无对齐时残差达标", plain["mean_residual"] < 1.0, plain["mean_residual"])
    check("尺度自检给出 cell_px(>15)",
          plain.get("cell_px", 0) > 15, plain.get("cell_px"))

    mir, _px2, _o2, _ = planar_calib.calibrate_from_image(path, mirror="y")
    check("mirror 后残差仍达标（回归点）",
          mir["mean_residual"] < 1.0, mir["mean_residual"])
    check("mirror=y 只翻第二行",
          np.allclose(mir["matrix"][0], plain["matrix"][0], atol=1e-4) and
          np.allclose(mir["matrix"][1], -np.array(plain["matrix"][1]), atol=1e-4),
          (mir["matrix"], plain["matrix"]))

    u, v = float(px[0][0]), float(px[0][1])
    anc, _px3, _o3, _ = planar_calib.calibrate_from_image(
        path, anchors=[(u, v, 500.0, 300.0)])
    g = np.array(anc["matrix"]) @ np.array([u, v, 1.0])
    check("锚点把角点 0 钉到 (500,300)",
          abs(g[0] - 500.0) < 1e-6 and abs(g[1] - 300.0) < 1e-6, g)
    check("锚点后残差仍达标", anc["mean_residual"] < 1.0, anc["mean_residual"])

    # ---- 方案B：STM32 回传坐标 -> 3 个锚点自动解旋转/手性 ----
    th = np.deg2rad(-7.5)
    A_true = np.array([[np.cos(th), -np.sin(th), 260.0],
                       [np.sin(th), np.cos(th), 140.0]])
    M_true = planar_calib.compose(A_true, np.array(plain["matrix"]))
    idxs = [0, 8, 53]
    anchors = []
    for i in idxs:
        uu, vv = float(px[i][0]), float(px[i][1])
        p = M_true @ np.array([uu, vv, 1.0])
        anchors.append((uu, vv, float(p[0]), float(p[1])))
    multi, _px4, _o4, _ = planar_calib.calibrate_from_image(path, anchors=anchors)
    check("3 锚点还原真实变换（含旋转）",
          np.abs(np.array(multi["matrix"]) - M_true).max() < 0.05,
          np.abs(np.array(multi["matrix"]) - M_true).max())
    check("3 锚点解出比例 ≈1", abs(multi.get("anchor_scale", 0) - 1.0) < 1e-3,
          multi.get("anchor_scale"))
    check("3 锚点判定不镜像", multi.get("anchor_mirror") is False,
          multi.get("anchor_mirror"))

    # 两锚点也能量出旋转，但必须如实说明"手性判不了"
    two, _px5, _o5, _ = planar_calib.calibrate_from_image(path, anchors=anchors[:2])
    check("2 锚点仍能对齐（手性按不镜像处理）",
          np.abs(np.array(two["matrix"]) - M_true).max() < 0.05,
          np.abs(np.array(two["matrix"]) - M_true).max())


def test_camera_model():
    """去畸变：直通/正常/坏内参/一致性告警。"""
    # ---- 用零畸变内参：点必须原地不动（防止把"启用"写成"乱改坐标"）----
    tmp = tempfile.mkdtemp(prefix="cam_")
    p0 = os.path.join(tmp, "zero.json")
    with open(p0, "w", encoding="utf-8") as f:
        json.dump({"mtx": [[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1]],
                   "dist": [0, 0, 0, 0, 0]}, f)
    old = camera_model.candidate_paths
    try:
        camera_model.candidate_paths = lambda: [p0]
        camera_model.reload()
        check("零畸变内参：点原地不动",
              np.allclose(camera_model.undistort_points([[100, 200], [500, 400]]),
                          [[100, 200], [500, 400]], atol=1e-6),
              camera_model.undistort_points([[100, 200]]))

        # ---- k1>0（枕形，OpenCV 约定 x_d = x(1+k1·r²)）：
        #      观测到的边缘像素对应的理想位置更靠近中心，修正后径向应当【变小】。
        #      项目里 C100 那组内参就是这个方向（四角从 428px 收到 225px）。
        p1 = os.path.join(tmp, "k1pos.json")
        with open(p1, "w", encoding="utf-8") as f:
            json.dump({"mtx": [[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1]],
                       "dist": [0.3, 0.0, 0.0, 0.0, 0.0]}, f)
        camera_model.candidate_paths = lambda: [p1]
        camera_model.reload()
        u0, v0 = camera_model.undistort_point(620, 460)     # 右下角
        r_in = np.hypot(620 - 320, 460 - 240)
        r_out = np.hypot(u0 - 320, v0 - 240)
        check("k1>0：修正后径向变小（往中心收）", r_out < r_in - 1.0,
              (r_in, r_out))

        # ---- k1<0（桶形）：方向必须相反 ----
        # 用 -0.08 而不是 -0.3：后者在画面内就已经"折叠"（正畸变不再单调），
        # 那种内参正是 is_model_sane 要拦下来的，不该拿来测"修正方向"。
        p1b = os.path.join(tmp, "k1neg.json")
        with open(p1b, "w", encoding="utf-8") as f:
            json.dump({"mtx": [[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1]],
                       "dist": [-0.08, 0.0, 0.0, 0.0, 0.0]}, f)
        camera_model.candidate_paths = lambda: [p1b]
        camera_model.reload()
        check("k1<0 内参能启用（画面内未折叠）", camera_model.is_active() is True,
              camera_model.NOTE)
        u1, v1 = camera_model.undistort_point(100, 200)
        r_neg_in = np.hypot(100 - 320, 200 - 240)
        check("k1<0：修正后径向变大（往外扩）",
              np.hypot(u1 - 320, v1 - 240) > r_neg_in + 1.0,
              (r_neg_in, np.hypot(u1 - 320, v1 - 240)))

        check("中心附近几乎不动",
              np.hypot(*np.subtract(camera_model.undistort_point(322, 242),
                                    (322, 242))) < 0.5,
              camera_model.undistort_point(322, 242))

        # ---- 坏内参：不能崩，退化直通 ----
        p2 = os.path.join(tmp, "bad.json")
        with open(p2, "w", encoding="utf-8") as f:
            json.dump({"mtx": [[0, 0, 0], [0, 0, 0], [0, 0, 1]], "dist": [0, 0, 0, 0]}, f)
        camera_model.candidate_paths = lambda: [p2]
        camera_model.reload()
        check("坏内参(fx=0)：不启用而不是崩", camera_model.is_active() is False)
        check("未启用时像素直通",
              np.allclose(camera_model.undistort_points([[1, 2]]), [[1, 2]]))

        # ---- 文件缺失 ----
        camera_model.candidate_paths = lambda: [os.path.join(tmp, "nope.json")]
        camera_model.reload()
        check("内参缺失：不启用", camera_model.is_active() is False)
        check("未启用时 max_correction=0", camera_model.max_correction_px() == 0.0)
    finally:
        camera_model.candidate_paths = old
        camera_model.reload()

    # ---- 一致性检查：标定/推理两边必须同口径 ----
    want = camera_model.is_active()
    check("一致性：两边同口径 -> 通过",
          camera_model.check_hand_eye_consistency({"undistorted": want}) is True)
    check("一致性：口径相反 -> 报警",
          camera_model.check_hand_eye_consistency({"undistorted": not want}) is False)
    check("一致性：旧版 json 无字段 -> 不判断",
          camera_model.check_hand_eye_consistency({}) is None)


def test_intrinsic_quality():
    """内参质量闸：真内参可信、坏内参（边缘折叠）必须被拒。

    2026-10-26 重标之前仓库里那套内参就是"折叠"的：r=1.0 处正逆差 379px，
    undistortPoints 给的是垃圾坐标。所以除了能算出数，还要能自己发现数算坏了。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = os.path.join(root, "camera_intrinsic.json")
    old = camera_model.candidate_paths
    if not os.path.isfile(real):                     # 板端 tools/ 是单独拷的
        check("没有 camera_intrinsic.json -> 跳过质量检查", True)
        return
    try:
        camera_model.candidate_paths = lambda: [real]
        camera_model.reload()
        check("真内参：能加载并启用", camera_model.is_active() is True,
              camera_model.NOTE)
        check("真内参：模型=no_tangent", camera_model.MODEL == "no_tangent",
              camera_model.MODEL)
        d = np.asarray(camera_model.D, dtype=float).ravel()
        check("真内参：切向项 p1=p2=0", abs(d[2]) < 1e-9 and abs(d[3]) < 1e-9,
              d.tolist())
        check("真内参：模型自检通过", camera_model.is_model_sane()[0] is True,
              camera_model.is_model_sane())
        newK = np.array(camera_model.K, dtype=float)
        newD = np.array(camera_model.D, dtype=float).ravel()
        check("真内参：径向单调", camera_model.radial_is_monotonic() is True)

        # 画面四角的修正量：太大说明模型夸张，太小说明没生效
        mc = camera_model.max_correction_px()
        check("真内参：边缘修正量在 40~120px 之间", 40.0 < mc < 120.0, mc)

        # 等半径修正量必须各向同性（纯径向模型，极差 <0.2px）
        vals = []
        for ang in range(0, 360, 45):
            a = np.radians(ang)
            u = 320.365 + 0.5 * 314.885 * np.cos(a)
            v = 255.356 + 0.5 * 317.042 * np.sin(a)
            uu, vv = camera_model.undistort_point(u, v)
            vals.append(float(np.hypot(uu - u, vv - v)))
        check("真内参：r=0.5 修正量各向同性", max(vals) - min(vals) < 0.2,
              [round(x, 2) for x in vals])

        # 覆盖区判定
        check("覆盖：主点附近在覆盖内",
              camera_model.is_covered(320, 255) is True)
        check("覆盖：画面正上方 235px 处 = 外推",
              camera_model.is_covered(320, 20) is False)
        check("覆盖：远角等级=far", camera_model.check_point(5, 5)[0] == "far",
              camera_model.check_point(5, 5))
        check("覆盖：主点等级=ok", camera_model.check_point(320, 255)[0] == "ok")

        # ---- 折叠模型必须被拦下（旧内参原样）----
        tmp = tempfile.mkdtemp(prefix="camq_")
        fold = os.path.join(tmp, "fold.json")
        with open(fold, "w", encoding="utf-8") as f:
            json.dump({"mtx": [[394.607, 0, 292.638], [0, 374.271, 228.429],
                               [0, 0, 1]],
                       "dist": [0.898878, -1.972433, -0.001387, -0.100980,
                                2.245554]}, f)
        camera_model.candidate_paths = lambda: [fold]
        camera_model.reload()
        check("折叠内参：拒绝启用（不拿垃圾坐标去算毫米）",
              camera_model.is_active() is False, camera_model.NOTE)
        check("折叠内参：说明里点出模型不合理",
              "不合理" in (camera_model.NOTE or ""), camera_model.NOTE)
        check("折叠内参：未启用时像素直通",
              np.allclose(camera_model.undistort_points([[0, 0]]), [[0, 0]]))

        # ---- 标定工具必须用同一把尺子（否则 planar_calib 会拿垃圾内参标手眼）----
        check("planar_calib 自检：新内参通过",
              planar_calib.intrinsic_selfcheck(newK, newD, (640, 480))[0] is True,
              planar_calib.intrinsic_selfcheck(newK, newD, (640, 480)))
        with open(fold, encoding="utf-8") as f:
            bad = json.load(f)
        ok_bad, why_bad = planar_calib.intrinsic_selfcheck(bad["mtx"], bad["dist"])
        check("planar_calib 自检：旧(折叠)内参被拒", ok_bad is False, why_bad)
    finally:
        camera_model.candidate_paths = old
        camera_model.reload()


def test_safety():
    """安全闸：没标定 / 串口掉了 必须拒绝下发（这是防事故的最后一道）。"""
    # 未标定 -> 拒绝
    r = safety.check_downlink(False)
    check("安全闸：未标定 -> 拒绝", r is not None)
    check("安全闸：拒绝原因=UNCALIBRATED",
          r is not None and r[0] == safety.REASON_UNCALIBRATED)
    # 已标定 -> 放行
    check("安全闸：已标定+串口正常 -> 放行", safety.check_downlink(True) is None)
    # 显式绕过
    check("安全闸：allow_uncalibrated=true 可绕过标定闸",
          safety.check_downlink(False, allow_uncalibrated=True) is None)
    # 串口掉线
    r = safety.check_downlink(True, serial_ok=False)
    check("安全闸：串口掉了 -> 拒绝",
          r is not None and r[0] == safety.REASON_SERIAL_DOWN)
    check("安全闸：serial_sim 是主动选的，不算串口故障",
          safety.check_downlink(True, serial_ok=False, serial_sim=True) is None)
    # 两个都坏时先报标定（更严重的那个）
    r = safety.check_downlink(False, serial_ok=False)
    check("安全闸：两个都坏 -> 先报标定",
          r is not None and r[0] == safety.REASON_UNCALIBRATED)
    # 原因码有人话
    check("安全闸：原因码可翻译",
          "标定" in safety.human(safety.REASON_UNCALIBRATED))
    # 启动横幅（返回 (level, 文本)，level 决定用哪一行 logger 调用）
    lines = safety.describe_gates(True, False, False, False, False)
    check("启动横幅：未标定+串口故障 -> 两条都带🔒",
          len(lines) == 2 and all("🔒" in s for _l, s in lines))
    check("启动横幅：两条都该用 warn 级别",
          len(lines) == 2 and all(lv == "warn" for lv, _s in lines))
    lines = safety.describe_gates(True, False, True, True, False)
    check("启动横幅：全部就绪 -> 两条都带✅",
          len(lines) == 2 and all("✅" in s for _l, s in lines))
    check("启动横幅：全部就绪 -> 两条都该用 info 级别",
          len(lines) == 2 and all(lv == "info" for lv, _s in lines))
    check("启动横幅：台架绕过要显式提示",
          any("绕过" in s for _l, s in
              safety.describe_gates(True, True, False, True, False)))
    # 混合级别：节点必须能分别走两个固定调用点（回归 rclpy 的 severity 缓存）
    mixed = dict((lv, s) for lv, s in
                 safety.describe_gates(True, True, False, True, True))
    check("启动横幅：混合级别时两种 level 都存在",
          set(mixed) == {"warn", "info"})


def test_hsv_calib():
    """HSV 标定助手：H 的周期是 180（不是 360），跨 0° 必须处理对。"""
    if cv2 is None:
        print("  (没装 opencv，跳过)")
        return
    try:
        import hsv_calib as HC
    except ImportError:
        print("  (找不到 hsv_calib.py，跳过)")
        return

    lo, hi, wrapped = HC.circ_range(np.array([115, 116, 117], np.uint8))
    check("H 圆统计：115 附近 -> 区间包住 115、不跨 0°",
          lo <= 115 <= hi and not wrapped, "got %s" % ((lo, hi, wrapped),))

    lo, hi, wrapped = HC.circ_range(np.array([177, 179, 0, 1, 3], np.uint8))
    check("H 圆统计：红色跨 0° -> wrapped 且 lo>hi",
          wrapped and lo > hi, "got %s" % ((lo, hi, wrapped),))
    # 跨 0° 的区间要能正确判断"和自身完全重叠"
    check("H 重叠：跨 0° 区间和自己重叠 > 6°",
          HC.overlap_h((170, 10, True), (170, 10, True)) > 6)
    check("H 重叠：绿色与蓝色不重叠",
          HC.overlap_h((40, 85, False), (100, 125, False)) == 0)

    # 端到端：合成图片 -> 建议阈值 -> 能盖住自己的色相
    import tempfile
    import json as _json
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"green": 60, "blue": 115}
        for name, h in spec.items():
            d = os.path.join(tmp, name)
            os.makedirs(d, exist_ok=True)
            img = np.zeros((480, 640, 3), np.uint8)
            img[140:340, 190:450] = (h, 200, 140)
            cv2.imwrite(os.path.join(d, "a.png"),
                        cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
        colors = HC.collect_pixels(tmp)
        check("合成数据：两种颜色都读到", set(colors) == {"green", "blue"})
        sugg = HC.suggest(colors)
        ok = True
        for name, h in spec.items():
            lo, hi, wrapped = sugg[name]["h"]
            if wrapped:
                ok = ok and (h >= lo or h <= hi)
            else:
                ok = ok and (lo <= h <= hi)
        check("合成数据：建议区间盖住真实色相", ok)
        check("合成数据：颜色互不冲突", HC.conflicts(sugg) == [])
        snippet = HC.config_snippet(sugg)
        check("config 片段可直接粘贴（含字典头尾）",
              snippet.startswith("COLOR_RANGES_HSV = {") and
              snippet.rstrip().endswith("}"))


def test_grasp_refine():
    """抓取点精修：用合成图验证"轮廓质心 vs 框中心"的贴合精度。"""
    if cv2 is None:
        print("skip (no cv2)")
        return
    import grasp_refine as GR

    def hsv_bgr(h, s, v):
        return tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])

    GREEN = hsv_bgr(60, 200, 200)
    BLUE = hsv_bgr(110, 200, 200)
    WHITE = (255, 255, 255)

    def blank():
        return np.zeros((200, 200, 3), np.uint8)

    # --- 1. 实心圆：质心/贴合框/填充率 ---
    img = blank()
    cv2.circle(img, (100, 101), 20, GREEN, -1)
    r = GR.refine(img, [72, 73, 128, 129], "green")
    check("精修：圆成功", r["ok"], r["reason"])
    err = ((r["point"][0] - 100) ** 2 + (r["point"][1] - 101) ** 2) ** 0.5
    check("精修：圆质心误差<0.7px", err < 0.7, "err=%.2f" % err)
    tb = r["tight_bbox"]
    check("精修：贴合框≈真实外接框(±2px)",
          abs(tb[0] - 80) <= 2 and abs(tb[1] - 81) <= 2 and
          abs(tb[2] - 120) <= 2 and abs(tb[3] - 121) <= 2, str(tb))
    check("精修：圆填充率≈πr²/框面积",
          abs(r["fill"] - 3.1416 * 400 / (56 * 56)) < 0.06,
          "fill=%.3f" % r["fill"])
    check("精修：等效半径≈20px", abs(r["radius"] - 20) <= 2, "r=%.1f" % r["radius"])

    # --- 2. 框故意偏心：框中心偏 6px，精修要拉回真值 ---
    img = blank()
    cv2.circle(img, (100, 101), 20, GREEN, -1)
    r2 = GR.refine(img, [60, 73, 128, 129], "green")
    box_cx = (60 + 128) / 2.0
    e_box = abs(box_cx - 100)
    e_ref = abs(r2["point"][0] - 100)
    check("精修：偏心框下框中心本来偏 6px", abs(e_box - 6) < 0.01, "%.1f" % e_box)
    check("精修：把偏心纠回真值(误差<1px)", e_ref < 1.0, "e=%.2f" % e_ref)

    # --- 3. 高光孔洞不能把质心拉走 ---
    img = blank()
    cv2.circle(img, (100, 101), 20, GREEN, -1)
    cv2.circle(img, (94, 96), 4, WHITE, -1)      # 高光
    r3 = GR.refine(img, [72, 73, 128, 129], "green")
    err3 = ((r3["point"][0] - 100) ** 2 + (r3["point"][1] - 101) ** 2) ** 0.5
    check("精修：高光孔洞下质心误差<1px", err3 < 1.0, "err=%.2f" % err3)

    # --- 4. 框内另一个小同色噪点：只能取最大轮廓 ---
    img = blank()
    cv2.circle(img, (100, 101), 20, GREEN, -1)
    cv2.rectangle(img, (74, 74), (80, 80), GREEN, -1)     # 角落噪点
    r4 = GR.refine(img, [72, 73, 128, 129], "green")
    err4 = ((r4["point"][0] - 100) ** 2 + (r4["point"][1] - 101) ** 2) ** 0.5
    check("精修：忽略框内小噪点", err4 < 1.0, "err=%.2f" % err4)

    # --- 5. 颜色对不上 -> 回退框中心 ---
    img = blank()
    cv2.circle(img, (100, 101), 20, BLUE, -1)
    r5 = GR.refine(img, [72, 73, 128, 129], "green")
    check("精修：颜色不符时标记失败", not r5["ok"], r5["reason"])
    check("精修：失败时退回框中心",
          abs(r5["point"][0] - 100) < 0.01 and abs(r5["point"][1] - 101) < 0.01,
          str(r5["point"]))

    # --- 6. 太小 -> 回退 ---
    img = blank()
    cv2.circle(img, (100, 101), 2, GREEN, -1)
    r6 = GR.refine(img, [72, 73, 128, 129], "green")
    check("精修：物体太小按回退处理", not r6["ok"], r6["reason"])

    # --- 7. 框贴边/越界不崩 ---
    img = blank()
    cv2.circle(img, (8, 8), 6, GREEN, -1)
    r7 = GR.refine(img, [-6, -6, 22, 22], "green")
    check("精修：越界框不崩且仍有有效点",
          isinstance(r7["point"], tuple) and len(r7["point"]) == 2, str(r7["point"]))

    # --- 8. 红色跨 0°：两段区间都要命中 ---
    img = blank()
    cv2.rectangle(img, (20, 20), (60, 60), hsv_bgr(5, 200, 200), -1)
    cv2.rectangle(img, (120, 120), (160, 160), hsv_bgr(175, 200, 200), -1)
    m = GR.color_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), "red")
    check("精修：红色跨 0° 两段都能分割",
          int(np.count_nonzero(m[20:61, 20:61])) > 1000 and
          int(np.count_nonzero(m[120:161, 120:161])) > 1000,
          "first=%d second=%d" % (
              int(np.count_nonzero(m[20:61, 20:61])),
              int(np.count_nonzero(m[120:161, 120:161]))))

    # --- 9. rect 模式取最小外接矩形中心 ---
    img = blank()
    cv2.rectangle(img, (70, 90), (130, 112), GREEN, -1)   # 60x22 横条
    r9 = GR.refine(img, [64, 84, 136, 118], "green", point_mode="rect")
    check("精修：rect 模式中心≈(100,101)",
          abs(r9["point"][0] - 100) < 1.0 and abs(r9["point"][1] - 101) < 1.0,
          str(r9["point"]))
    check("精修：rect 模式尺寸≈60x22",
          abs(r9["rect_size"][0] - 60) <= 2 and abs(r9["rect_size"][1] - 22) <= 2,
          str(r9["rect_size"]))

    # --- 10. bbox 模式等价于不精修 ---
    img = blank()
    cv2.circle(img, (100, 101), 20, GREEN, -1)
    r10 = GR.refine(img, [60, 73, 128, 129], "green", point_mode="bbox")
    check("精修：bbox 模式返回框中心",
          r10["point"] == ((60 + 128) / 2.0, (73 + 129) / 2.0), str(r10["point"]))

    # --- 11. describe 两种分支都不抛异常 ---
    ok_desc = GR.describe(r)
    fb_desc = GR.describe(r5)
    check("精修：describe 成功/回退都能出字符串",
          isinstance(ok_desc, str) and "精修[" in ok_desc and
          isinstance(fb_desc, str) and "回退" in fb_desc,
          ok_desc + " | " + fb_desc)

    # --- 12. 关掉精修时 config 开关存在且为真 ---
    check("精修：config.GRASP_REFINE 存在",
          hasattr(C, "GRASP_REFINE"), "缺 GRASP_REFINE")
    check("精修：默认模式是 centroid",
          getattr(C, "GRASP_POINT_MODE", "") == "centroid",
          getattr(C, "GRASP_POINT_MODE", ""))


def main():
    print("=" * 70)
    print("离线单测：协议 + 标定链路")
    print("=" * 70)
    for fn in (test_downlink, test_pos_frame, test_done_paths,
               test_hand_eye, test_math, test_aligned_residual,
               test_camera_model, test_intrinsic_quality,
               test_safety, test_hsv_calib, test_grasp_refine):
        print("\n--- %s ---" % fn.__name__)
        try:
            fn()
        except Exception as e:
            FAIL.append(fn.__name__)
            import traceback
            traceback.print_exc()
            print("FAIL %s 抛异常: %s" % (fn.__name__, e))
    print("\n" + "=" * 70)
    print("通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
    if FAIL:
        for n in FAIL:
            print("  FAILED: %s" % n)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())