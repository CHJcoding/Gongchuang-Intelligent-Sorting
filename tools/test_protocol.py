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
import planar_calib          # noqa: E402

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
    check("7 号预留（无人占用）", 7 not in comm.COLOR_ID.values())
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


def main():
    print("=" * 70)
    print("离线单测：协议 + 标定链路")
    print("=" * 70)
    for fn in (test_downlink, test_pos_frame, test_done_paths,
               test_hand_eye, test_math):
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