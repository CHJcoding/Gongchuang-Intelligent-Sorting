# -*- coding: utf-8 -*-
"""采集手眼标定锚点：STM32 回传坐标 -> 锚点文件（不用手抄数字）。

用法（板子上，标定时用；此时不要同时跑 sort_vision_node，串口是独占的）：

    # 1. 先拍一张棋盘格照片（见 BOARD_README 第八节）
    # 2. 采集 3 个锚点（把夹爪尖依次降到 3 个角点上方，每次回车）
    #    --port 省略即自动找 /dev/ttyACM0（STM32 是 USB CDC）
    python3 tools/collect_anchors.py --corners 0 8 53 \
            --image /root/board.jpg --out /root/vision_ws/anchors.json

    # 2'. 停顿自动采（推荐）：电机停住 hold_s 秒自动记，运动中/缓慢漂移的位姿丢弃
    python3 tools/collect_anchors.py --auto --corners 0 8 53 \
            --image /root/board.jpg --pose-log /root/vision_ws/poses.jsonl

    # 3. 解算（自动读锚点文件，旋转和手性都不用猜）
    python3 tools/planar_calib.py --image /root/board.jpg \
            --anchors-file /root/vision_ws/anchors.json

离线自测（不需要串口，不需要 STM32）：

    python3 tools/collect_anchors.py --simulate

离线复盘现场录的坐标（每行 t,x,y，或带 x/y 的 JSONL）：

    python3 tools/collect_anchors.py --replay poses.csv --hold-s 1.5 --tol-mm 0.6

为什么需要 3 个点：
  1 个点只能修平移；2 个点能修旋转但分辨不了手性（是否镜像）；
  3 个分散的点能同时解出 比例/旋转/平移/手性 —— 这时 --rot/--mirror 就完全不用管了。
  选角点要尽量分散（如 0 / 8 / 53，即两个对角 + 一个异侧点），三点共线会解不出旋转。
"""
import argparse
import io
import json
import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (os.path.join(_PARENT, "rdk", "vision_sorter"),
           os.path.join(_PARENT, "vision_sorter"),
           "/root/vision_ws/vision_sorter", _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import config as C
except Exception:                                       # noqa: BLE001
    C = None

for _stream in (sys.stdout, sys.stderr):                # Windows 控制台 GBK 兜底
    try:
        _stream.reconfigure(errors="replace")
    except Exception:                                   # noqa: BLE001
        pass


def _cfg(name, default):
    return getattr(C, name, default) if C is not None else default


def _default_port():
    """默认串口：复用 comm.resolve_serial_port（by-id → config → 候选表）。

    STM32 是 USB CDC，设备号可能从 ttyACM0 漂到 ttyACM1，
    这里统一成同一个解析函数，免得工具和节点认定的端口不一样。
    """
    try:
        import comm
        return comm.resolve_serial_port() or _cfg("SERIAL_PORT", "/dev/ttyACM0")
    except Exception:                                   # noqa: BLE001
        return _cfg("SERIAL_PORT", "/dev/ttyACM0")


POS_HEAD = int(_cfg("POS_FRAME_HEAD", 0x07))
POS_LEN = int(_cfg("POS_FRAME_LEN", 6))

# 文本坐标帧（现场固件实测格式，2026-10-28）：
#   b"POS A:655279 B:3 X:-199.98 Y:-199.97 Adeg:-3599.56 Bdeg:-0.02 valid:3\r\n"
_TXT_X = re.compile(r"\bX:(-?\d+(?:\.\d+)?)")
_TXT_Y = re.compile(r"\bY:(-?\d+(?:\.\d+)?)")
_TXT_VALID = re.compile(r"\bvalid:(-?\d+)")
_TXT_A = re.compile(r"\bA:(-?\d+)")
_TXT_B = re.compile(r"\bB:(-?\d+)")
_TXT_ADEG = re.compile(r"\bAdeg:(-?\d+(?:\.\d+)?)")
_TXT_BDEG = re.compile(r"\bBdeg:(-?\d+(?:\.\d+)?)")


def crc8(data):
    """CRC-8/SMBUS，poly 0x07，init 0x00（与 comm._crc8 一致）。"""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_pos(x_mm, y_mm):
    """组一帧 0x07 坐标帧（自测用）。"""
    xi = max(-32768, min(32767, int(round(x_mm))))
    yi = max(-32768, min(32767, int(round(y_mm))))
    body = bytes([POS_HEAD,
                  xi & 0xFF, (xi >> 8) & 0xFF,
                  yi & 0xFF, (yi >> 8) & 0xFF])
    return body + bytes([crc8(body)])


class PosReader:
    """流式读坐标帧，只保留最新一帧（丢帧无所谓）。

    同时支持两种格式（现场固件两种都可能出现）：
      1) 文本行（2026-10-28 现场实测，当前固件就是这个）：
         b"POS A:655279 B:3 X:-199.98 Y:-199.97 Adeg:-3599.56 Bdeg:-0.02 valid:3\r\n"
      2) 二进制帧（协议备用格式）：0x07 | x int16 LE | y int16 LE | crc8
    文本帧里 valid<=0 表示没归零/不可信，此时不更新坐标（否则把 0,0 当真实点）。
    """

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0
        self.total = 0
        self.last = None            # (x_mm, y_mm) 浮点
        self.last_t = 0.0
        self.last_valid = None      # 固件报的 valid 值
        self.last_text = None       # 整条文本帧解析结果 dict
        self.invalid = 0            # valid<=0 被丢弃的条数

    # ---- 文本帧 ----
    def _consume_text(self, line):
        text = line.decode("ascii", "replace")
        mx = _TXT_X.search(text)
        my = _TXT_Y.search(text)
        if not mx or not my:
            self.bad += 1
            return
        mv = _TXT_VALID.search(text)
        valid = int(mv.group(1)) if mv else 1
        self.last_valid = valid

        def _num(pat, cast):
            m = pat.search(text)
            try:
                return cast(m.group(1)) if m else None
            except (TypeError, ValueError):
                return None

        self.last_text = {
            "x": float(mx.group(1)), "y": float(my.group(1)), "valid": valid,
            "a": _num(_TXT_A, int), "b": _num(_TXT_B, int),
            "adeg": _num(_TXT_ADEG, float), "bdeg": _num(_TXT_BDEG, float),
        }
        if valid <= 0:
            self.invalid += 1
            return
        # 刚上电/没回零时固件会发 valid:3 但坐标与编码器全 0（实测），挡住
        a_val = self.last_text.get("a")
        b_val = self.last_text.get("b")
        if (self.last_text["x"] == 0.0 and self.last_text["y"] == 0.0
                and a_val in (None, 0) and b_val in (None, 0)):
            self.invalid += 1
            self.last_valid = 0
            return
        self.last = (float(mx.group(1)), float(my.group(1)))
        self.last_t = time.time()
        self.total += 1

    def feed(self, data):
        self.buf.extend(data)
        if len(self.buf) > 8192:                 # 防噪声撑爆（文本行比二进制长）
            del self.buf[:-1024]
        while True:
            it = self.buf.find(b"POS ")
            ib = self.buf.find(bytes([POS_HEAD]))
            if it < 0 and ib < 0:
                self.buf.clear()
                return
            if it >= 0 and (ib < 0 or it <= ib):
                # ---- 文本行 ----
                nl = self.buf.find(b"\n", it)
                if nl < 0:
                    if it:
                        del self.buf[:it]
                    if len(self.buf) > 256:      # 没有换行的超长垃圾
                        self.bad += 1
                        del self.buf[:1]
                        continue
                    return                       # 半行，等后续字节
                line = bytes(self.buf[it:nl + 1])
                del self.buf[:nl + 1]
                self._consume_text(line)
                continue
            # ---- 二进制帧 ----
            if ib:
                del self.buf[:ib]
            if len(self.buf) < POS_LEN:
                return
            frame = bytes(self.buf[:POS_LEN])
            body = frame[:5]
            x = int.from_bytes(body[1:3], "little", signed=True)
            y = int.from_bytes(body[3:5], "little", signed=True)
            if crc8(body) != frame[5]:
                self.bad += 1
                del self.buf[:1]                     # 丢 1 字节重同步
                continue
            if x != -32768 and y != -32768:          # -32768 当"无效"约定
                self.last = (float(x), float(y))
                self.last_t = time.time()
                self.total += 1
            del self.buf[:POS_LEN]

    def age(self):
        return 0.0 if not self.last else time.time() - self.last_t


def _fmt_pos(pos):
    if not pos:
        return "(等待坐标...)"
    return "(x=%7.1f, y=%7.1f)" % (pos[0], pos[1])


def _median_pos(samples):
    """多帧取中位数，压掉抖动。"""
    if not samples:
        return None
    a = np.array(samples, dtype=np.float64)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


class HoldDetector:
    """从坐标流里挑出「停顿位姿」——电机停住不动的那几秒才算一个可用位姿。

    为什么需要它：标定时人把夹爪挪到角点上方，手会抖、电机会微动，
    靠「按回车」记录的时间点取决于手速，容易采到正在运动的坐标。
    改成看坐标流：位置在容差内静止够久，才认为这一帧的位姿可信。

    判定规则（全部可调）：
      1. 时间戳连续——相邻两帧间隔 > max_gap_s 视为断流，窗口重开
         （防止 STM32 掉线重连后，把两个不同位置拼进同一个窗口）。
      2. 窗口内所有点到窗口中位数的距离 <= tol_mm。
      3. 窗口时长 >= hold_s 且样本数 >= min_samples。
      4. 与上一个已接受位姿的距离 >= min_move_mm
         （同一个位置只报一次，防止一直停着被反复记录）。
    """

    def __init__(self, tol_mm=0.6, hold_s=1.5, min_samples=8,
                 min_move_mm=15.0, max_gap_s=1.0):
        self.tol_mm = float(tol_mm)
        self.hold_s = float(hold_s)
        self.min_samples = int(min_samples)
        self.min_move_mm = float(min_move_mm)
        self.max_gap_s = float(max_gap_s)
        self.reset()

    def reset(self):
        self._win = []            # [(t, x, y), ...] 当前静止窗口
        self._t_last = None
        self._accepted = []       # [(x, y), ...] 已接受过的位姿
        self.gaps = 0             # 断流次数（诊断用）
        self.dup = 0              # 因「没挪开」被丢掉的窗口数（诊断用）

    # ---------------- 窗口统计 ----------------
    def _center(self):
        a = np.array([[p[1], p[2]] for p in self._win], dtype=np.float64)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    def _spread(self, cx, cy):
        d = [math.hypot(p[1] - cx, p[2] - cy) for p in self._win]
        return float(max(d)) if d else 0.0

    def feed(self, t, x, y):
        """喂一帧 (时间戳秒, x_mm, y_mm)。凑够一个停顿位姿就返回事件 dict，否则 None。

        事件字段：t0/t1 窗口起止时间，x/y 窗口中位数，spread 窗口内最大偏离(mm)，
        n 样本数，moved_mm 相对上一个已接受的位姿移动了多少。
        """
        t, x, y = float(t), float(x), float(y)
        if self._t_last is not None and (t - self._t_last) > self.max_gap_s:
            self.gaps += 1
            self._win = []
        self._t_last = t

        if not self._win:
            self._win = [(t, x, y)]
            return None

        cx, cy = self._center()
        if math.hypot(x - cx, y - cy) > self.tol_mm:
            # 还在动：窗口从当前点重新开始
            self._win = [(t, x, y)]
            return None

        self._win.append((t, x, y))
        if len(self._win) < self.min_samples:
            return None
        if (t - self._win[0][0]) < self.hold_s:
            return None

        cx, cy = self._center()
        spread = self._spread(cx, cy)
        if spread > self.tol_mm:
            # 窗口里慢慢漂了：丢掉前半段，用后半段重等（不至于永远采不到）
            self._win = self._win[len(self._win) // 2:]
            return None

        moved = None
        if self._accepted:
            moved = min(math.hypot(cx - ax, cy - ay) for (ax, ay) in self._accepted)
            if moved < self.min_move_mm:
                self.dup += 1
                self._win = []
                return None

        ev = {"t0": self._win[0][0], "t1": t, "x": cx, "y": cy,
              "spread": spread, "n": len(self._win),
              "moved_mm": moved}
        self._accepted.append((cx, cy))
        self._win = []
        return ev

    def set_accepted(self, pts):
        """撤销锚点时同步内部状态，免得刚撤销的位置又被当成「重复」丢掉。"""
        self._accepted = [(float(p[0]), float(p[1])) for p in pts]
        self._win = []


def _out_path(args):
    """锚点文件路径：命令行 > 板端约定 > 仓库根。"""
    if getattr(args, "out", ""):
        return args.out
    if "vision_ws" in os.path.abspath(_HERE):        # 板端
        return "/root/vision_ws/anchors.json"
    return os.path.join(os.path.dirname(_PARENT), "anchors.json")


def _write_out(path, anchors, image, extra=None):
    data = {"image": image,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "anchors": [{"idx": int(i), "x": round(float(x), 2), "y": round(float(y), 2)}
                        for (i, x, y) in anchors]}
    if extra:
        data.update(extra)
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _summary(anchors):
    print("\n" + "=" * 62)
    print("采集完成，共 %d 个锚点：" % len(anchors))
    for (i, x, y) in anchors:
        print("  角点 #%-3d -> 龙门 (%.1f, %.1f) mm" % (i, x, y))
    if len(anchors) >= 2:
        p = np.array([[a[1], a[2]] for a in anchors], dtype=np.float64)
        spread = float(np.max(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)))
        print("   锚点最大间距 %.1f mm" % spread)
        if spread < 50:
            print("   [警告] 锚点太集中，旋转解不准；尽量取画面两端的角点")
    if len(anchors) >= 3:
        p = np.array([[a[1], a[2]] for a in anchors], dtype=np.float64)
        area = abs(float(np.cross(p[1] - p[0], p[2] - p[0]))) / 2.0
        print("   三点构成的面积 %.0f mm²" % area)
        if area < 2000:
            print("   [警告] 三点几乎共线，解不出旋转；换更分散的角点")
    print("\n下一步：")
    print("  python3 tools/planar_calib.py --image %s --anchors-file %s"
          % "<棋盘格照片>", "<上面 --out 的路径>")


def _sim_hold():
    """自测停顿检测：静止要能采到、运动要丢掉、原地重复不能算新锚点、断流要开门。"""
    ok = True
    dt, rate = 0.05, 20.0        # 20 Hz，和 STM32 回传差不多
    hd = HoldDetector(tol_mm=0.6, hold_s=1.0, min_samples=8, min_move_mm=15.0)

    rnd = np.random.RandomState(7)

    def _plateau(t, x, y, secs, jitter=0.15):
        out = []
        for i in range(int(secs * rate)):
            out.append((t + i * dt, x + rnd.uniform(-jitter, jitter),
                        y + rnd.uniform(-jitter, jitter)))
        return out

    def _move(t, p0, p1, secs):
        n = max(2, int(secs * rate))
        return [(t + i * dt,
                 p0[0] + (p1[0] - p0[0]) * i / (n - 1),
                 p0[1] + (p1[1] - p0[1]) * i / (n - 1)) for i in range(n)]

    # 段 1：静止 1.6s（应采到）-> 运动 1s（不该采到）-> 静止 1.6s（应采到）
    stream = []
    t = 0.0
    stream += _plateau(t, 20.0, 30.0, 1.6); t += 1.6
    stream += _move(t, (20.0, 30.0), (300.0, 40.0), 1.0); t += 1.0
    stream += _plateau(t, 300.0, 40.0, 1.6); t += 1.6
    evs = []
    for (tt, x, y) in stream:
        ev = hd.feed(tt, x, y)
        if ev:
            evs.append(ev)
    if len(evs) != 2:
        print("❌ 停顿检测：应采到 2 个位姿，实际 %d 个" % len(evs))
        ok = False
    elif abs(evs[0]["x"] - 20) > 0.6 or abs(evs[1]["x"] - 300) > 0.6:
        print("❌ 停顿检测：坐标不对 %s" % ([(round(e["x"], 1), round(e["y"], 1)) for e in evs],))
        ok = False
    else:
        print("✅ 停顿检测：静止段采到 2 个、运动段不采（抖动 %.2fmm）" % evs[0]["spread"])

    # 原地不动继续喂 3 秒：不能又冒出新的锚点
    n_before = len(evs)
    for i in range(int(3 * rate)):
        ev = hd.feed(t + i * dt, 300.0, 40.0)
        if ev:
            evs.append(ev)
    if len(evs) != n_before:
        print("❌ 原地重复被当成新锚点（多了 %d 个）" % (len(evs) - n_before))
        ok = False
    else:
        print("✅ 原地重复不重复记录（dup=%d）" % hd.dup)

    # 断流：时间跳 3 秒后换位置，旧窗口必须作废
    ev = None
    t2 = t + 3 * rate * dt + 5.0
    for (tt, x, y) in _plateau(t2, 600.0, 100.0, 1.4):
        ev = hd.feed(tt, x, y) or ev
    if ev is None or abs(ev["x"] - 600) > 0.6:
        print("❌ 断流后没重新开始窗口（ev=%s）" % (ev,))
        ok = False
    elif hd.gaps < 1:
        print("❌ 断流没被计数（gaps=%d）" % hd.gaps)
        ok = False
    else:
        print("✅ 断流识别正常（gaps=%d），断流后仍能采到新位姿" % hd.gaps)

    # 抖动过大（连续缓慢漂移）时不能算静止
    hd2 = HoldDetector(tol_mm=0.6, hold_s=1.0, min_samples=8, min_move_mm=0.0)
    drift = [(i * dt, 100.0 + i * 0.5, 100.0) for i in range(100)]   # 10mm/s 缩放
    got = [hd2.feed(tt, x, y) for (tt, x, y) in drift]
    if any(got):
        print("❌ 缓慢漂移被误判成静止（%d 个）" % sum(1 for g in got if g))
        ok = False
    else:
        print("✅ 缓慢漂移不会误判成停顿位姿")
    return ok


def run_simulate():
    """不需要硬件：造一段带杂音/坏 CRC 的字节流，验证解析和记录流程。"""
    print("=== 模拟模式：验证解析器与锚点记录（不连串口）===")
    truth = [(-12, 407), (310, 175), (520, 88), (0, 0)]
    stream = bytearray(b"\xAA\x55junk")
    for (x, y) in truth:
        stream.extend(encode_pos(x, y))
        stream.extend(encode_pos(x, y))            # 连发两帧，验证去重
        stream.extend(b"\x07\x00")                 # 半包/假帧头
    stream.extend(b"\x07\x11\x22\x33\x44\x55")     # CRC 必错的一帧

    r = PosReader()
    got = []
    for i in range(0, len(stream), 3):             # 打散成 3 字节一段，测半包
        r.feed(stream[i:i + 3])
        if r.last:
            got.append(r.last)

    ok = True
    uniq = []
    for p in got:
        if not uniq or uniq[-1] != p:
            uniq.append(p)
    if uniq != truth:
        print("❌ 解析结果不符：%s" % (uniq,))
        ok = False
    else:
        print("✅ 打散输入仍能正确还原 %d 个坐标" % len(uniq))
    if r.bad == 0:
        print("❌ 坏 CRC 没被检出（bad=0）")
        ok = False
    else:
        print("✅ 坏 CRC 已检出（bad=%d，丢字节重同步后没卡死）" % r.bad)

    tmpdir = os.path.join(os.environ.get("TEMP", "/tmp"), "anchors_sim")
    out = os.path.join(tmpdir, "anchors.json")
    anchors = [(0, uniq[0][0], uniq[0][1]),
               (8, uniq[1][0], uniq[1][1]),
               (53, uniq[2][0], uniq[2][1])]
    try:
        _write_out(out, anchors, "/root/board.jpg", {"simulate": True})
        with io.open(out, encoding="utf-8") as f:
            back = json.load(f)
        if len(back["anchors"]) != 3 or back["anchors"][0]["idx"] != 0:
            print("❌ 写出的锚点文件不对：%s" % (back,))
            ok = False
        else:
            print("✅ 锚点文件读写正常: %s" % out)
    except Exception as e:                            # noqa: BLE001
        print("❌ 写锚点文件失败: %s: %s" % (type(e).__name__, e))
        ok = False

    # 往返到 planar_calib 的读取逻辑（有就测）
    try:
        import planar_calib
        idxs, xys, meta = planar_calib.load_anchors_file(out)
        if idxs == [0, 8, 53] and abs(xys[1][0] - uniq[1][0]) < 1e-6:
            print("✅ planar_calib 能直接读这个文件（--anchors-file 可用）")
        else:
            print("❌ planar_calib 读出来不对：%s %s" % (idxs, xys))
            ok = False
    except Exception as e:                            # noqa: BLE001
        print("   (跳过 planar_calib 联读：%s)" % (e,))

    if not _sim_hold():
        ok = False

    print("\n%s" % ("自检通过 ✅" if ok else "自检失败 ❌"))
    return 0 if ok else 1


class _FrameGrabber:
    """停顿位姿确认时顺手存一张原图（--frames-dir，可选）。

    板子上拍照时相机不能被检测节点占着：先跑 /root/vision_ws/stop_all.sh。
    """

    def __init__(self, out_dir, width=640, height=480):
        self.out_dir = out_dir
        self.width = int(width)
        self.height = int(height)
        self.cap = None
        self.cv2 = None
        self.fail = 0

    def _open(self):
        if self.cap is not None:
            return True
        try:
            import cv2
        except Exception as e:                          # noqa: BLE001
            print("\n[拍照] 没装 cv2，跳过存图：%s" % (e,))
            return False
        for dev in _camera_devices():
            cap = cv2.VideoCapture(dev)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap, self.cv2 = cap, cv2
                print("\n[拍照] 相机已打开 %s (%dx%d)"
                      % (dev, frame.shape[1], frame.shape[0]))
                return True
            cap.release()
        print("\n[拍照] 打不开相机（检测节点占着的话先跑 stop_all.sh）")
        return False

    def grab(self, tag):
        if not self._open():
            self.fail += 1
            return ""
        for _ in range(3):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                path = os.path.join(self.out_dir, "%s.png" % tag)
                d = os.path.dirname(os.path.abspath(path))
                if d and not os.path.isdir(d):
                    os.makedirs(d)
                self.cv2.imwrite(path, frame)
                return path
            time.sleep(0.05)
        self.fail += 1
        return ""

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:                               # noqa: BLE001
            pass
        self.cap = None


def _camera_devices():
    """优先 /dev/cam0（udev 稳定符号链接），再退回 video0/1/2。"""
    out = []
    for p in ("/dev/cam0", "/dev/video0", "/dev/video1", "/dev/video2"):
        if os.path.exists(p) and p not in out:
            out.append(p)
    return out or ["/dev/video0"]


def _poll_command():
    """非阻塞读一个命令字符：u=撤销、q=结束；没有输入返回 None。"""
    try:
        import select
    except ImportError:
        return None
    try:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return (sys.stdin.readline().strip().lower()[:1] or None)
    except (OSError, ValueError):
        return None


def _ex(anchors, args):
    """写进锚点文件的附加说明（哪种模式、什么参数采的，方便复盘）。"""
    ex = {"corners": [a[0] for a in anchors],
          "mode": "auto_hold" if getattr(args, "auto", False) else "manual"}
    if getattr(args, "auto", False):
        ex.update({"hold_s": args.hold_s, "tol_mm": args.tol_mm,
                   "min_samples": args.min_samples, "min_move_mm": args.min_move_mm,
                   "pose_log": getattr(args, "pose_log", "") or "",
                   "frames_dir": getattr(args, "frames_dir", "") or ""})
    return ex


def run_auto(args):
    """停顿自动采集：电机停住够久就记一个位姿，还在动的位姿直接丢弃。

    操作：把夹爪尖依次挪到 --corners 给的角点上方、贴住桌面，松手别动
    约 2 秒 -> 自动记录；然后再挪下一个。不用按回车。
    """
    try:
        import serial
    except ImportError:
        print("没装 pyserial。板子上应该有；PC 上用 python -m pip install pyserial")
        return 1

    corners = [int(c) for c in args.corners]
    hd = HoldDetector(tol_mm=args.tol_mm, hold_s=args.hold_s,
                      min_samples=args.min_samples, min_move_mm=args.min_move_mm)
    grabber = (_FrameGrabber(args.frames_dir, args.width, args.height)
               if args.frames_dir else None)
    out = _out_path(args)
    anchors = []
    pose_log = None
    if args.pose_log:
        d = os.path.dirname(os.path.abspath(args.pose_log))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        pose_log = io.open(args.pose_log, "a", encoding="utf-8")

    print("=== 停顿自动采集 %d 个锚点：角点 %s ===" % (len(corners), corners))
    print("串口 %s @ %d | 判定：静止 %.1fs / 容差 %.2fmm / 样本≥%d / 两次之间要挪开 %.0fmm"
          % (args.port, args.baud, args.hold_s, args.tol_mm,
             args.min_samples, args.min_move_mm))
    print("操作：把夹爪尖挪到角点上方贴住，松手别动，停住约 %.0f 秒自动记录。"
          % (args.hold_s + 0.5))
    print("      命令行随时可敲：u=撤销上一个   q=结束\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
    except Exception as e:                              # noqa: BLE001
        print("打开串口失败：%s: %s" % (type(e).__name__, e))
        print("提示：sort_vision_node 在跑的话会占着串口，先 bash /root/vision_ws/stop_all.sh")
        return 1

    r = PosReader()
    k = 0
    try:
        while k < len(corners):
            try:
                r.feed(ser.read(64))
            except Exception as e:                      # noqa: BLE001
                print("\n串口读失败：%s: %s" % (type(e).__name__, e))
                break
            now = time.time()
            pos = r.last
            stale = (pos is None) or (r.age() > 1.0)
            sys.stdout.write("\r" + ("下一个 #%-4d | %s | %s | 已记录 %d/%d  bad=%d"
                                     % (corners[k], _fmt_pos(pos),
                                        "⚠ 坐标超时 %.1fs" % r.age() if stale else "数据新鲜",
                                        len(anchors), len(corners), r.bad)).ljust(78))
            sys.stdout.flush()

            cmd = _poll_command()
            if cmd == "q":
                print("\n提前结束，已记录 %d 个" % len(anchors))
                break
            if cmd == "u" and anchors:
                dropped = anchors.pop()
                print("\n已撤销 #%d (%.1f, %.1f)，这一个重新采"
                      % (dropped[0], dropped[1], dropped[2]))
                hd.set_accepted([(a[1], a[2]) for a in anchors])
                k = max(0, k - 1)
                _write_out(out, anchors, args.image or "", _ex(anchors, args))
                continue

            if stale:
                time.sleep(0.02)
                continue

            ev = hd.feed(now, pos[0], pos[1])
            if ev is None:
                time.sleep(0.02)
                continue

            idx = corners[k]
            frame = (grabber.grab("anchor_%02d_idx%d" % (len(anchors) + 1, idx))
                     if grabber else "")
            anchors.append((idx, ev["x"], ev["y"]))
            _write_out(out, anchors, args.image or "", _ex(anchors, args))  # 边采边存
            print("\n✅ 记录 #%d -> (%.1f, %.1f) mm | 静止 %.1fs / %d 帧 / 抖动 %.2fmm%s"
                  % (idx, ev["x"], ev["y"], ev["t1"] - ev["t0"], ev["n"], ev["spread"],
                     " | 存图 %s" % frame if frame else ""))
            if pose_log:
                pose_log.write(json.dumps(
                    {"idx": idx, "t0": round(ev["t0"], 3), "t1": round(ev["t1"], 3),
                     "x": round(ev["x"], 3), "y": round(ev["y"], 3),
                     "spread": round(ev["spread"], 3), "n": ev["n"], "frame": frame},
                    ensure_ascii=False) + "\n")
                pose_log.flush()
            k += 1
            if k < len(corners):
                print("   现在挪到角点 #%d 上方，停住别动…" % corners[k])
    finally:
        try:
            ser.close()
        except Exception:                               # noqa: BLE001
            pass
        if pose_log:
            pose_log.close()
        if grabber:
            grabber.close()

    if not anchors:
        print("\n一个锚点都没采到。检查：STM32 在发 0x07 吗？电机是不是真的停住了？"
              "（容差 %.2fmm 可放宽到 1.5，静止时间也可调短）" % args.tol_mm)
        return 1
    print("\n已写入 %s（断流 %d 次，同位置重复丢弃 %d 次）" % (out, hd.gaps, hd.dup))
    _summary(anchors)
    return 0


def run_replay(args):
    """离线回放一段坐标记录（每行 t,x,y 或 JSONL），看停顿检测挑出哪些位姿。

    没有电机时用它验证判定参数；现场采的数据也能事后复盘。
    """
    path = args.replay
    rows = []
    with io.open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            if raw.startswith("{"):
                d = json.loads(raw)
                rows.append((float(d.get("t", len(rows))), float(d["x"]), float(d["y"])))
            else:
                parts = [p for p in raw.replace(";", ",").split(",") if p.strip()]
                if len(parts) < 3:
                    continue
                rows.append((float(parts[-3]), float(parts[-2]), float(parts[-1])))
        except (ValueError, KeyError, TypeError):
            continue
    if not rows:
        print("回放文件里没读到有效数据（每行需要 t,x,y 三个数，或带 x/y 的 JSONL）")
        return 1
    rows.sort(key=lambda r: r[0])
    hd = HoldDetector(tol_mm=args.tol_mm, hold_s=args.hold_s,
                      min_samples=args.min_samples, min_move_mm=args.min_move_mm)
    events = []
    for (t, x, y) in rows:
        ev = hd.feed(t, x, y)
        if ev:
            events.append(ev)
    span = rows[-1][0] - rows[0][0]
    print("回放 %s：%d 帧 / %.1f 秒（%.1f Hz），检出 %d 个停顿位姿，断流 %d 次"
          % (path, len(rows), span, (len(rows) / span) if span > 0 else 0.0,
             len(events), hd.gaps))
    for i, ev in enumerate(events):
        print("  #%d  静止 %.2fs  %d 帧  (%.1f, %.1f) mm  抖动 %.2fmm%s"
              % (i + 1, ev["t1"] - ev["t0"], ev["n"], ev["x"], ev["y"], ev["spread"],
                 "" if ev["moved_mm"] is None else "  距上次 %.1fmm" % ev["moved_mm"]))
    if args.out and events:
        corners = [int(c) for c in args.corners][:len(events)]
        anchors = [(corners[i], ev["x"], ev["y"])
                   for i, ev in enumerate(events) if i < len(corners)]
        if anchors:
            _write_out(args.out, anchors, args.image or "",
                       {"mode": "replay", "source": os.path.abspath(path)})
            print("已写入 %s（%d 个锚点）" % (args.out, len(anchors)))
    return 0


def run_serial(args):
    try:
        import serial
    except ImportError:
        print("没装 pyserial。板子上应该有；PC 上用 python -m pip install pyserial")
        return 1

    corners = [int(c) for c in args.corners]
    print("=== 采集 %d 个锚点：角点 %s ===" % (len(corners), corners))
    print("串口 %s @ %d" % (args.port, args.baud))
    print("操作：把夹爪尖降到该角点【正上方并贴住桌面】，稳定后按回车记录")
    print("      （留空回车=重新采当前点，输入 q 回车=提前结束）\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
    except Exception as e:                            # noqa: BLE001
        print("打开串口失败：%s: %s" % (type(e).__name__, e))
        print("提示：sort_vision_node 在跑的话会占着串口，先停掉它")
        return 1

    r = PosReader()
    anchors = []
    try:
        for k, idx in enumerate(corners):
            print("-" * 62)
            print("第 %d/%d 个锚点：角点 #%d" % (k + 1, len(corners), idx))
            while True:
                r.feed(ser.read(64))
                line = "%s  %s  %s" % (
                    _fmt_pos(r.last),
                    "数据新鲜" if r.age() < 0.5 else "⚠ 坐标超时 %.1fs" % r.age(),
                    "bad=%d" % r.bad)
                sys.stdout.write("\r" + line.ljust(58))
                sys.stdout.flush()
                # 非阻塞等回车
                import select
                try:
                    ready = select.select([sys.stdin], [], [], 0.05)[0]
                except (OSError, ValueError):
                    ready = []
                    time.sleep(0.05)
                if not ready:
                    continue
                txt = sys.stdin.readline().strip().lower()
                if txt == "q":
                    print("\n提前结束，已采集 %d 个" % len(anchors))
                    return _finish(args, anchors)
                if r.last is None or r.age() > 1.0:
                    print("\n[跳过] 没收到新鲜坐标（STM32 没发 0x07？POS_ENABLE 开了吗？）")
                    return 1
                samples = []
                t_end = time.time() + 0.4
                while time.time() < t_end:            # 多帧取中位数，压抖动
                    r.feed(ser.read(64))
                    if r.last:
                        samples.append(r.last)
                    time.sleep(0.01)
                pos = _median_pos(samples) or r.last
                anchors.append((idx, pos[0], pos[1]))
                print("\n已记录 #%d -> (%.1f, %.1f) mm  [%d 帧中位数]"
                      % (idx, pos[0], pos[1], len(samples)))
                break
        return _finish(args, anchors)
    finally:
        try:
            ser.close()
        except Exception:                             # noqa: BLE001
            pass


def _finish(args, anchors):
    if not anchors:
        print("一个锚点都没采到，什么都没写。")
        return 1
    out = _out_path(args)
    _write_out(out, anchors, args.image or "",
               {"corners": [a[0] for a in anchors]})
    print("\n已写入 %s" % out)
    _summary(anchors)
    return 0


def _parse_xy_line(text):
    """把 "12.5 -30" / "-120.5,33" / "12.5  -30.2" 解析成 (x, y)；不合法返回 None。"""
    t = (text or "").replace(",", " ").replace("，", " ").strip()
    parts = t.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def run_manual(args):
    """手工录坐标：STMP32 回传不稳时的兜底路径（坐标从 32 的显示/上位机读）。

    用法：每个位置把夹爪尖对到角点正上方 → 停住 → 直接敲 X Y 两个数。
    不碰串口，不需要固件支持，采到的 anchors.json 和自动模式完全通用。
    """
    corners = [int(c) for c in args.corners]
    print("=== 手工记录 %d 个锚点：角点 %s ===" % (len(corners), corners))
    print("每个位置：夹爪尖对到该角点正上方、停住，然后输入 X Y（空格或逗号分隔，q 退出）")
    anchors = []
    for k, idx in enumerate(corners):
        while True:
            sys.stdout.write("\n[%d/%d] 角点 #%d  坐标 X Y: "
                             % (k + 1, len(corners), idx))
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:                                  # Ctrl-D
                return _finish(args, anchors)
            t = line.strip().lower()
            if t in ("q", "quit", "exit"):
                return _finish(args, anchors)
            xy = _parse_xy_line(t)
            if xy is None:
                print("  格式不对：要两个数字，例如   -120.5 33.0")
                continue
            anchors.append((idx, xy[0], xy[1]))
            print("  已记录 #%d -> (%.1f, %.1f) mm" % (idx, xy[0], xy[1]))
            break
    return _finish(args, anchors)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="采集手眼标定锚点（STM32 回传坐标，不用手抄）；"
                    "--auto 用停顿检测自动记，--replay 离线复盘")
    ap.add_argument("--port", default=_default_port(),
                    help="串口设备；默认自动找 /dev/ttyACM0（STM32 CDC）")
    ap.add_argument("--baud", type=int, default=_cfg("SERIAL_BAUD", 115200))
    ap.add_argument("--corners", nargs="+", default=["0", "8", "53"],
                    help="要采的棋盘格角点序号（看 planar_calib 打印的抽样表）")
    ap.add_argument("--image", default="", help="对应那张棋盘格照片（记进文件，防错配）")
    ap.add_argument("--out", default="", help="输出锚点文件（默认 /root/vision_ws/anchors.json）")

    g = ap.add_argument_group("停顿检测（--auto / --replay 用）")
    g.add_argument("--auto", action="store_true",
                   help="自动模式：电机停住够久就记一个位姿，运动中的位姿丢弃（不用按回车）")
    g.add_argument("--hold-s", type=float, default=1.5,
                   help="静止多久才算一个可用位姿（默认 1.5s）")
    g.add_argument("--tol-mm", type=float, default=0.6,
                   help="静止判定的位置容差（默认 0.6mm；抖动大就放宽到 1.5）")
    g.add_argument("--min-samples", type=int, default=8,
                   help="一个停顿窗口至少要几帧（默认 8）")
    g.add_argument("--min-move-mm", type=float, default=15.0,
                   help="两次锚点之间至少要挪开多少 mm（默认 15，防重复记录）")
    g.add_argument("--pose-log", default="",
                   help="把每个停顿位姿追加写成 JSONL（复盘用）")
    g.add_argument("--frames-dir", default="",
                   help="每个停顿位姿顺手存一张原图（相机没被占用时可用）")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--replay", default="",
                   help="离线回放坐标记录（每行 t,x,y 或 JSONL），只打印检出的位姿")

    ap.add_argument("--simulate", action="store_true", help="离线自测，不连硬件")
    ap.add_argument("--manual", action="store_true",
                    help="手工录坐标：不读串口，每个位置敲 X Y（回传不稳定时的兜底）")
    args = ap.parse_args(argv)

    if args.simulate:
        return run_simulate()
    if args.replay:
        return run_replay(args)
    if args.manual:
        return run_manual(args)
    if args.auto:
        return run_auto(args)
    return run_serial(args)


if __name__ == "__main__":
    sys.exit(main())