# -*- coding: utf-8 -*-
"""采集手眼标定锚点：STM32 回传坐标 -> 锚点文件（不用手抄数字）。

用法（板子上，标定时用；此时不要同时跑 sort_vision_node，串口是独占的）：

    # 1. 先拍一张棋盘格照片（见 BOARD_README 第八节）
    # 2. 采集 3 个锚点（把夹爪尖依次降到 3 个角点上方，每次回车）
    python3 tools/collect_anchors.py --port /dev/ttyUSB0 --corners 0 8 53 \
            --image /root/board.jpg --out /root/vision_ws/anchors.json

    # 3. 解算（自动读锚点文件，旋转和手性都不用猜）
    python3 tools/planar_calib.py --image /root/board.jpg \
            --anchors-file /root/vision_ws/anchors.json

离线自测（不需要串口，不需要 STM32）：

    python3 tools/collect_anchors.py --simulate

为什么需要 3 个点：
  1 个点只能修平移；2 个点能修旋转但分辨不了手性（是否镜像）；
  3 个分散的点能同时解出 比例/旋转/平移/手性 —— 这时 --rot/--mirror 就完全不用管了。
  选角点要尽量分散（如 0 / 8 / 53，即两个对角 + 一个异侧点），三点共线会解不出旋转。
"""
import argparse
import io
import json
import os
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


POS_HEAD = int(_cfg("POS_FRAME_HEAD", 0x07))
POS_LEN = int(_cfg("POS_FRAME_LEN", 6))


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
    """流式读 0x07 坐标帧，只保留最新一帧（丢帧无所谓）。"""

    def __init__(self):
        self.buf = bytearray()
        self.bad = 0
        self.total = 0
        self.last = None
        self.last_t = 0.0

    def feed(self, data):
        self.buf.extend(data)
        if len(self.buf) > 4096:                     # 防噪声撑爆
            del self.buf[:-POS_LEN * 4]
        while True:
            i = self.buf.find(bytes([POS_HEAD]))
            if i < 0:
                self.buf.clear()
                return
            if i:
                del self.buf[:i]
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
                self.last = (x, y)
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

    print("\n%s" % ("自检通过 ✅" if ok else "自检失败 ❌"))
    return 0 if ok else 1


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
    out = args.out or os.path.join(os.path.dirname(_PARENT), "anchors.json")
    if "vision_ws" in os.path.abspath(_HERE):       # 板端
        out = args.out or "/root/vision_ws/anchors.json"
    _write_out(out, anchors, args.image or "",
               {"corners": [a[0] for a in anchors]})
    print("\n已写入 %s" % out)
    _summary(anchors)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="采集手眼标定锚点（STM32 回传坐标，不用手抄）")
    ap.add_argument("--port", default=_cfg("SERIAL_PORT", "/dev/ttyUSB0"),
                    help="串口设备，PC 上是 COM7 这种")
    ap.add_argument("--baud", type=int, default=_cfg("SERIAL_BAUD", 115200))
    ap.add_argument("--corners", nargs="+", default=["0", "8", "53"],
                    help="要采的棋盘格角点序号（看 planar_calib 打印的抽样表）")
    ap.add_argument("--image", default="", help="对应那张棋盘格照片（记进文件，防错配）")
    ap.add_argument("--out", default="", help="输出锚点文件（默认 /root/vision_ws/anchors.json）")
    ap.add_argument("--simulate", action="store_true", help="离线自测，不连硬件")
    args = ap.parse_args(argv)

    if args.simulate:
        return run_simulate()
    return run_serial(args)


if __name__ == "__main__":
    sys.exit(main())