# -*- coding: utf-8 -*-
"""读一小段串口坐标，打印稳定值，并可把这一组 (角点序号 -> 夹爪 mm) 追加进 anchors.json。

为什么单独做这个工具：
    collect_anchors.py --auto 是"看到静止就记"，适合无人值守的长流程；
    但现场标定时人要先说"我把夹爪对准角点 #0 了"，再由程序记录才准确。
    这个工具就是那一次"记录"动作：0.6 秒取平均、报抖动，抖太大能立刻发现。

用法（板端）：
    # 只看当前坐标和抖动
    python3 tools/record_pose.py

    # 记录到角点 #0（追加/覆盖该序号，其它锚点保留）
    python3 tools/record_pose.py --index 0
    python3 tools/record_pose.py --index 8
    ...
    # 全部采完后查看
    cat /root/vision_ws/anchors.json
"""
import argparse
import io
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "vision_sorter"),
           "/root/vision_ws/vision_sorter", "/root/vision_ws/tools"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import collect_anchors as CA                       # noqa: E402

try:
    import serial
except ImportError:                                # noqa: BLE001
    serial = None


def main(argv=None):
    ap = argparse.ArgumentParser(description="记录单个锚点的夹爪坐标")
    ap.add_argument("--port", default=CA._default_port())
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=0.6,
                    help="采样时长（默认 0.6s，约 5 帧）")
    ap.add_argument("--index", type=int, default=None,
                    help="棋盘角点序号（给了才写文件）")
    ap.add_argument("--out", default="/root/vision_ws/anchors.json")
    ap.add_argument("--image", default="/root/vision_ws/handeye_shot.png",
                    help="与锚点对应的那张棋盘照片")
    ap.add_argument("--jitter-warn", type=float, default=1.0,
                    help="抖动超过多少 mm 就告警（默认 1.0）")
    a = ap.parse_args(argv)

    if serial is None:
        print("未安装 pyserial")
        return 1
    if not a.port or not os.path.exists(a.port):
        print("串口不存在：%s" % a.port)
        return 1

    ser = serial.Serial(a.port, a.baud, timeout=0)
    time.sleep(0.15)
    ser.reset_input_buffer()
    r = CA.PosReader()
    samples = []
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        d = ser.read(128)
        if d:
            r.feed(d)
        if r.last is not None:
            samples.append(r.last)
        time.sleep(0.005)
    ser.close()

    if not samples:
        print("⚠ 没收到坐标：检查 32 在不在发（valid 是否 >0）")
        return 1

    xs = [p[0] for p in samples]
    ys = [p[1] for p in samples]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    jit = max(max(xs) - min(xs), max(ys) - min(ys))
    print("坐标 x=%8.2f  y=%8.2f  | 样本=%d  抖动=%.2fmm  %s"
          % (mx, my, len(samples), jit,
             "" if jit <= a.jitter_warn else "⚠ 抖动偏大，确认是不是还在动"))

    if a.index is None:
        return 0

    data = {"image": a.image, "anchors": []}
    if os.path.exists(a.out):
        try:
            data = json.load(io.open(a.out, encoding="utf-8"))
        except (ValueError, OSError):
            data = {"image": a.image, "anchors": []}
    anchors = [x for x in data.get("anchors", []) if x.get("idx") != a.index]
    anchors.append({"idx": int(a.index), "x": round(mx, 2), "y": round(my, 2)})
    anchors.sort(key=lambda d: d["idx"])
    data["anchors"] = anchors
    data["image"] = a.image
    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("已写入 %s：当前 %d 个锚点 %s"
          % (a.out, len(anchors), [x["idx"] for x in anchors]))
    return 0


if __name__ == "__main__":
    sys.exit(main())