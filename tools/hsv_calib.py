# -*- coding: utf-8 -*-
"""HSV 颜色阈值标定助手：从"每种颜色一叠照片"里自动算出建议区间。

解决的问题：
    现在 config.COLOR_RANGES_HSV 是手调的默认值（S 下限 70，现场实测绿色
    S 中位数只有 39 —— 一上手就判不出来）。手工试阈值全靠感觉，也没有
    客观依据说明"到底会不会和别的颜色混"。

这个工具的输入是"已知颜色的照片"，输出是：
    1. 每种颜色的 H/S/V 分位数（客观分布，不是猜的）
    2. 建议区间（H 用圆统计处理红色跨 0° 的问题）
    3. 颜色之间的 H 区间冲突检查（最容易混的地方）
    4. 用当前 config 阈值跑一遍这批照片的召回/污染（改之前先看基线）
    5. 可直接粘贴进 config.py 的代码片段

用法：
    # 目录结构：一种颜色一个子目录，里面放该颜色物料的照片
    #   标定图片/green/1.jpg  标定图片/green/2.jpg
    #   标定图片/red/*.jpg    ...
    python tools/hsv_calib.py --data 标定图片
    python tools/hsv_calib.py --data 标定图片 --report hsv_report.md --apply

拍照要求（很重要，否则算出来的是垃圾）：
    * 一种颜色一个目录，目录名用 config 里的英文色名（green/red/blue...）
    * 物料放在画面【中间】，占画面 1/3 以上（工具默认只取中心区域）
    * 用最终现场的光照条件拍（补光灯开着），不要用手机闪光灯
    * 每种颜色 ≥8 张，覆盖不同的摆放角度
"""
import argparse
import io
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "rdk", "vision_sorter"),
           os.path.join(_HERE, "..", "vision_sorter"),
           "/root/vision_ws/vision_sorter", _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import config as C
except Exception:                                       # noqa: BLE001
    C = None

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:                                   # noqa: BLE001
        pass

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 忽略的色名：black/white 主要看 V/S，H 没意义，不参与 H 区间计算
SKIP_H = ("black", "white")


# ============================ 采样 ============================
def center_roi(img, ratio=0.40):
    """取中心区域：假设物料在画面正中（拍照要求里写明了）。"""
    h, w = img.shape[:2]
    rh, rw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    return img[y0:y0 + rh, x0:x0 + rw]


def collect_pixels(data_dir, ratio=0.40, min_s=0, min_v=0):
    """遍历 <data_dir>/<color>/*.jpg，返回 {color: Nx3 HSV数组}。"""
    colors = {}
    for name in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, name)
        if not os.path.isdir(d):
            continue
        files = [f for f in sorted(os.listdir(d))
                 if f.lower().endswith(IMG_EXT)]
        if not files:
            print("[跳过] %s：没有图片" % name)
            continue
        chunks = []
        for f in files:
            img = cv2.imread(os.path.join(d, f))
            if img is None:
                print("[警告] 读不了 %s/%s" % (name, f))
                continue
            hsv = cv2.cvtColor(center_roi(img, ratio), cv2.COLOR_BGR2HSV)
            px = hsv.reshape(-1, 3)
            if min_s or min_v:
                keep = (px[:, 1] >= min_s) & (px[:, 2] >= min_v)
                px = px[keep]
            if len(px):
                chunks.append(px)
        if chunks:
            arr = np.vstack(chunks)
            colors[name] = arr
            print("[读取] %-8s %2d 张，%7d 像素" % (name, len(files), len(arr)))
        else:
            print("[跳过] %s：没有可用像素" % name)
    return colors


# ============================ H 圆统计 ============================
def circ_range(h, cover=0.98, margin=2):
    """H(0~179) 的"最紧连续区间"，自动处理红色跨 0° 的情况。

    关键：OpenCV 的 H 周期是 **180**（不是 360），所以映射到角度时必须用
    h * pi/180，否则 H 靠近 179 的点会被算到圆的另一边，区间整个错掉。

    返回 (lo, hi, wrapped)。wrapped=True 表示区间跨 0°（hi < lo）。
    """
    ang = h.astype(np.float64) * (np.pi / 180.0)     # 周期 180 -> 角周期 2pi
    s = np.sin(ang).mean()
    c = np.cos(ang).mean()
    if abs(s) < 1e-12 and abs(c) < 1e-12:            # 均匀分布，没有主方向
        return 0, 179, False
    mean = np.arctan2(s, c)
    d = np.angle(np.exp(1j * (ang - mean)))          # (-pi, pi]
    ad = np.abs(d)
    half = float(np.quantile(ad, cover)) * 180.0 / np.pi
    ctr = (mean * 180.0 / np.pi) % 180.0
    lo = int(round((ctr - half - margin) % 180))
    hi = int(round((ctr + half + margin) % 180))
    return lo, hi, hi < lo


def overlap_h(a, b):
    """两个 H 区间（可能是跨 0° 的）重叠多少度。"""
    def to_set(r, wrapped_lo, hi):
        if not wrapped_lo:
            return set(range(r, hi + 1))
        return set(range(r, 180)) | set(range(0, hi + 1))
    sa = to_set(a[0], a[2], a[1])
    sb = to_set(b[0], b[2], b[1])
    return len(sa & sb)


# ============================ 当前阈值评估 ============================
def _ranges_for(color):
    """当前 config 里这个颜色的 (lower, upper) 列表（红色有两段）。"""
    if C is None:
        return []
    out = []
    rng = getattr(C, "COLOR_RANGES_HSV", {}).get(color)
    if rng:
        out.append((np.array(rng[0]), np.array(rng[1])))
    if color == "red":
        r2 = getattr(C, "RED_HSV_LOWER2", None)
        if r2:
            out.append((np.array(r2[0]), np.array(r2[1])))
    return out


def in_range(px, lo, hi):
    return np.all((px >= lo) & (px <= hi), axis=1)


def evaluate(colors):
    """用当前 config 跑一遍：每种颜色的召回率，以及被别的颜色"抢"走多少。"""
    if C is None:
        return {}
    names = [n for n in colors if _ranges_for(n)]
    result = {}
    for name in names:
        px = colors[name]
        own = np.zeros(len(px), dtype=bool)
        for lo, hi in _ranges_for(name):
            own |= in_range(px, lo, hi)
        recall = float(own.mean())
        steal = {}
        for other in names:
            if other == name:
                continue
            m = np.zeros(len(px), dtype=bool)
            for lo, hi in _ranges_for(other):
                m |= in_range(px, lo, hi)
            v = float((m & ~own).mean())
            if v > 0.02:
                steal[other] = v
        result[name] = {"recall": recall, "steal": steal}
    return result


# ============================ 建议 ============================
def suggest(colors, s_margin=10, v_margin=15):
    """给出每种颜色的建议区间。"""
    out = {}
    for name, px in colors.items():
        h, s, v = px[:, 0], px[:, 1], px[:, 2]
        item = {
            "n_pixels": int(len(px)),
            "s_p1": float(np.percentile(s, 1)),
            "s_p50": float(np.percentile(s, 50)),
            "s_p99": float(np.percentile(s, 99)),
            "v_p1": float(np.percentile(v, 1)),
            "v_p50": float(np.percentile(v, 50)),
            "v_p99": float(np.percentile(v, 99)),
        }
        if name not in SKIP_H:
            lo, hi, wrapped = circ_range(h)
            item["h"] = (lo, hi, wrapped)
        # 建议 S 下限：取 p1 再往下留余量（现场光照变化要留空间）
        s_lo = max(0, int(item["s_p1"] - s_margin))
        s_hi = 255
        v_lo = max(0, int(item["v_p1"] - v_margin))
        v_hi = 255
        if name == "black":
            # 黑色靠"暗"，S 和 V 都要封顶
            s_lo, s_hi = 0, max(30, int(item["s_p99"] + s_margin))
            v_lo, v_hi = 0, max(30, int(item["v_p99"] + v_margin))
        elif name == "white":
            s_lo, s_hi = 0, max(30, int(item["s_p99"] + s_margin))
            v_lo = max(120, int(item["v_p1"] - v_margin))
            v_hi = 255
        item["suggest"] = (s_lo, s_hi, v_lo, v_hi)
        out[name] = item
    return out


def conflicts(sugg):
    """H 区间两两重叠检查。"""
    pairs = []
    names = [n for n in sugg if "h" in sugg[n]]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sugg[names[i]]["h"], sugg[names[j]]["h"]
            ov = overlap_h(a, b)
            if ov > 0:
                pairs.append((names[i], names[j], ov))
    return sorted(pairs, key=lambda x: -x[2])


def config_snippet(sugg):
    """生成可粘贴的 config 片段。

    注意 S/V 下限一律用【实测分位数 - 余量】，不要人为抬到 70/90 ——
    现场绿色的 S 中位数只有 39，抬到 70 就直接全判失败了。
    """
    lines = ["COLOR_RANGES_HSV = {"]
    red_extra = []
    for name in sorted(sugg):
        it = sugg[name]
        slo, shi, vlo, vhi = it["suggest"]
        if "h" not in it:                      # black / white：H 全覆盖
            lines.append('    "%s": [(  0, %3d, %3d), (179, %3d, %3d)],'
                         % (name, slo, vlo, shi, vhi))
            continue
        lo, hi, wrapped = it["h"]
        if wrapped:
            # 红色跨 0°：主区间取 [0, hi]，另一段放进 RED_HSV_LOWER2
            lines.append('    "%s": [(  0, %3d, %3d), (%3d, %3d, %3d)],'
                         % (name, slo, vlo, hi, shi, vhi))
            red_extra.append('RED_HSV_LOWER2 = [(%3d, %3d, %3d), (179, %3d, %3d)]'
                             % (lo, slo, vlo, shi, vhi))
        else:
            lines.append('    "%s": [(%3d, %3d, %3d), (%3d, %3d, %3d)],'
                         % (name, lo, slo, vlo, hi, shi, vhi))
    lines.append("}")
    lines.extend(red_extra)
    return "\n".join(lines)


# ============================ 报告 ============================
def build_report(colors, sugg, conf, ev, ratio):
    L = []
    L.append("# HSV 颜色阈值标定报告\n")
    L.append("- 采样目录中心区域（ratio=%.2f），只统计中心区域像素" % ratio)
    L.append("- 颜色数：%d，总像素：%d\n"
             % (len(colors), sum(len(v) for v in colors.values())))

    # 1. 分布
    L.append("## 一、各颜色 H/S/V 分布（实测）\n")
    L.append("| 颜色 | 像素数 | H 区间(建议) | S p1/p50/p99 | V p1/p50/p99 |")
    L.append("|---|---|---|---|---|")
    for name in sorted(sugg):
        it = sugg[name]
        h = ("%d~%d%s" % (it["h"][0], it["h"][1], "（跨0°）" if it["h"][2] else "")
             if "h" in it else "—")
        L.append("| %s | %d | %s | %.0f / %.0f / %.0f | %.0f / %.0f / %.0f |"
                 % (name, it["n_pixels"], h,
                    it["s_p1"], it["s_p50"], it["s_p99"],
                    it["v_p1"], it["v_p50"], it["v_p99"]))

    # 2. 冲突
    L.append("\n## 二、颜色冲突检查（H 区间重叠）\n")
    if not conf:
        L.append("✅ 各颜色的 H 区间互不重叠，HSV 单靠色相就能分开。\n")
    else:
        L.append("⚠ 以下颜色在色相上重叠，需要靠 S/V 再分，或者检查物料本身是否太接近：\n")
        L.append("| 颜色 A | 颜色 B | 重叠度数 |")
        L.append("|---|---|---|")
        for a, b, ov in conf:
            L.append("| %s | %s | %d° |" % (a, b, ov))

    # 3. 当前阈值表现
    if ev:
        L.append("\n## 三、当前 config.COLOR_RANGES_HSV 的表现\n")
        L.append("| 颜色 | 召回率（自身像素落在自身区间） | 被其他颜色抢走 |")
        L.append("|---|---|---|")
        for name in sorted(ev):
            e = ev[name]
            steal = "，".join("%s %.0f%%" % (k, v * 100)
                             for k, v in sorted(e["steal"].items(),
                                                key=lambda x: -x[1])) or "—"
            flag = "✅" if e["recall"] >= 0.60 else "❌"
            L.append("| %s | %s %.0f%% | %s |" % (name, flag, e["recall"] * 100, steal))
        L.append("\n> 召回率低于 60% 说明该颜色阈值基本抓不住，必须先调；"
                 "出现明显\"被抢\"说明两种颜色会互相误判。")

    # 4. 建议
    L.append("\n## 四、建议阈值（可直接粘贴）\n")
    L.append("```python")
    L.append(config_snippet(sugg))
    L.append("```")
    L.append("\n> S/V 下限取的是【实测 1% 分位数再留余量】。不要手动抬到 70/90，"
             "现场绿色的 S 中位数只有 39，抬高了就直接全判失败。"
             "如果发现某颜色被别的颜色误判，优先缩 H 区间，其次抬 S 下限。")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HSV 颜色阈值标定助手")
    ap.add_argument("--data", required=True,
                    help="照片目录，结构：<dir>/<颜色名>/*.jpg")
    ap.add_argument("--roi", type=float, default=0.40,
                    help="只取画面中心多大比例（默认 0.40）")
    ap.add_argument("--report", default="", help="把报告写成 markdown")
    ap.add_argument("--apply", action="store_true", help="只打印 config 片段")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.data):
        print("目录不存在: %s" % args.data)
        return 1
    colors = collect_pixels(args.data, args.roi)
    if not colors:
        print("没读到任何图片。目录结构应为 %s/green/*.jpg 这样" % args.data)
        return 1

    sugg = suggest(colors)
    conf = conflicts(sugg)
    ev = evaluate(colors)

    if args.apply:
        print(config_snippet(sugg))
        return 0

    report = build_report(colors, sugg, conf, ev, args.roi)
    print()
    print(report)
    if args.report:
        d = os.path.dirname(os.path.abspath(args.report))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with io.open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print("\n报告已写入 %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())