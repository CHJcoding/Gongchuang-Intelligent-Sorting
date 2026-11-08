# -*- coding: utf-8 -*-
"""三个 bin 同批图对照：gc_best_1029 / gc_best_1032b / gc_1032b_v2
A 组：dataset 每类抽样，衡量类别正确率与分数分布
B 组：现场连拍帧（无真值），看同一帧三个 bin 各报什么、分数多少
"""
import os, sys, glob, time
import numpy as np, cv2
from collections import Counter, defaultdict
sys.path.insert(0, "/root/vision_ws")
from vision_sorter.yolov8_bpu import Yolov8Bpu

CLASSES = ["hollow_cube", "tetrahedron", "pentagonal_prism", "triangular_prism",
           "cone", "cylinder", "hemisphere", "cube"]
MAP = {"LK": "hollow_cube", "SMT": "tetrahedron", "WLZ": "pentagonal_prism",
       "SLZ": "triangular_prism", "YZHU": "cone", "YZ": "cylinder",
       "BQ": "hemisphere", "CUBE": "cube", "negative": None}
RAW = "/root/vision_ws/dataset/20261031_1126/raw"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
BINS = ["/root/vision_ws/gc_best_1029.bin",
        "/root/vision_ws/gc_best_1032b.bin",
        "/root/vision_ws/gc_1032b_v2.bin"]

samples = []
for d in sorted(os.listdir(RAW)):
    p = os.path.join(RAW, d)
    if not os.path.isdir(p):
        continue
    fs = sorted(glob.glob(os.path.join(p, "*.jpg")))
    step = max(1, len(fs) // N)
    samples += [(d, f) for f in fs[::step][:N]]
live = sorted(glob.glob("/tmp/burst_*.jpg")) + sorted(glob.glob("/tmp/f[0-3].jpg")) + sorted(glob.glob("/tmp/live_*.jpg"))
print("A 组样本", len(samples), dict(Counter(d for d, _ in samples)), "conf", CONF, "| B 组现场帧", len(live))

results = {}
for b in BINS:
    tag = os.path.basename(b)
    t0 = time.time()
    m = Yolov8Bpu(b, CLASSES, conf_thres=CONF)
    per_acc = defaultdict(list); per_score = defaultdict(list); per_seen = defaultdict(int)
    neg_fp = 0; neg_n = 0; wrong = Counter()
    for truth, f in samples:
        im = cv2.imread(f)
        if im is None:
            continue
        ds = sorted(m.detect(im), key=lambda x: -float(x.score))
        if truth == "negative":
            neg_n += 1
            if ds:
                neg_fp += 1
            continue
        per_seen[truth] += 1
        if not ds:
            per_acc[truth].append(0); continue
        top = ds[0]
        ok = (top.name == MAP.get(truth))
        per_acc[truth].append(1 if ok else 0)
        per_score[truth].append(float(top.score))
        if not ok:
            wrong["%s->%s" % (truth, top.name)] += 1
    good = sum(sum(v) for v in per_acc.values()); pos = sum(len(v) for v in per_acc.values())
    print("\n==", tag)
    print("   正样本 top1 正确 %d/%d = %.1f%%   空盘误检 %d/%d" % (good, pos, 100.0 * good / max(1, pos), neg_fp, neg_n))
    for c in sorted(per_acc):
        sc = per_score[c]
        print("   %-8s %2d/%2d  分 median %.2f  min %.2f" % (c, sum(per_acc[c]), len(per_acc[c]),
              float(np.median(sc)) if sc else -1, float(np.min(sc)) if sc else -1))
    if wrong:
        print("   错法", dict(wrong.most_common(6)))
    live_res = []
    for f in live:
        im = cv2.imread(f)
        ds = sorted(m.detect(im), key=lambda x: -float(x.score)) if im is not None else []
        live_res.append((os.path.basename(f), ds[0].name if ds else "-", round(float(ds[0].score), 3) if ds else 0.0, len(ds)))
    results[tag] = live_res
    print("   载入+推理耗时 %.1fs" % (time.time() - t0))

print("\n==== B 组现场帧（同一帧，三个 bin 各自 top1）====")
print("%-14s %-28s %-28s %-28s" % ("frame", "1029", "1032b", "v2"))
for i in range(len(live)):
    row = [results[os.path.basename(b)][i] for b in BINS]
    print("%-14s %-28s %-28s %-28s" % (row[0][0],
        "%s %.2f n=%d" % (row[0][1], row[0][2], row[0][3]),
        "%s %.2f n=%d" % (row[1][1], row[1][2], row[1][3]),
        "%s %.2f n=%d" % (row[2][1], row[2][2], row[2][3])))
