# -*- coding: utf-8 -*-
"""用「人工量取的实测 XY」直接解手眼标定（不依赖棋盘格照片）。

思路：把物料放到画面里若干个不同的像素位置，量出（或读 32 显示）该处龙门
真实坐标 X/Y，得到 (u, v) -> (X, Y) 的对应点。3 个以上不共线的点就能最小
二乘解出 2x3 仿射矩阵（6 个未知数，2 个方程/点），4~6 个点更稳。

为什么必须至少有一个「绝对值」：
    只量两点之间的距离差，只能确定比例和旋转，确定不了平移（原点）。
    所以第一个点必须是绝对坐标（32 读数 / 点动读数 / 已知基准），
    其余点可以只给相对第一个点的距离差（--delta）。

用法（都在 /root/vision_ws 下）：
    python3 tools/xy_fit.py --add 320 240 120 40 --note "料盘左下"   # 加一个点对
    python3 tools/xy_fit.py --add 420 200 120 40 --delta             # 相对第一点给差
    python3 tools/xy_fit.py --list                                   # 看已采的点
    python3 tools/xy_fit.py --fit --dry-run                          # 只解算不落盘
    python3 tools/xy_fit.py --fit                                    # 解算并写 hand_eye.json
    python3 tools/xy_fit.py --probe 300 250                          # 反查某像素的毫米值
    python3 tools/xy_fit.py --self-test                              # 合成数据自检

注意：
  * 采点时相机参数（曝光/伽马/亮度）必须与生产时一致，否则判色和检测都会飘。
  * 像素坐标取【预览画面】上的坐标即可，工具会自己调 camera_model.undistort_point，
    与节点内 pixel_to_robot 的口径保持一致（节点是先 undistort 再乘矩阵）。
  * 矩阵写进 hand_eye.json 后，节点需要重跑 bash restart_vision.sh 才生效。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

K_DIR = "/root/vision_ws"
sys.path.insert(0, os.path.join(K_DIR, "vision_sorter"))

import camera_model                                    # noqa: E402

DEFAULT_STORE = os.path.join(K_DIR, "xy_pairs.json")
DEFAULT_OUT = os.path.join(K_DIR, "hand_eye.json")

# 残差门槛：3 个点解 6 个未知数几乎必过，4 个点以上才有说服力
RESIDUAL_MEAN_MAX = 1.0      # mm
RESIDUAL_MAX_MAX = 3.0       # mm
# 本机 640x480 / 24mm 格 35.5px -> 约 0.68 mm/px；放宽到 0.2~1.5 做体检
SCALE_RANGE = (0.20, 1.50)


def load_store(path):
    if not os.path.exists(path):
        return {"pairs": [], "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("pairs", [])
    return data


def save_store(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt_pair(i, p):
    return "  #%d  像素(%7.2f,%7.2f) -> 龙门(%8.2f,%8.2f)mm  %s" % (
        i, p["u"], p["v"], p["x"], p["y"], p.get("note", ""))


def do_add(args, store):
    u, v, x, y = args.add
    if args.delta:
        if not store["pairs"]:
            raise SystemExit("--delta 需要先有一个绝对点（还没有任何点对）")
        b = store["pairs"][0]
        x = b["x"] + x
        y = b["y"] + y
        print("[delta] 相对 #0 (%.1f,%.1f) 偏移 (%.1f,%.1f) -> 绝对 (%.1f,%.1f)"
              % (b["x"], b["y"], args.add[2], args.add[3], x, y))
    # 同一像素/同一龙门点重复采样就覆盖，避免手滑采两遍把拟合带偏
    keep = [p for p in store["pairs"]
            if not (abs(p["u"] - u) < 3.0 and abs(p["v"] - v) < 3.0)]
    if len(keep) != len(store["pairs"]):
        print("[覆盖] 像素 (%.1f,%.1f) 附近已有的点被替换" % (u, v))
    keep.append({"u": float(u), "v": float(v), "x": float(x), "y": float(y),
                 "note": args.note or "",
                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    store["pairs"] = keep
    save_store(args.store, store)
    print("[加] 现有 %d 个点对（存 %s）" % (len(keep), args.store))
    for i, p in enumerate(store["pairs"]):
        print(fmt_pair(i, p))
    if len(keep) >= 3:
        print("      够了，可以 --fit 了（点越多、越散开越准）")
    else:
        print("      还差 %d 个点（至少 3 个，建议 4~6 个，尽量铺开不成一条直线）"
              % (3 - len(keep)))


def do_list(args, store):
    if not store["pairs"]:
        print("还没有任何点对。用 --add u v X Y 加，例如：")
        print("  python3 tools/xy_fit.py --add 320 240 120 40 --note 料盘左下")
        return
    print("已采 %d 个点对（%s）：" % (len(store["pairs"]), args.store))
    for i, p in enumerate(store["pairs"]):
        print(fmt_pair(i, p))


def fit_matrix(pairs, undistort=True):
    """返回 (matrix 2x3, residuals(N,), uvf(N,2), 原像素 Nx2)"""
    raw = np.array([[p["u"], p["v"]] for p in pairs], dtype=np.float64)
    uvf = np.array([camera_model.undistort_point(u, v) for (u, v) in raw],
                   dtype=np.float64) if undistort else raw.copy()
    tgt = np.array([[p["x"], p["y"]] for p in pairs], dtype=np.float64)
    A = np.hstack([uvf, np.ones((len(uvf), 1))])        # (N,3)
    coef, _res, _rank, _sv = np.linalg.lstsq(A, tgt, rcond=None)   # (3,2)
    M = coef.T.copy()                                   # (2,3): X,Y 各一行
    pred = A @ coef
    res = np.linalg.norm(pred - tgt, axis=1)
    return M, res, uvf, raw


def describe_matrix(M):
    """返回 (mm/px 比例, 旋转角(度), 是否镜像)"""
    L = M[:, :2]
    det = float(np.linalg.det(L))
    sx = float(np.hypot(L[0, 0], L[1, 0]))
    sy = float(np.hypot(L[0, 1], L[1, 1]))
    scale = float(np.sqrt(abs(det)))
    ang = float(np.degrees(np.arctan2(L[1, 0], L[0, 0])))
    return scale, ang, det < 0, sx, sy


def do_fit(args, store):
    pairs = store["pairs"]
    if len(pairs) < 3:
        raise SystemExit("至少需要 3 个点对，现在只有 %d 个" % len(pairs))
    M, res, uvf, raw = fit_matrix(pairs, undistort=not args.no_undistort)

    # 共线性体检：像素点铺成一条线时仿射矩阵会病态
    uv = uvf - uvf.mean(axis=0)
    sv = np.linalg.svd(uv, compute_uv=False)
    cond = float(sv[0] / sv[1]) if sv[1] > 1e-9 else 1e9

    scale, ang, mirrored, sx, sy = describe_matrix(M)
    print("[解算] %d 个点对（%s）" % (len(pairs), "已去畸变" if not args.no_undistort else "未去畸变"))
    print("       matrix = [[%.6f, %.6f, %.4f], [%.6f, %.6f, %.4f]]" % (
        M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2]))
    print("       比例 %.4f mm/px（x向 %.4f / y向 %.4f）| 旋转 %.2f° | %s"
          % (scale, sx, sy, ang, "镜像" if mirrored else "不镜像"))
    print("       残差：平均 %.3fmm 最大 %.3fmm | 像素点条件数 %.1f"
          % (float(res.mean()), float(res.max()), cond))
    for i, (p, r) in enumerate(zip(pairs, res)):
        flag = "  << 偏差大" if r > RESIDUAL_MAX_MAX else ""
        print("        #%d 残差 %.3fmm%s" % (i, r, flag))

    bad = []
    if float(res.mean()) > RESIDUAL_MEAN_MAX:
        bad.append("平均残差 %.2fmm > %.1fmm" % (res.mean(), RESIDUAL_MEAN_MAX))
    if float(res.max()) > RESIDUAL_MAX_MAX:
        bad.append("最大残差 %.2fmm > %.1fmm" % (res.max(), RESIDUAL_MAX_MAX))
    if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
        bad.append("比例 %.3f mm/px 不在 %.2f~%.2f，像素/坐标可能配错了"
                   % (scale, SCALE_RANGE[0], SCALE_RANGE[1]))
    if cond > 30.0:
        bad.append("像素点条件数 %.0f 偏大：点几乎排成一条线，请围成三角形/四边形" % cond)
    if len(pairs) == 3:
        bad.append("只有 3 个点：6 个未知数被 6 个方程刚好定死，残差必然≈0，"
                   "看起来完美但不说明任何问题，建议至少再补 1 个点")

    if bad:
        print("\n[不合格]")
        for b in bad:
            print("   - %s" % b)
        if args.dry_run or not args.force:
            print("\n没有写 %s（确认要写就加 --force）" % args.out)
            return
    if args.dry_run:
        print("\n[dry-run] 没有写 %s" % args.out)
        return

    info = {
        "matrix": M.tolist(),
        "mean_residual": round(float(res.mean()), 4),
        "max_residual": round(float(res.max()), 4),
        "n_points": int(len(pairs)),
        "method": "xy-pairs（人工实测 XY 最小二乘仿射）",
        "scale_mm_per_px": round(scale, 5),
        "rotation_deg": round(ang, 3),
        "mirror": "mirrored" if mirrored else "none",
        "undistorted": bool(not args.no_undistort),
        "pairs": [{"u": p["u"], "v": p["v"], "x": p["x"], "y": p["y"],
                   "note": p.get("note", ""), "residual_mm": round(float(r), 3)}
                  for p, r in zip(pairs, res)],
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if os.path.exists(args.out) and not args.no_backup:
        bak = args.out + ".bak." + time.strftime("%m%d_%H%M")
        with open(args.out, "r", encoding="utf-8") as f:
            old = f.read()
        with open(bak, "w", encoding="utf-8") as f:
            f.write(old)
        print("\n[备份] 旧标定 -> %s" % bak)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[写入] %s" % args.out)
    print("       生效：bash /root/vision_ws/restart_vision.sh")


def do_probe(args, store):
    if len(store["pairs"]) < 3:
        raise SystemExit("至少需要 3 个点对才能反查")
    M, _res, _uvf, _raw = fit_matrix(store["pairs"], undistort=not args.no_undistort)
    u, v = args.probe
    uu, vv = camera_model.undistort_point(u, v) if not args.no_undistort else (u, v)
    X = M[0, 0] * uu + M[0, 1] * vv + M[0, 2]
    Y = M[1, 0] * uu + M[1, 1] * vv + M[1, 2]
    print("像素 (%.1f,%.1f) -> 去畸变 (%.2f,%.2f) -> 龙门 (%.1f,%.1f)mm"
          % (u, v, uu, vv, X, Y))
    print("（拿这个值去点动对一下：夹爪移到该像素对应位置，看 32 显示是不是这个数）")


def do_self_test():
    """合成 6 个点，用已知矩阵造 XY，看能不能原样解回来。"""
    true_M = np.array([[-0.6800, 0.0120, 332.50],
                       [0.0080, 0.6900, 175.20]])
    uv = np.array([[220.0, 200.0], [420.0, 205.0], [300.0, 330.0],
                   [450.0, 330.0], [250.0, 260.0], [400.0, 270.0]])
    A = np.hstack([uv, np.ones((len(uv), 1))])
    XY = A @ true_M.T
    pairs = [{"u": float(a), "v": float(b), "x": float(c), "y": float(d)}
             for (a, b), (c, d) in zip(uv, XY)]
    M, res, _uvf, _raw = fit_matrix(pairs, undistort=False)
    err = np.abs(M - true_M).max()
    scale, ang, mirrored, _sx, _sy = describe_matrix(M)
    print("[自检] 6 个合成点：矩阵最大误差 %.3e（应≈0）、残差最大 %.3e" % (err, res.max()))
    print("       比例 %.4f mm/px、旋转 %.2f°、%s" % (scale, ang, "镜像" if mirrored else "不镜像"))
    _tscale, _tang, _tmir, _sx, _sy = describe_matrix(true_M)
    print("       真值 比例 %.4f、旋转 %.2f°、%s"
          % (_tscale, _tang, "镜像" if _tmir else "不镜像"))
    # 加噪声，看看 1px 抖动会放大成多少毫米
    rng = np.random.RandomState(0)
    noisy = []
    for p in pairs:
        noisy.append({"u": p["u"] + rng.normal(0, 1.0),
                      "v": p["v"] + rng.normal(0, 1.0),
                      "x": p["x"], "y": p["y"]})
    M2, res2, _a, _b = fit_matrix(noisy, undistort=False)
    print("[自检] 像素加 1px 高斯噪声后：矩阵误差最大 %.4f mm，残差最大 %.2f mm"
          % (float(np.abs(M2 - true_M).max()), float(res2.max())))
    ok = err < 1e-8 and res.max() < 1e-8
    print("[自检] %s" % ("通过" if ok else "失败"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="用人工实测的 XY 点对直接解手眼标定（不依赖棋盘格）")
    ap.add_argument("--store", default=DEFAULT_STORE, help="点对文件")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 hand_eye.json")
    ap.add_argument("--add", nargs=4, type=float, metavar=("U", "V", "X", "Y"),
                    help="追加一个点对：像素 u v + 龙门 X Y(mm)")
    ap.add_argument("--delta", action="store_true",
                    help="配合 --add：x y 给的是相对第一个点的距离差，不是绝对值")
    ap.add_argument("--note", default="", help="这个点的备注（例如物料种类/位置）")
    ap.add_argument("--list", action="store_true", help="列出已采点对")
    ap.add_argument("--del", dest="delete", type=int, default=None, help="删掉第 N 个点")
    ap.add_argument("--clear", action="store_true", help="清空所有点对")
    ap.add_argument("--fit", action="store_true", help="最小二乘解算并写 hand_eye.json")
    ap.add_argument("--probe", nargs=2, type=float, metavar=("U", "V"),
                    help="用已采点对反查某像素的毫米坐标")
    ap.add_argument("--dry-run", action="store_true", help="只解算不落盘")
    ap.add_argument("--force", action="store_true", help="残差/体检不合格也照写")
    ap.add_argument("--no-backup", action="store_true", help="不备份旧 hand_eye.json")
    ap.add_argument("--no-undistort", action="store_true",
                    help="不调 camera_model 去畸变（内参不对时用，一般别用）")
    ap.add_argument("--self-test", action="store_true", help="合成数据自检")
    args = ap.parse_args(argv)

    if args.self_test:
        return do_self_test()

    store = load_store(args.store)
    touched = False
    if args.clear:
        store["pairs"] = []
        touched = True
        print("[清空] 点对已清空")
    if args.delete is not None:
        if 0 <= args.delete < len(store["pairs"]):
            gone = store["pairs"].pop(args.delete)
            touched = True
            print("[删] #%d %s" % (args.delete, gone))
        else:
            print("[删] 序号 %d 越界（0~%d）" % (args.delete, len(store["pairs"]) - 1))
    if args.add:
        do_add(args, store)
        return 0
    if touched:
        save_store(args.store, store)
    if args.list:
        do_list(args, store)
        return 0
    if args.fit:
        do_fit(args, store)
        return 0
    if args.probe:
        do_probe(args, store)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
