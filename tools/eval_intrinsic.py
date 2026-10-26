#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内参可信度评估 / 重解算（用棋盘格照片挑出真正对的那组 K、D）

问题背景
--------
同一台相机算出两组完全不同的内参时，只看 cv2.calibrateCamera 的
reproj_error 是不够的：焦距和径向畸变会互相补偿，
样本姿态不够丰富时"重投影误差更小"的那组，反而可能是错的。
典型症状：fx 变小、k1 从 +0.9 掉到 +0.04，两者正好互相抵消。

判据（不依赖标定本身的自由度）
------------------------------
棋盘是平面：正确去畸变之后，所有角点必须能被一个 2D 单应变换精确解释。
    u = undistortPoints(c, K, D, P=K)
    H = findHomography(objp_2d, u)
    resid = |u - H·objp_2d|
畸变模型错了，残余就会从亚像素涨到几个像素。这个量只看"网格还直不直"，
不参与任何参数拟合，所以能区分两组内参谁对。

用法
----
    python tools/eval_intrinsic.py --captures capture/ \
        --candidates camera_intrinsic.json camera_intrinsic_new.json \
        --square 24 --recalib-out camera_intrinsic_recalib.json
"""
import argparse
import glob
import io
import json
import os
import sys

import cv2
import numpy as np

COLS, ROWS = 9, 6


def detect(gray):
    ok, c = cv2.findChessboardCorners(
        gray, (COLS, ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        return True, cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
    try:
        ok, c = cv2.findChessboardCornersSB(gray, (COLS, ROWS), cv2.CALIB_CB_ACCURACY)
        if ok:
            return True, c.astype(np.float32)
    except AttributeError:
        pass
    return False, None


def load_corners(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")) +
                   glob.glob(os.path.join(folder, "*.jpg")))
    out = []
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ok, c = detect(img)
        out.append((os.path.basename(p), ok, c, img.shape))
    return out


def homography_residual(corners, K, D):
    """去畸变后网格还直不直 —— 返回 (mean_px, max_px)"""
    objp2 = np.zeros((ROWS * COLS, 2), np.float32)
    objp2[:, :] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    u = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    H, _ = cv2.findHomography(objp2, u, 0)
    if H is None:
        return float("inf"), float("inf")
    proj = cv2.perspectiveTransform(objp2.reshape(-1, 1, 2), H).reshape(-1, 2)
    d = np.linalg.norm(u - proj, axis=1)
    return float(d.mean()), float(d.max())


def edge_correction(K, D, w=640, h=480):
    """画面四个角、四条边中点被去畸变挪了多少像素（安全边界评估）"""
    pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                    [w / 2.0, 0], [w / 2.0, h - 1], [0, h / 2.0], [w - 1, h / 2.0]],
                   np.float32).reshape(-1, 1, 2)
    u = cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)
    d = np.linalg.norm(u - pts.reshape(-1, 2), axis=1)
    return float(d[:4].max()), float(d[4:].max())


def fit(corners_list, square, size, flags=0):
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    objp *= float(square)
    op = [objp.copy() for _ in corners_list]
    ip = [c.astype(np.float32) for c in corners_list]
    r, m, d, rv, tv = cv2.calibrateCamera(op, ip, size, None, None, flags=flags)
    errs = []
    for i in range(len(ip)):
        pr, _ = cv2.projectPoints(op[i], rv[i], tv[i], m, d)
        errs.append(cv2.norm(ip[i], pr, cv2.NORM_L2) / len(pr))
    return float(np.mean(errs)), m, d, errs, rv


# 畸变模型复杂度候选：纸张不平时，切向项 (p1,p2) 会把"板子弯了"吸收进去，
# 结果拿到一组各向异性的大切向畸变（p1 ~ -0.02，真实镜头通常 <0.005）。
# 这里把几种约束都解一遍，用"网格还直不直"来挑，而不是看 reproj_error。
VARIANTS = (
    ("free", 0),
    ("no_tangent", cv2.CALIB_ZERO_TANGENT_DIST),
    ("no_tangent_no_k3", cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3),
    ("radial_only", cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 |
     cv2.CALIB_FIX_PRINCIPAL_POINT),
)
VARIANT_FLAGS = dict(VARIANTS)


def radial_profile(K, D):
    """前向径向 f(r)=r(1+k1r2+k2r4+k3r6) 的单调上限，以及覆盖统计

    返回 (可逆半径px, 有效半径(归一化)=样本角点覆盖 p90+0.08)
    """
    k = np.array(D, np.float64).ravel()
    k1, k2, k3 = k[0], k[1], (k[4] if len(k) > 4 else 0.0)
    rr = np.arange(0.02, 3.0, 0.002)
    f = rr * (1 + k1 * rr ** 2 + k2 * rr ** 4 + k3 * rr ** 6)
    i = int(np.argmax(f))
    return float(f[i] * K[0][0]), float(rr[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--square", type=float, default=24.0)
    ap.add_argument("--recalib-out", default="")
    ap.add_argument("--recalib-flags", default="free", choices=list(VARIANT_FLAGS),
                    help="重解算用的畸变模型约束；no_tangent = 只留 k1,k2,k3（各向同性、必定可逆）")
    ap.add_argument("--try-variants", action="store_true",
                    help="把几种畸变模型约束都解一遍（看切向项是不是在吸收纸张不平）")
    ap.add_argument("--hi-r", type=float, default=0.55,
                    help="高半径子集门槛（归一化半径，1.0=画面半对角线）")
    args = ap.parse_args()

    data = load_corners(args.captures)
    ok_data = [d for d in data if d[1]]
    print("captures: %d found %d" % (len(data), len(ok_data)))
    miss = [d[0] for d in data if not d[1]]
    if miss:
        print("  not detected:", " ".join(miss))

    if len(ok_data) < 6:
        print("too few; abort")
        return 1

    h, w = ok_data[0][3]
    size = (w, h)
    cands = []
    for p in args.candidates:
        with io.open(p, encoding="utf-8") as f:
            j = json.load(f)
        cands.append((os.path.basename(p), np.array(j["mtx"], np.float64),
                      np.array(j["dist"], np.float64).ravel(), j))

    if args.recalib_out:
        flags = VARIANT_FLAGS.get(args.recalib_flags, 0)
        mean_err, m, d, errs, rv = fit([x[2] for x in ok_data], args.square, size, flags)
        print("\n[recalib] %d images, model=%s, reproj %.4f px, fx=%.2f fy=%.2f"
              % (len(ok_data), args.recalib_flags, mean_err, m[0][0], m[1][1]))
        rv = np.array(rv).reshape(-1, 3)
        # 姿态多样性：标定可辨识的前提是棋盘有足够大的俯仰/偏转
        tilt = np.degrees(np.linalg.norm(rv[:, :2], axis=1))
        print("[recalib] tilt deg: min %.1f  median %.1f  max %.1f"
              % (tilt.min(), np.median(tilt), tilt.max()))
        # 径向覆盖：解算时只看"角点被采到过多大的半径"，决定去畸变结果哪里可信
        cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
        half = float(np.hypot(cx0, cy0))
        flat = np.vstack([c.reshape(-1, 2) for c in [x[2] for x in ok_data]])
        rr_n = np.hypot(flat[:, 0] - cx0, flat[:, 1] - cy0) / half
        cov = {"p50": round(float(np.percentile(rr_n, 50)), 3),
               "p90": round(float(np.percentile(rr_n, 90)), 3),
               "max": round(float(rr_n.max()), 3)}
        inv_px, inv_r = radial_profile(m, d)
        valid_r = round(min(inv_r, cov["p90"] + 0.08), 3)
        print("[recalib] radial coverage p50=%.2f p90=%.2f max=%.2f | "
              "模型可逆到 %.0fpx (r=%.2f) | valid_radius=%.2f"
              % (cov["p50"], cov["p90"], cov["max"], inv_px, inv_r, valid_r))
        jj = {"mtx": m.tolist(), "dist": d.ravel().tolist(),
              "reproj_error": mean_err, "size": [int(w), int(h)],
              "chessboard": [COLS, ROWS, float(args.square)],
              "n_images": len(ok_data),
              "model": args.recalib_flags,
              "radial_coverage": cov,
              "invertible_radius_px": round(inv_px, 1),
              "valid_radius": valid_r,
              "per_image_err": [round(float(e), 4) for e in errs],
              "images": [x[0] for x in ok_data],
              "note": "recalibrated by tools/eval_intrinsic.py (%s)"
                      % args.recalib_flags}
        with io.open(args.recalib_out, "w", encoding="utf-8") as f:
            json.dump(jj, f, indent=2, ensure_ascii=False)
        print("[recalib] written -> %s" % args.recalib_out)
        cands.append(("recalib", m, d, jj))

    if args.try_variants:
        print("\n[variants] 同一批图像、不同畸变模型约束：")
        for name, fl in VARIANTS:
            try:
                me, m, d, _e, rv = fit([x[2] for x in ok_data], args.square, size, fl)
            except cv2.error as ex:
                print("  %-18s FAILED %s" % (name, ex))
                continue
            cands.append((name, m, d, {"reproj_error": me}))
            print("  %-18s reproj %.4f px  fx=%.2f fy=%.2f  p1=%+.5f p2=%+.5f"
                  % (name, me, m[0][0], m[1][1], d.ravel()[2], d.ravel()[3]))

    if not cands:
        return 0

    # 高半径子集：这些图才真正约束径向项，也是实际工作区外沿
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    half = float(np.hypot(cx0, cy0))
    hi_idx = []
    for i, (_, _, c, _) in enumerate(ok_data):
        cc = c.reshape(-1, 2)
        if float(np.hypot(cc[:, 0] - cx0, cc[:, 1] - cy0).max()) / half >= args.hi_r:
            hi_idx.append(i)
    print("\nhigh-r subset (r_norm >= %.2f): %d/%d images"
          % (args.hi_r, len(hi_idx), len(ok_data)))

    print("\n%-22s %9s %9s %9s %9s %9s %8s %8s %8s" %
          ("candidate", "fx", "fy", "cx", "cy", "k1", "reproj", "H-resid", "H-hi"))
    print("-" * 102)
    rows = []
    for name, K, D, j in cands:
        res = []
        for _, _, c, _ in ok_data:
            res.append(homography_residual(c, K, D))
        hm = float(np.mean([r[0] for r in res]))
        hx = float(np.max([r[1] for r in res]))
        hhi = (float(np.mean([res[i][0] for i in hi_idx])) if hi_idx
               else float("nan"))
        corner_move, edge_move = edge_correction(K, D, w, h)
        rep = float(j.get("reproj_error", float("nan")))
        print("%-22s %9.2f %9.2f %9.2f %9.2f %9.4f %8.3f %8.3f %8.3f"
              % (name, K[0][0], K[1][1], K[0][2], K[1][2], D.ravel()[0], rep, hm, hhi))
        rows.append((name, hm, hx, hhi, corner_move, edge_move))

    print("\n%-22s %10s %10s %10s %12s %12s" %
          ("candidate", "H-mean", "H-max", "H-hi", "corner_move", "edge_move"))
    print("-" * 82)
    for name, hm, hx, hhi, cm, em in rows:
        print("%-22s %10.3f %10.3f %10.3f %12.1f %12.1f"
              % (name, hm, hx, hhi, cm, em))
    print("\nH-resid 越小说明去畸变后网格越直（判据不看标定拟合自由度）。")
    print("H-hi 只看半径大的那批图，这才是外沿能不能信的关键。")
    print("corner_move/edge_move 是画面角、边中点在去畸变时被挪动的像素数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())