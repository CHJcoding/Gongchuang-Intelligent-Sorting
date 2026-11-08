#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上棋盘格内参标定服务（无显示器，浏览器看画面）

为什么不用 tools/camera_intrinsic_calib.py：
  那个版本靠 cv2.imshow，板子没桌面；而且它拍一张要按一次空格，
  串口控制台打字会被吞。这里改成：
    - 板子直接开 /dev/video0（独占，标定期间不要起 ROS 管线）
    - 实时检测 9x6 内角点，浏览器 http://<ip>:8080 看带角点叠加的画面
    - 自动抓拍：检测到棋盘格、且是"没见过的姿态"就自动存一张，存够就自动解算
    - 全程不用键盘（伸 /capture /calibrate /quit 也能手动推）

终端输出一律 ASCII（串口控制台中文会乱码），网页里才用中文。

用法：
    python3 board_calib_server.py                 # 默认 8080 端口，自动抓拍
    python3 board_calib_server.py --port 8080 --square 21
"""
import argparse
import glob
import io
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

COLS, ROWS = 9, 6          # 内角点
GRID = 3                   # 画面 3x3 分区，用来催用户覆盖每个区域
MIN_SAMPLES = 12           # 至少这么多张才解算
MAX_SAMPLES = 26           # 到这么多就不再自动抓
NOVEL_PX = 45.0            # 角点平均位移超过这么多像素才算"新姿态"
MIN_BOARD_PX = 110.0       # 棋盘格包围盒最小边长，太小角点不准
MARGIN_PX = 6              # 角点离画面边缘至少这么多像素
SAVE_COOLDOWN = 2.0        # 两张之间至少隔这么久（秒）
# 有效区域：用户明确说了真实物料盘只占画面中心大部分，不会铺满全幅。
# 所以不再要求把板子推到画面角落（物理上也做不到既推角又保证 9x6 全在框内），
# 只要求板子尽量大：最远角点至少到画面半径的 62%（≈ 板子占画面 1/3 面积）。
# 解算后会给出实际覆盖 p90，超过它的像素位置一律视为外推，生产端会按 r_valid 拒发。
EDGE_R = 0.62              # 保存门槛：最远角点至少到画面半径的 62%
QUAD_NAMES = ["左上", "右上", "左下", "右下"]

OUT_DIR = "/root/vision_ws/capture"
NEW_JSON = "/root/vision_ws/camera_intrinsic_new.json"
OLD_JSON = "/root/vision_ws/camera_intrinsic.json"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


class State(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpg = b""          # 最新叠加图（给浏览器）
        self.found = False
        self.samples = []             # [(path, corners_2d float32)]
        self.cells = set()            # 已覆盖的分区 index
        self.last_save_t = 0.0
        self.last_msg = "waiting for chessboard"
        self.done = False
        self.result = None
        self.fps = 0.0
        self.quit = False
        # 取图线程与检测线程之间共享
        self.gray = None
        self.gray_ts = 0.0
        self.corners = None
        self.corners_ts = 0.0
        self.det_fps = 0.0
        self.edge_quads = set()       # 已覆盖的四角（各需一张 r>=EDGE_STRONG）


def annotate(frame, corners, st, cells):
    disp = frame.copy()
    h, w = disp.shape[:2]
    cw, ch = w // GRID, h // GRID
    # 未覆盖区域涂暗
    for gy in range(GRID):
        for gx in range(GRID):
            if (gy * GRID + gx) not in cells:
                x0, y0 = gx * cw, gy * ch
                roi = disp[y0:y0 + ch, x0:x0 + cw]
                disp[y0:y0 + ch, x0:x0 + cw] = (roi * 0.55).astype(np.uint8)
    if corners is not None:
        cv2.drawChessboardCorners(disp, (COLS, ROWS), corners, True)
        c = corners.reshape(-1, 2)
        r, _ = radial_stats(corners, w, h)
        col = (0, 255, 0) if r >= EDGE_R else (0, 200, 255)
        cv2.putText(disp, "FOUND r=%.0f%%" % (100 * r), (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    else:
        cv2.putText(disp, "NO CHESSBOARD", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    # 四角指示：亮=该象限至少采到过一张（只为提醒“四个方向都摆一下”，不是硬门槛）
    for q in range(4):
        qx = 0 if q in (0, 2) else w - 34
        qy = 0 if q in (0, 1) else h - 34
        on = q in st.edge_quads
        cv2.rectangle(disp, (qx + 2, qy + 2), (qx + 32, qy + 32),
                      (0, 255, 0) if on else (60, 60, 200), -1)
    n = len(st.samples)
    cv2.putText(disp, "samples %d/%d  poses %d/4" % (n, MIN_SAMPLES, len(st.edge_quads)),
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(disp, st.last_msg, (10, 86), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 2)
    return disp


def board_center_cell(corners, w, h):
    m = corners.reshape(-1, 2).mean(axis=0)
    gx = min(GRID - 1, max(0, int(m[0] / (w / float(GRID)))))
    gy = min(GRID - 1, max(0, int(m[1] / (h / float(GRID)))))
    return gy * GRID + gx


def cells_touched(corners, w, h):
    """角点落到哪些分区：用"任意一个角点进过该区"来算覆盖

    不能用棋盘格中心：24mm 格子印出来整块板有 240x168mm，靠近时几乎占满画面，
    中心永远在中间那一格，覆盖度永远不涨，用户会被逼着采满 30 张也覆盖不到边缘——
    而边缘恰恰是径向畸变最需要约束的地方。改成看角点，就等于要求"把板子推到画面四角去"。
    """
    c = corners.reshape(-1, 2)
    out = set()
    for x, y in c:
        gx = min(GRID - 1, max(0, int(x / (w / float(GRID)))))
        gy = min(GRID - 1, max(0, int(y / (h / float(GRID)))))
        out.add(gy * GRID + gx)
    return out


def radial_stats(corners, w, h):
    """最远角点的归一化半径（1.0 = 画面四角）和它所在的象限

    半径按"到画面中心距离 / 中心到画角的距离"归一化，这样 0.72 就代表
    "已经伸到画面边缘附近"，和画面宽高比无关。
    """
    c = corners.reshape(-1, 2)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.hypot(c[:, 0] - cx, c[:, 1] - cy) / np.hypot(cx, cy)
    i = int(np.argmax(r))
    x, y = c[i]
    quad = (0 if x < cx else 1) + (0 if y < cy else 2)   # 0左上 1右上 2左下 3右下
    return float(r[i]), quad


def is_novel(corners, saved):
    if not saved:
        return True
    c = corners.reshape(-1, 2)
    best = 1e9
    for _, s in saved:
        best = min(best, float(np.mean(np.linalg.norm(c - s.reshape(-1, 2), axis=1))))
    return best > NOVEL_PX


def detect(gray, fast=True):
    """返回 (found, corners)

    实测（X5，640x480）：SB 各种 flag 都要 ~790ms，经典法+亚像素只要 126ms。
    所以默认走经典法；经典法失败时（模糊/边缘强畸变）再用 SB 兜底，
    而且兜底限流，别把画面帧率拖死。
    """
    ok, c = cv2.findChessboardCorners(
        gray, (COLS, ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
        return True, c
    if fast:
        return False, None
    try:
        ok, c = cv2.findChessboardCornersSB(gray, (COLS, ROWS), cv2.CALIB_CB_ACCURACY)
        if ok:
            return True, c.astype(np.float32)
    except AttributeError:
        pass
    return False, None


def calibrate(st, square):
    """从已存 PNG 重新检测角点并解算（可复现，不依赖内存状态）"""
    paths = [p for p, _ in st.samples]
    if len(paths) < 6:
        return None, "need >=6 samples"
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    objp *= float(square)

    op, ip, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        ok, c = detect(img)
        if not ok:
            # 存图时可能用的是 SB 兜底检出的，这里必须也退到 SB，否则会白白丢样本
            ok, c = detect(img, fast=False)
        if not ok:
            log("  skip (no corners): %s" % os.path.basename(p))
            continue
        op.append(objp.copy())
        ip.append(c.astype(np.float32))
        used.append(p)
    if len(op) < 6:
        return None, "only %d usable" % len(op)

    def run(ipp):
        r, m, d, rv, tv = cv2.calibrateCamera(op[:len(ipp)], ipp, size, None, None)
        errs = []
        for i in range(len(ipp)):
            pr, _ = cv2.projectPoints(op[i], rv[i], tv[i], m, d)
            errs.append(cv2.norm(ipp[i], pr, cv2.NORM_L2) / len(pr))
        return float(np.mean(errs)), m, d, errs

    mean_err, mtx, dist, errs = run(ip)
    # 一轮剔除：重投影明显偏大的图（晃了/糊了）
    med = float(np.median(errs))
    keep = [i for i, e in enumerate(errs) if e <= max(1.5, 2.5 * med)]
    if 6 <= len(keep) < len(errs):
        log("  prune %d outlier(s) (median %.3f px)" % (len(errs) - len(keep), med))
        op = [op[i] for i in keep]
        ip = [ip[i] for i in keep]
        used = [used[i] for i in keep]
        mean_err, mtx, dist, errs = run(ip)

    result = {
        "mtx": mtx.tolist(),
        "dist": dist.ravel().tolist(),
        "reproj_error": mean_err,
        "size": [int(size[0]), int(size[1])],
        "chessboard": [COLS, ROWS, float(square)],
        "n_images": len(ip),
        "per_image_err": [round(float(e), 4) for e in errs],
        "images": [os.path.basename(p) for p in used],
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "board_calib_server.py auto capture",
    }
    # 记录径向覆盖，下次一眼就能看出这组内参的边缘是实测还是外推
    cx, cy = (size[0] - 1) / 2.0, (size[1] - 1) / 2.0
    half = float(np.hypot(cx, cy))
    flat = np.vstack([c.reshape(-1, 2) for c in ip])
    rr = np.hypot(flat[:, 0] - cx, flat[:, 1] - cy) / half
    result["radial_coverage"] = {
        "p50": round(float(np.percentile(rr, 50)), 3),
        "p90": round(float(np.percentile(rr, 90)), 3),
        "max": round(float(rr.max()), 3),
    }
    # 可信半径：实测覆盖的 p90 再留 8% 余量，超出的像素位置属于外推，
    # 生产端（camera_model）按它判断“这个点的去畸变结果能不能信”。
    result["valid_radius"] = round(min(1.0, float(np.percentile(rr, 90)) + 0.08), 3)
    return result, None


def compare_with_old(new):
    try:
        with io.open(OLD_JSON, encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return "no old json"
    om = old.get("mtx")
    nm = new["mtx"]
    if not om:
        return "old json has no mtx"
    ofx, ofy = om[0][0], om[1][1]
    nfx, nfy = nm[0][0], nm[1][1]
    return ("fx %.1f -> %.1f (%+.1f%%)  fy %.1f -> %.1f (%+.1f%%)  "
            "reproj %.3f -> %.3f px  [%d imgs]"
            % (ofx, nfx, 100.0 * (nfx - ofx) / ofx,
               ofy, nfy, 100.0 * (nfy - ofy) / ofy,
               float(old.get("reproj_error", -1)), new["reproj_error"],
               new["n_images"]))


def save_sample(frame, corners, st):
    os.makedirs(OUT_DIR, exist_ok=True)
    idx = len(st.samples) + 1
    path = os.path.join(OUT_DIR, "calib_%02d.png" % idx)
    cv2.imwrite(path, frame)          # PNG 无损，避免 jpeg 振铃影响角点
    st.samples.append((path, corners.astype(np.float32).copy()))
    st.cells |= cells_touched(corners, frame.shape[1], frame.shape[0])
    r, quad = radial_stats(corners, frame.shape[1], frame.shape[0])
    if r >= EDGE_R:
        st.edge_quads.add(quad)
    st.last_save_t = time.time()
    log("  saved #%d r=%.0f%% quad=%s poses=%d/4 -> %s"
        % (idx, 100 * r, QUAD_NAMES[quad], len(st.edge_quads),
           os.path.basename(path)))


def grab_loop(cap, st, args):
    """只负责取图 + 编码推流（检测在另一个线程，别把帧率拖死）

    USB 相机在这块板子上会偶发掉线重枚举（dmesg: USB disconnect ->
    Failed to resubmit video URB (-19) -> 重新枚举，而且节点号会从 video0
    漂到 video1/video2，因为旧 fd 还占着 minor）。所以读失败不能就这么干等，
    必须关掉句柄重新扫描节点、重新打开。
    """
    t0, nframe, fails = time.time(), 0, 0
    while not st.quit:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails % 15 == 0:
                log("camera read failed x%d, reopening..." % fails)
                try:
                    cap.release()
                except Exception:
                    pass
                nc = open_camera(args)
                if nc is not None:
                    cap = nc
                    fails = 0
                    t0, nframe = time.time(), 0
                    log("camera reopened")
                else:
                    log("reopen failed, retrying in 2s")
                    time.sleep(2.0)
            time.sleep(0.05)
            continue
        fails = 0
        nframe += 1
        if nframe % 20 == 0:
            st.fps = nframe / max(1e-6, time.time() - t0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with st.lock:
            st.gray = gray
            st.gray_ts = time.time()
            fresh = (time.time() - st.corners_ts) < 0.8
            corners = st.corners if fresh else None
            disp = annotate(gray, corners, st, st.cells)
            ok2, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok2:
                st.frame_jpg = buf.tobytes()
    cap.release()


def detect_loop(st, args):
    """检测 + 自动抓拍 + 判断何时收工"""
    last_ts, n, t0 = -1.0, 0, time.time()
    last_sb = 0.0
    while not st.quit:
        with st.lock:
            gray = st.gray
            ts = st.gray_ts
        if gray is None or ts == last_ts:
            time.sleep(0.01)
            continue
        last_ts = ts
        found, corners = detect(gray)
        if not found and time.time() - last_sb > 1.0:
            last_sb = time.time()
            found, corners = detect(gray, fast=False)   # SB 兜底，限流 1s
        n += 1
        if n % 10 == 0:
            st.det_fps = n / max(1e-6, time.time() - t0)
        with st.lock:
            st.found = found
            st.corners = corners if found else None
            st.corners_ts = time.time()
            if found and len(st.samples) < MAX_SAMPLES:
                c = corners.reshape(-1, 2)
                h, w = gray.shape
                inside = (c[:, 0] > MARGIN_PX) & (c[:, 0] < w - MARGIN_PX) & \
                         (c[:, 1] > MARGIN_PX) & (c[:, 1] < h - MARGIN_PX)
                bw = float(np.ptp(c[:, 0]))
                bh = float(np.ptp(c[:, 1]))
                rmax, quad = radial_stats(corners, w, h)
                if not inside.all():
                    st.last_msg = "board touches edge, pull back a bit"
                elif min(bw, bh) < MIN_BOARD_PX:
                    st.last_msg = "board too small, move closer"
                elif rmax < EDGE_R:
                    st.last_msg = ("board too small (r=%.0f%%, need %.0f%%): "
                                   "step closer / zoom so it fills ~1/3 of the frame"
                                   % (100 * rmax, 100 * EDGE_R))
                elif time.time() - st.last_save_t < SAVE_COOLDOWN:
                    st.last_msg = "holding... move to a new pose"
                elif is_novel(corners, st.samples):
                    st.last_msg = "captured, next pose"
                    save_sample(gray, corners, st)
                else:
                    st.last_msg = "same pose, change tilt/position"
        if len(st.samples) >= MIN_SAMPLES:
            log("collected %d samples, calibrating..." % len(st.samples))
            break
        if len(st.samples) >= MAX_SAMPLES:
            log("reached MAX_SAMPLES, calibrating...")
            break
    with st.lock:
        st.done = True
        st.last_msg = "capture finished"
        st.quit = True


def do_calib(st, args):
    log("calibrating with %d images (square=%.2fmm)..." % (len(st.samples), args.square))
    res, err = calibrate(st, args.square)
    if res is None:
        log("calibration FAILED: %s" % err)
        with st.lock:
            st.last_msg = "calibration failed: " + err
        return
    with io.open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    m = res["mtx"]
    log("reproj error: %.4f px over %d images" % (res["reproj_error"], res["n_images"]))
    log("fx=%.3f fy=%.3f cx=%.3f cy=%.3f" % (m[0][0], m[1][1], m[0][2], m[1][2]))
    log("dist: " + " ".join("%+.6f" % d for d in res["dist"]))
    cmp_txt = compare_with_old(res)
    log("vs old: " + cmp_txt)
    rc = res["radial_coverage"]
    log("radial coverage: p50=%.2f p90=%.2f max=%.2f (1.0 = frame corner)"
        % (rc["p50"], rc["p90"], rc["max"]))
    if rc["p90"] < 0.55:
        log("WARNING: p90 < 0.55 -> samples too central, radial terms weakly constrained;")
        log("WARNING: treat r > p90 as extrapolation (production should refuse those).")
    log("saved -> " + NEW_JSON)
    with st.lock:
        st.result = res
        st.last_msg = "OK reproj %.3f px" % res["reproj_error"]
    with io.open("/root/vision_ws/calib_report.txt", "w", encoding="utf-8") as f:
        f.write("reproj_error=%.4f px\nn_images=%d\n" % (res["reproj_error"], res["n_images"]))
        f.write("fx=%.4f fy=%.4f cx=%.4f cy=%.4f\n" % (m[0][0], m[1][1], m[0][2], m[1][2]))
        f.write("dist=" + json.dumps(res["dist"]) + "\n")
        f.write("vs_old=" + cmp_txt + "\n")
        f.write("per_image_err=" + json.dumps(res["per_image_err"]) + "\n")


HTML = u"""<!doctype html><html><head><meta charset="utf-8">
<title>棋盘格内参标定</title>
<style>body{background:#111;color:#eee;font-family:monospace;margin:0;padding:12px}
img{width:100%;max-width:960px;border:1px solid #444}
.b{font-size:15px;line-height:1.7} .k{color:#4fc3f7} .w{color:#ffd54f} .g{color:#81c784}</style>
</head><body>
<div class="b">
<b>棋盘格内参标定 · 第二轮（补边缘）</b>（9x7 格 / 9x6 内角点）<br>
<span class="w">关键：把板子推到画面四个角，四个角都要到。</span>
画面角落那 4 个小方块亮起来 = 该象限已采到（要伸到半径 85%）。<br>
做法：<span class="k">退远一点让板子变小</span>（宽度 150~300px），然后往上左、上右、下左、下右各推一次，
每个位置再<span class="w">倾斜 ±30°</span>。板子被画面边缘切到会提示 pull back。<br>
到 12 张且四角全亮就自动解算，<span class="g">不用按键</span>。
</div>
<img id="v" src="/stream?t=0">
<div class="b" id="s">…</div>
<script>
setInterval(function(){
  fetch('/stats').then(r=>r.json()).then(function(j){
    document.getElementById('s').innerHTML =
      '样本 <b>'+j.samples+'</b> / '+j.need+' &nbsp; 四角 <b>'+j.corners+'</b>/4 &nbsp; 分区 '+j.cells+'/'+j.grid+
      ' &nbsp; fps '+j.fps.toFixed(1)+' &nbsp; <span class="w">'+j.msg+'</span>'+
      (j.result? '<br><span class="g">解算完成: reproj '+j.result.reproj_error.toFixed(3)+' px</span>':'');
    if(j.done) document.getElementById('v').style.opacity=0.99;
  }).catch(function(){});
},1000);
</script></body></html>"""


def serve(st, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(8)
    log("http://<board-ip>:%d  ready" % port)

    def handle(conn):
        try:
            conn.settimeout(10)
            req = b""
            while b"\r\n\r\n" not in req and len(req) < 8192:
                b = conn.recv(4096)
                if not b:
                    break
                req += b
            line = req.split(b"\r\n")[0].decode("latin-1")
            path = line.split(" ")[1] if len(line.split(" ")) > 1 else "/"
            if path.startswith("/stream"):
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=fr\r\n"
                             b"Cache-Control: no-store\r\n\r\n")
                last = None
                while not st.quit:
                    with st.lock:
                        jpg = st.frame_jpg
                    if jpg and jpg is not last:
                        last = jpg
                        conn.sendall(b"--fr\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                                     + jpg + b"\r\n")
                    time.sleep(0.05)
                return
            body = b""
            ctype = "text/html; charset=utf-8"
            if path.startswith("/stats"):
                with st.lock:
                    j = {"samples": len(st.samples), "need": MIN_SAMPLES,
                         "cells": len(st.cells), "grid": GRID * GRID, "corners": len(st.edge_quads),
                         "fps": st.fps, "det_fps": st.det_fps, "found": st.found, "msg": st.last_msg,
                         "done": st.done, "result": st.result}
                body = json.dumps(j).encode()
                ctype = "application/json"
            elif path.startswith("/capture"):
                with st.lock:
                    n = len(st.samples)
                body = json.dumps({"before": n, "now": len(st.samples)}).encode()
                ctype = "application/json"
            elif path.startswith("/calibrate"):
                if not st.done:
                    threading.Thread(target=do_calib, args=(st, ARGS), daemon=True).start()
                body = b'{"started":true}'
                ctype = "application/json"
            elif path.startswith("/quit"):
                st.quit = True
                body = b"bye"
            elif path.startswith("/snap"):
                with st.lock:
                    jpg = st.frame_jpg
                body = jpg
                ctype = "image/jpeg"
            else:
                body = HTML.encode("utf-8")
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: " + ctype.encode() +
                         b"\r\nContent-Length: " + str(len(body)).encode() +
                         b"\r\nCache-Control: no-store\r\n\r\n" + body)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    while not st.quit:
        try:
            c, _ = srv.accept()
        except Exception:
            break
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def candidate_devices(preferred):
    """相机掉线重枚举后节点号会漂（video0 -> video1/video2），所以不能死认一个节点。

    旧句柄占着 minor 时，内核会把新注册的节点排到下一个空闲号，
    所以这里把 /dev/video* 全试一遍，用“能不能真的读出 640x480 帧”来判定。
    """
    out = []
    if preferred:
        out.append(preferred)
    for p in sorted(glob.glob("/dev/video*")):
        if p not in out:
            out.append(p)
    return out


def open_camera(args):
    for dev in candidate_devices(args.device):
        for backend, fourcc, label in (
                (cv2.CAP_V4L2, "MJPG", "v4l2+mjpg"),
                (cv2.CAP_V4L2, "YUYV", "v4l2+yuyv"),
                (0, None, "default")):
            try:
                cap = (cv2.VideoCapture(dev, backend) if backend
                       else cv2.VideoCapture(dev))
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, fr = False, None
            for _ in range(3):        # 刚打开的节点头几帧常是空的
                ok, fr = cap.read()
                if ok and fr is not None:
                    break
                time.sleep(0.1)
            if ok and fr is not None:
                cc = int(cap.get(cv2.CAP_PROP_FOURCC))
                log("camera open: %s %s %dx%d fourcc=%s" % (
                    dev, label, fr.shape[1], fr.shape[0],
                    "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4))))
                if dev != args.device:
                    log("NOTE: %s not usable, using %s instead" % (args.device, dev))
                return cap
            cap.release()
    return None


ARGS = None


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--square", type=float, default=21.0, help="格子边长 mm（只影响平移量，不影响 fx/fy）")
    ap.add_argument("--no-capture", action="store_true")
    ARGS = ap.parse_args()

    st = State()
    cap = open_camera(ARGS)
    if cap is None:
        log("ERROR: cannot open %s" % ARGS.device)
        return 2
    if os.path.isdir(OUT_DIR):
        old = sorted(x for x in os.listdir(OUT_DIR) if x.startswith("calib_"))
        if old:
            log("note: %d old capture(s) in %s will be reused" % (len(old), OUT_DIR))
            for x in old:
                p = os.path.join(OUT_DIR, x)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                ok, c = detect(img)
                if ok:
                    st.samples.append((p, c.astype(np.float32)))
                    st.cells |= cells_touched(c, img.shape[1], img.shape[0])
                    r, quad = radial_stats(c, img.shape[1], img.shape[0])
                    if r >= EDGE_R:
                        st.edge_quads.add(quad)
            log("recovered %d usable sample(s)" % len(st.samples))

    threading.Thread(target=serve, args=(st, ARGS.port), daemon=True).start()
    t_grab = threading.Thread(target=grab_loop, args=(cap, st, ARGS), daemon=True)
    t_det = threading.Thread(target=detect_loop, args=(st, ARGS), daemon=True)
    t_grab.start()
    t_det.start()
    log("capturing... open http://<board-ip>:%d in a browser" % ARGS.port)
    try:
        while t_det.is_alive():
            t_det.join(0.5)
    except KeyboardInterrupt:
        st.quit = True
    st.quit = True
    t_grab.join(3.0)
    log("stream fps %.1f, detect fps %.1f, %d sample(s)"
        % (st.fps, st.det_fps, len(st.samples)))
    if len(st.samples) >= 6:
        do_calib(st, ARGS)
    else:
        log("too few samples (%d), nothing calibrated" % len(st.samples))
    time.sleep(0.3)
    log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())