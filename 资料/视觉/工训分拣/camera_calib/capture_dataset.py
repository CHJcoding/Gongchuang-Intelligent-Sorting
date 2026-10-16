# -*- coding: utf-8 -*-
"""自动化数据集采集脚本（工创赛智能分拣视觉识别）

在最终俯视机位 + 灰托盘 + 定光源环境下，采集 7 类几何体的 YOLO 训练数据：
多姿态（边转动/移动物体边连拍），每类目标 200~300 张。

用法示例：
    python capture_dataset.py
    python capture_dataset.py --device 1 --width 1280 --height 720 --interval 0.5
    python capture_dataset.py --out D:\\data\\dataset

按键控制：
    1-7      切换当前拍摄类别（顺序见 CLASSES）
    s        单张拍摄
    空格     切换连拍（开启后每 interval 秒自动抓帧存盘，再按空格停止）
    q / ESC  退出

仅依赖 cv2 + numpy；相机打开复用 core/camera.py 的 _open_capture
（DSHOW 优先 -> MSMF 回退），保证 USB 相机可秒开。
"""
import argparse
import os
import re
import sys
import time

import cv2
import numpy as np  # noqa: F401  保持与项目依赖一致，后续如需帧处理可直接用

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.camera import _open_capture  # noqa: E402

# 类别定义：(目录名 / OSD 英文名, 中文注释)
CLASSES = [
    ("hollow_cube", "镂空正方体"),
    ("tetrahedron", "正四面体"),
    ("pentagonal_prism", "五棱柱"),
    ("triangular_prism", "三棱柱"),
    ("cone", "圆锥"),
    ("cylinder", "圆柱"),
    ("hemisphere", "半球"),
]

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]  # JPEG 质量 95

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _reconfigure_stdout():
    """终端输出中文：优先切 UTF-8；失败则 errors=replace 防止 UnicodeEncodeError"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def next_seq(out_dir: str, cls_name: str) -> int:
    """返回该类别目录下已有图片最大序号 + 1（从现有文件续起，不覆盖已有图）"""
    max_seq = 0
    subdir = os.path.join(out_dir, cls_name)
    if os.path.isdir(subdir):
        pattern = re.compile(r"^" + re.escape(cls_name) + r"_(\d+)\.jpg$")
        for fn in os.listdir(subdir):
            m = pattern.match(fn)
            if m:
                max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1


def save_frame(out_dir: str, cls_name: str, seq: int, frame) -> str:
    """保存一帧到 <out>/<cls>/<cls>_<seq:04d>.jpg，返回完整路径；失败返回空串

    用 cv2.imencode + ndarray.tofile 写盘：项目根目录含中文（工训分拣），
    直接 cv2.imwrite 在 Windows 下会因非 ASCII 路径失败，imencode 则无此问题。
    编码参数与 imwrite 完全一致（JPEG 质量 95）。
    """
    subdir = os.path.join(out_dir, cls_name)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, "{}_{:04d}.jpg".format(cls_name, seq))
    ok, buf = cv2.imencode(".jpg", frame, JPEG_PARAMS)
    if not ok:
        return ""
    try:
        buf.tofile(path)
    except Exception:
        return ""
    return path


def draw_osd(disp, cls_name: str, count: int, recording: bool,
             hint: str, saved_text: str):
    """在画面叠加英文 OSD：左上角类别/数量/录制状态，底部操作提示，不遮挡主体"""
    h = disp.shape[0]
    # 左上角：类别 + 已拍数量 + 连拍状态
    cv2.putText(disp, "CLASS: " + cls_name, (10, 24), FONT, 0.65,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(disp, "Count: {}".format(count), (10, 50), FONT, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    status = "REC" if recording else "STOP"
    color = (0, 0, 255) if recording else (255, 255, 255)
    cv2.putText(disp, status, (10, 76), FONT, 0.6, color, 2, cv2.LINE_AA)
    # 底部：操作提示 + 最近保存反馈
    cv2.putText(disp, hint, (10, h - 14), FONT, 0.5, (200, 200, 200), 1,
                cv2.LINE_AA)
    if saved_text:
        cv2.putText(disp, saved_text, (10, h - 38), FONT, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)


def main():
    _reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="工创赛智能分拣 YOLO 数据集采集（8 类几何体，多姿态连拍）")
    parser.add_argument("--device", type=int, default=0,
                        help="相机编号，默认 0")
    parser.add_argument("--width", type=int, default=640,
                        help="采集宽度，默认 640")
    parser.add_argument("--height", type=int, default=480,
                        help="采集高度，默认 480")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="连拍间隔秒，默认 0.5")
    parser.add_argument("--out", type=str,
                        default=os.path.join(PROJECT_ROOT, "dataset"),
                        help="输出根目录，默认 <项目根>/dataset")
    args = parser.parse_args()

    interval = max(0.05, args.interval)  # 防抖：最小 50ms

    cap = _open_capture(args.device, args.width, args.height)
    if cap is None:
        print("[错误] 无法打开相机 device={}（DSHOW 与默认后端均失败），"
              "请检查相机编号 / 连接 / 是否被占用。".format(args.device))
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print("输出根目录: {}".format(args.out))

    # 各类别起始序号 = 目录现有文件最大序号 + 1（续拍不覆盖）
    seqs = {name: next_seq(args.out, name) for name, _ in CLASSES}
    counts = {name: seq - 1 for name, seq in seqs.items()}
    for name, cn in CLASSES:
        print("  类别 {}（{}）：已有 {} 张，续拍从 {} 开始".format(
            name, cn, counts[name], seqs[name]))

    cur_idx = 0
    recording = False
    last_shot = 0.0
    last_saved_t = 0.0
    last_saved_txt = ""
    hint = "1-7:switch  S:shot  SPACE:burst  Q:quit"

    print("按键说明：1-7 切换类别 | S 单张 | 空格 连拍开/关 | Q 退出")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # 单帧失败跳过，不中断采集
                continue

            now = time.time()
            # 连拍：间隔到点自动抓帧存盘
            if recording and now - last_shot >= interval:
                cls_name = CLASSES[cur_idx][0]
                seq = seqs[cls_name]
                path = save_frame(args.out, cls_name, seq, frame)
                if path:
                    seqs[cls_name] = seq + 1
                    counts[cls_name] = seq
                    last_saved_txt = "Saved: {}".format(
                        os.path.basename(path))
                    print("[连拍] {}（{}）第 {} 张 -> {}".format(
                        cls_name, CLASSES[cur_idx][1], seq, path))
                else:
                    last_saved_txt = "Save FAILED!"
                    print("[错误] 保存失败：{}".format(cls_name))
                last_saved_t = now
                last_shot = now

            # 最近保存提示显示 0.6 秒
            saved_text = last_saved_txt if (now - last_saved_t) < 0.6 else ""

            cls_name = CLASSES[cur_idx][0]
            disp = frame.copy()
            draw_osd(disp, cls_name, counts[cls_name], recording, hint,
                     saved_text)
            cv2.imshow("Capture Dataset - {}".format(cls_name), disp)

            key = cv2.waitKey(1) & 0xFF
            if key == -1:
                continue
            if ord("1") <= key < ord("1") + len(CLASSES):
                cur_idx = key - ord("1")
                print(">>> 切换类别：{}（{}）当前已拍 {}".format(
                    CLASSES[cur_idx][0], CLASSES[cur_idx][1],
                    counts[CLASSES[cur_idx][0]]))
            elif key in (ord("s"), ord("S")):
                seq = seqs[cls_name]
                path = save_frame(args.out, cls_name, seq, frame)
                if path:
                    seqs[cls_name] = seq + 1
                    counts[cls_name] = seq
                    last_saved_txt = "Saved: {}".format(os.path.basename(path))
                    print("[单张] {}（{}）第 {} 张 -> {}".format(
                        cls_name, CLASSES[cur_idx][1], seq, path))
                else:
                    last_saved_txt = "Save FAILED!"
                    print("[错误] 保存失败：{}".format(cls_name))
                last_saved_t = time.time()
            elif key == 32:  # 空格
                recording = not recording
                last_shot = time.time()
                print(">>> 连拍 {}".format("开启" if recording else "停止"))
            elif key in (ord("q"), ord("Q"), 27):
                print(">>> 退出采集")
                break
            # 其他按键忽略
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n本次采集汇总：")
    total = 0
    for name, cn in CLASSES:
        total += counts[name]
        print("  {:18s}（{}）：{} 张".format(name, cn, counts[name]))
    print("  合计：{} 张（输出目录：{}）".format(total, args.out))


if __name__ == "__main__":
    main()
