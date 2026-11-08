# -*- coding: utf-8 -*-
"""离线验证旋转贴合框：拿实拍帧跑一遍识别+精修，画出旋转四边形并存图。

判据（不靠肉眼）：
  1) rect_points 四个点构成矩形：两两相邻边长交替、对角线相等；
  2) 四边形面积 ≈ rect_size[0]*rect_size[1]（相对误差 <2%）；
  3) 画到图上后，该颜色像素数 > 0（说明真画上去了）。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C  # noqa: E402
import grasp_refine  # noqa: E402
import yolov8_bpu as Y  # noqa: E402
from color_classifier import classify_color  # noqa: E402

FRAME = sys.argv[1] if len(sys.argv) > 1 else "/root/vision_ws/frame_now.jpg"
OUT = "/root/vision_ws/rotbox_check.jpg"
BIN = "/root/vision_ws/gc_best_1029.bin"
LABELS = [l.strip() for l in open("/root/vision_ws/gc_best_1029_labels.list") if l.strip()]


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def main():
    frame = cv2.imread(FRAME)
    if frame is None:
        sys.exit("读不到图: %s" % FRAME)
    print("帧 %s %dx%d" % (FRAME, frame.shape[1], frame.shape[0]))
    det = Y.Yolov8Bpu(BIN, LABELS, conf_thres=C.YOLO_CONF_THRES, verbose=False)
    res = det.detect(frame)
    print("检出 %d 个" % len(res))

    canvas = frame.copy()
    for d in res:
        bbox = [int(round(d.x1)), int(round(d.y1)), int(round(d.x2)), int(round(d.y2))]
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):min(frame.shape[0], y2),
                    max(0, x1):min(frame.shape[1], x2)]
        color, cconf = classify_color(roi) if roi.size else (None, 0.0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 255), 1)
        if color is None:
            print("  %-5s %.2f 颜色判不出，跳过" % (d.name, d.score))
            continue
        gp = grasp_refine.refine(frame, bbox, color)
        if not gp["ok"]:
            print("  %-5s 颜色=%-6s 精修失败（%s）" % (d.name, color, gp["reason"][:40]))
            continue
        rp = gp["rect_points"]
        w, h = gp["rect_size"]
        a_rect = w * h
        a_poly = poly_area(rp)
        # 相邻边长
        edges = [float(np.linalg.norm(rp[i] - rp[(i + 1) % 4])) for i in range(4)]
        err = abs(a_poly - a_rect) / max(1e-6, a_rect)
        cv2.polylines(canvas, [rp.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 210, 0), 2, cv2.LINE_AA)
        cx, cy = gp["point"]
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1)
        ok = "✓" if err < 0.02 else "✗"
        print("  %-5s %.2f 颜色=%-6s 尺寸=%.1fx%.1f 角度=%.0f° 面积(多边形)=%.0f 面积(尺寸)=%.0f 误差=%.1f%% %s"
              % (d.name, d.score, color, w, h, gp["angle"], a_poly, a_rect, err * 100, ok))
        print("        四角=%s  边长=%s" % (np.round(rp, 1).tolist(),
                                          np.round(edges, 1).tolist()))

    # 画上去的绿色像素数（纯绿 0,210,0 在 JPEG 压缩后会有偏差，用范围判）
    g = (canvas[:, :, 1].astype(np.int16) - canvas[:, :, 2].astype(np.int16) > 60) & \
        (canvas[:, :, 1].astype(np.int16) - canvas[:, :, 0].astype(np.int16) > 60)
    cv2.imwrite(OUT, canvas)
    print("绿色(贴合框)像素数=%d  已存 %s" % (int(g.sum()), OUT))


if __name__ == "__main__":
    main()