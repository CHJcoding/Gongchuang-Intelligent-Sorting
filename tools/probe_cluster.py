# -*- coding: utf-8 -*-
import sys, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
from color_classifier import classify_color
from yolov8_bpu import Yolov8Bpu
import camera_model, comm
from hand_eye import pixel_to_robot

def grab(n=14):
    cap = cv2.VideoCapture("/dev/cam0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img = None
    for _ in range(n):
        ok, fr = cap.read()
        if ok and fr is not None:
            img = fr
    cap.release()
    return img

def xy_mm(cx, cy):
    try:
        u, v = camera_model.undistort_point(cx, cy)
        x, y, _ = pixel_to_robot(u, v)
        return (round(float(x), 1), round(float(y), 1))
    except Exception:
        return None

def in_ws(p):
    if p is None:
        return None
    return (C.WORKSPACE_X_MIN <= p[0] <= C.WORKSPACE_X_MAX) and (C.WORKSPACE_Y_MIN <= p[1] <= C.WORKSPACE_Y_MAX)

det = Yolov8Bpu(getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin"),
                C.SHAPE_NAMES, conf_thres=0.25,
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=False)

frames = []
for i in range(5):
    im = grab()
    frames.append(im)
print("5 帧采集完成")

det0 = []
for f in frames:
    out = []
    for d in det.detect(f):
        x1, y1, x2, y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
        roi = f[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        cn, cc = (None, 0.0)
        if roi.size:
            cn, cc = classify_color(roi)
        out.append(dict(name=d.name, score=float(d.score), cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0,
                        box=(x1, y1, x2, y2), color=cn, conf=float(cc)))
    det0.append(out)

clusters = []
for fi, out in enumerate(det0):
    for d in out:
        for c in clusters:
            if abs(c["cx"] - d["cx"]) <= 30 and abs(c["cy"] - d["cy"]) <= 30:
                break
        else:
            c = dict(cx=d["cx"], cy=d["cy"], hits={})
            clusters.append(c)
        c["hits"].setdefault(fi, []).append(d)
        c["cx"] = 0.5 * c["cx"] + 0.5 * d["cx"]
        c["cy"] = 0.5 * c["cy"] + 0.5 * d["cy"]

print("== 5 帧聚类：%d 个稳定位置 ==" % len(clusters))
last = frames[-1]
canvas = last.copy()
rows = []
for i, c in enumerate(sorted(clusters, key=lambda c: (c["cy"], c["cx"])), 1):
    seen = sorted(c["hits"].keys())
    names = [d["name"] for fi in seen for d in c["hits"][fi]]
    scores = [d["score"] for fi in seen for d in c["hits"][fi]]
    colors = [d["color"] for fi in seen for d in c["hits"][fi] if d["color"]]
    best = max((d for fi in seen for d in c["hits"][fi]), key=lambda d: d["score"])
    p = xy_mm(c["cx"], c["cy"])
    top = max(set(names), key=names.count)
    same = names.count(top)
    rows.append((i, top, same, len(names), round(min(scores), 2), round(max(scores), 2),
                 (max(set(colors), key=colors.count) if colors else None), (round(c["cx"]), round(c["cy"])), p, in_ws(p)))
    col = (0, 210, 0) if (same == len(names) and min(scores) >= 0.6) else (0, 165, 255)
    cv2.rectangle(canvas, (int(best["box"][0]), int(best["box"][1])), (int(best["box"][2]), int(best["box"][3])), col, 2)
    cv2.putText(canvas, "#%d %s %.2f" % (i, top, best["score"]), (int(best["box"][0]) + 2, max(12, int(best["box"][1]) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    print("#%d 位置=%s  出现 %d/5 帧  多数类=%-15s 一致=%d/%d  分数 %.2f~%.2f  颜色=%s  mm=%s  限位=%s"
          % (i, (round(c["cx"]), round(c["cy"])), len(seen), top, same, len(names),
             min(scores), max(scores), rows[-1][6], p, in_ws(p)))
    if len(set(names)) > 1:
        print("      帧间不同标签: %s" % ", ".join("%s@%d帧" % (n, names.count(n)) for n in set(names)))
cv2.imwrite("/root/vision_ws/hsv_tune/g4_cluster.jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
cv2.imwrite("/root/vision_ws/hsv_tune/g4_last.jpg", last)
print("标注图: /root/vision_ws/hsv_tune/g4_cluster.jpg")
