# -*- coding: utf-8 -*-
import sys, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
from color_classifier import classify_color
from yolov8_bpu import Yolov8Bpu
import camera_model, comm
from hand_eye import pixel_to_robot


def xy_mm(cx, cy):
    try:
        u, v = camera_model.undistort_point(cx, cy)
        x, y, _ = pixel_to_robot(u, v)
        return (round(float(x), 1), round(float(y), 1))
    except Exception as e:
        return None


def in_ws(p):
    if p is None:
        return None
    xlo = getattr(C, "WORKSPACE_X_MIN", None)
    if xlo is None:
        return True
    return (xlo <= p[0] <= C.WORKSPACE_X_MAX) and (C.WORKSPACE_Y_MIN <= p[1] <= C.WORKSPACE_Y_MAX)


def brightness(img):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    return int(np.median(v))


def analyse(det, img, tag, thr, save=None):
    dets = det.detect(img)
    rows = []
    canvas = img.copy()
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
        roi = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        cn, cc, hsvm = None, 0.0, None
        if roi.size:
            cn, cc = classify_color(roi)
            h = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            hsvm = tuple(int(v) for v in np.median(h, axis=0))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        p = xy_mm(cx, cy)
        rows.append(dict(n=i, name=d.name, score=round(float(d.score), 3), color=cn,
                         conf=round(float(cc), 3), hsv=hsvm, px=(int(cx), int(cy)),
                         mm=p, ws=in_ws(p), mid=(comm.material_id(d.name, cn)[0] if cn else None)))
        col = (0, 210, 0) if float(d.score) >= 0.6 else (0, 165, 255)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(canvas, "#%d %s %.2f" % (i, d.name, d.score),
                    (int(x1) + 2, max(12, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 2, cv2.LINE_AA)
    if save:
        cv2.imwrite(save, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("==== %s  conf>=%.2f  %d 个 ====" % (tag, thr, len(rows)))
    for r in rows:
        print("#%d %-16s %.3f color=%-5s conf=%.2f HSV=%s px=%s mm=%s 限位=%s id=%s"
              % (r["n"], r["name"], r["score"], str(r["color"]), r["conf"], r["hsv"],
                 r["px"], r["mm"], r["ws"], r["mid"]))
    return rows


low = Yolov8Bpu(getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin"),
                C.SHAPE_NAMES, conf_thres=0.25,
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=False)
old = cv2.imread("/root/vision_ws/hsv_tune/f00.jpg")
new = cv2.imread("/root/vision_ws/hsv_tune/g2.jpg")
print("f00 亮度中位 V=%d   g2 亮度中位 V=%d" % (brightness(old), brightness(new)))
analyse(low, old, "上一张 f00.jpg", 0.25)
analyse(low, new, "新抓 g2.jpg", 0.25, "/root/vision_ws/hsv_tune/g2_annot.jpg")
