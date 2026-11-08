# -*- coding: utf-8 -*-
import sys, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
from color_classifier import classify_color
from yolov8_bpu import Yolov8Bpu
import comm
try:
    import camera_model as CM
except Exception as e:
    CM = None
    print("camera_model 导入失败:", e)


def grab(path, n=14):
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
    if img is None:
        raise SystemExit("抓帧失败")
    cv2.imwrite(path, img)
    return img


def xy_mm(img, cx, cy):
    if CM is None:
        return None
    try:
        u, v = CM.undistort_point(img, cx, cy) if "img" in CM.undistort_point.__code__.co_varnames else CM.undistort_point(cx, cy)
        x, y, z = CM.pixel_to_robot(u, v)
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


def analyse(det, img, tag, save=None):
    dets = det.detect(img)
    rows = []
    canvas = img.copy()
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
        roi = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        cn, cc = (None, 0.0)
        hsvm = None
        if roi.size:
            cn, cc = classify_color(roi)
            h = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            hsvm = tuple(int(v) for v in np.median(h, axis=0))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        p = xy_mm(img, cx, cy)
        rows.append(dict(n=i, name=d.name, score=round(float(d.score), 3), color=cn,
                         conf=round(float(cc), 3), hsv=hsvm, px=(int(cx), int(cy)),
                         mm=p, ws=in_ws(p),
                         mid=(comm.material_id(d.name, cn)[0] if cn else None)))
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 255), 2)
        cv2.putText(canvas, "#%d %s %.2f %s" % (i, d.name, d.score, cn or "?"),
                    (int(x1) + 2, max(12, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
    if save:
        cv2.imwrite(save, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("==== %s: %d 个检出 ====" % (tag, len(rows)))
    for r in rows:
        print("#%d %-16s %.3f  color=%-5s conf=%.2f  HSV=%s  px=%s  mm=%s  限位=%s  material_id=%s"
              % (r["n"], r["name"], r["score"], str(r["color"]), r["conf"], r["hsv"],
                 r["px"], r["mm"], r["ws"], r["mid"]))
    return rows


det = Yolov8Bpu(getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin"),
                C.SHAPE_NAMES,
                conf_thres=getattr(C, "YOLO_CONF_THRES", 0.6),
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=False)
old = cv2.imread("/root/vision_ws/hsv_tune/f00.jpg")
again = analyse(det, old, "上一张 f00.jpg（刚才那张）")

new = grab("/root/vision_ws/hsv_tune/g2.jpg")
now = analyse(det, new, "新抓 g2.jpg", "/root/vision_ws/hsv_tune/g2_annot.jpg")
print("新图已存 /root/vision_ws/hsv_tune/g2.jpg 与 g2_annot.jpg")
