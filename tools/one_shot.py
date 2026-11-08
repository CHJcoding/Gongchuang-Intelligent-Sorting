import sys, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
from yolov8_bpu import Yolov8Bpu
from color_classifier import classify_color
import grasp_refine, comm

img = cv2.imread("/root/vision_ws/one_shot_in.jpg")
print("image:", img.shape)
binp = getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin")
det = Yolov8Bpu(binp, C.SHAPE_NAMES,
                conf_thres=getattr(C, "YOLO_CONF_THRES", 0.6),
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=True)
dets = det.detect(img)
print("detections:", len(dets))
canvas = img.copy()
rows = []
for d in dets:
    x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
    roi = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    cname, cconf = (None, 0.0)
    if roi.size:
        cname, cconf = classify_color(roi)
    gp, fitted = None, False
    if cname is not None:
        try:
            gp = grasp_refine.refine(img, [x1, y1, x2, y2], cname)
            fitted = bool(gp and gp.get("ok"))
        except Exception as e:
            print("refine err:", e)
    mid = comm.material_id(d.name, cname)[0] if cname else None
    rows.append((d.name, round(float(d.score), 3), cname, round(float(cconf), 3),
                 fitted, str((gp or {}).get("reason") or ""), mid,
                 comm.SHAPE_ID.get(d.name, 255),
                 comm.COLOR_ID.get(cname, 255) if cname else 255))
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 255), 1)
    if fitted:
        bb = [int(round(v)) for v in gp["tight_bbox"]]
        rp = gp.get("rect_points")
        if rp is not None:
            cv2.polylines(canvas, [np.asarray(rp, np.int32).reshape(-1, 1, 2)], True, (0, 210, 0), 2)
        else:
            cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (0, 210, 0), 2)
    else:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
    cv2.putText(canvas, "%s %.2f %s" % (d.name, d.score, cname or "?"),
                (int(x1) + 2, max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite("/root/vision_ws/one_shot_out.jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print("---- rows: name score color conf fitted reason material_id shape_id color_id ----")
for r in rows:
    print(r)
