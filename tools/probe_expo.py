# -*- coding: utf-8 -*-
import sys, time, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")
import config as C
from yolov8_bpu import Yolov8Bpu

def V(img):
    return int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]))

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

print("== 连抓 3 帧看曝光是否稳定 ==")
shots = []
for i in range(3):
    im = grab()
    shots.append(im)
    print("  第%d帧 V中位=%d 均值=%.1f" % (i + 1, V(im), float(im.mean())))
    time.sleep(1.0)
cv2.imwrite("/root/vision_ws/hsv_tune/g3.jpg", shots[-1])

def bright(img, k):
    out = img.astype(np.float32) * k
    return np.clip(out, 0, 255).astype(np.uint8)

det = Yolov8Bpu(getattr(C, "YOLO_BIN_MODEL", "/root/vision_ws/gc_best_1029.bin"),
                C.SHAPE_NAMES, conf_thres=0.25,
                iou_thres=getattr(C, "YOLO_IOU_THRES", 0.45), verbose=False)

old = cv2.imread("/root/vision_ws/hsv_tune/f00.jpg")
new = shots[-1]
k = V(old) / max(1.0, V(new))
cases = [("f00 原始 (V=%d)" % V(old), old),
         ("g3 原始 (V=%d)" % V(new), new),
         ("g3 x%.2f 提亮 (V=%d)" % (k, V(bright(new, k))), bright(new, k))]
for tag, img in cases:
    ds = det.detect(img)
    strong = [d for d in ds if float(d.score) >= 0.6]
    print("%-26s 检出=%d  conf>=0.6 的=%d  最强3: %s"
          % (tag, len(ds), len(strong),
             ", ".join("%s %.2f" % (d.name, d.score) for d in sorted(ds, key=lambda d: -float(d.score))[:3])))
