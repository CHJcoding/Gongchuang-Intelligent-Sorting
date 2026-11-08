# -*- coding: utf-8 -*-
import sys, cv2, numpy as np
sys.path.insert(0, "/root/vision_ws/vision_sorter")

def sharp(img, tag):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    roi = g[int(h * 0.25):int(h * 0.85), int(w * 0.25):int(w * 0.85)]
    lap = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    ten = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    print("%-28s 拉普拉斯方差=%8.1f  Tenengrad=%6.2f  亮度V中位=%3d"
          % (tag, lap, ten, int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]))))
    return lap, ten

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

for p in ["/root/vision_ws/hsv_tune/f00.jpg", "/root/vision_ws/hsv_tune/g2.jpg",
          "/root/vision_ws/hsv_tune/g4_last.jpg"]:
    im = cv2.imread(p)
    if im is not None:
        sharp(im, p.split("/")[-1] + " (历史帧)")

im = grab()
if im is not None:
    cv2.imwrite("/root/vision_ws/hsv_tune/now.jpg", im)
    sharp(im, "现在实时抓帧 now.jpg")
else:
    print("抓帧失败")
