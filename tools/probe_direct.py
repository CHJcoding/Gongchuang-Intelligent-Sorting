# -*- coding: utf-8 -*-
import subprocess, time, cv2, numpy as np


def sharp(img, tag):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    roi = g[int(h * 0.25):int(h * 0.85), int(w * 0.25):int(w * 0.85)]
    v = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    print("%-34s %dx%d 清晰度=%8.1f V中位=%3d" % (tag, img.shape[1], img.shape[0], v,
                                              int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]))))
    return v


print("停掉整条链路，独占相机…")
subprocess.run(["bash", "/root/vision_ws/stop_all.sh"], capture_output=True)
subprocess.run(["pkill", "-f", "preview_server.py"], capture_output=True)
subprocess.run(["pkill", "-f", "python3 sort_vision_node.py"], capture_output=True)
time.sleep(2.5)

cap = cv2.VideoCapture("/dev/cam0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frames = []
for _ in range(30):
    ok, fr = cap.read()
    if ok and fr is not None:
        frames.append(fr)
cap.release()
print("直接抓帧数:", len(frames))
if not frames:
    raise SystemExit("独占抓帧失败")
last = frames[-1]
cv2.imwrite("/root/vision_ws/hsv_tune/direct_now.png", last)
v_direct = sharp(last, "直接抓(无损PNG, 无二次JPEG)")
for q in (80, 92):
    ok, enc = cv2.imencode(".jpg", last, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    sharp(dec, "同一帧再压成 JPEG q%d" % q)
ref = cv2.imread("/root/vision_ws/hsv_tune/f00.jpg")
if ref is not None:
    print("参考：f00(早上, 当时也是 MJPG 直接抓) 清晰度=%.1f" % sharp(ref, "f00 参考"))
print("结论用数：直接抓=%.1f  q80=%.1f" % (v_direct, 0.0))
