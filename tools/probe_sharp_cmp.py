# -*- coding: utf-8 -*-
"""相机清晰度对照实验：同场景下比较 分辨率/格式/sharpness 三组变量。
停止 det 后独占 /dev/video0 抓帧，算拉普拉斯方差与归一化梯度，并落盘样张。"""
import cv2, numpy as np, subprocess, time, os, json

OUT = "/root/vision_ws/tools/probe_out"
os.makedirs(OUT, exist_ok=True)

def ctl(**kw):
    for k, v in kw.items():
        subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", "%s=%s" % (k, v)],
                       capture_output=True)

def grab(w, h, fourcc, n=8):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames, t0 = [], time.time()
    while len(frames) < n and time.time() - t0 < 10:
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    cap.release()
    return frames

def metrics(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lv = float(cv2.Laplacian(g, cv2.CV_32F).var())
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    rms = float(g.std()) or 1.0
    gn = float(np.percentile(mag, 99.0) / rms)   # 归一化高频：对对比度不敏感
    return lv, gn, float(g.mean())

def run(tag, w, h, fourcc, sharp=None, expo=None, scale_to=None):
    if sharp is not None:
        ctl(sharpness=sharp)
    if expo is not None:
        ctl(auto_exposure=1, exposure_time_absolute=expo)
    time.sleep(0.4)
    fr = grab(w, h, fourcc)
    if not fr:
        print(json.dumps({"tag": tag, "err": "打不开或没有帧"}, ensure_ascii=False))
        return
    img = fr[len(fr) // 2]
    lv, gn, mean = metrics(img)
    cv2.imwrite(os.path.join(OUT, tag.replace(" ", "_").replace("/", "-") + ".png"), img)
    rec = {"tag": tag, "size": "%dx%d" % (img.shape[1], img.shape[0]), "frames": len(fr),
           "lap": round(lv, 1), "grad99": round(gn, 2), "mean": round(mean, 1)}
    if scale_to:
        small = cv2.resize(img, scale_to, interpolation=cv2.INTER_AREA)
        lv2, gn2, mean2 = metrics(small)
        cv2.imwrite(os.path.join(OUT, tag.replace(" ", "_").replace("/", "-") + "_to640.png"), small)
        rec.update({"to640_lap": round(lv2, 1), "to640_grad99": round(gn2, 2)})
    print(json.dumps(rec, ensure_ascii=False), flush=True)

subprocess.run(["bash", "/root/vision_ws/stop_all.sh"], capture_output=True)
time.sleep(2.5)
ctl(auto_exposure=3, sharpness=3)
run("A 640 MJPG sharp3", 640, 480, "MJPG", sharp=3)
run("B 640 MJPG sharp0", 640, 480, "MJPG", sharp=0)
run("C 640 MJPG sharp6", 640, 480, "MJPG", sharp=6)
run("D 640 YUYV sharp6", 640, 480, "YUYV", sharp=6)
run("E 1280 MJPG sharp6", 1280, 720, "MJPG", sharp=6, scale_to=(640, 480))
run("F 640 MJPG 手动曝光600", 640, 480, "MJPG", sharp=6, expo=600)
ctl(auto_exposure=3, sharpness=3)
print("DONE")
