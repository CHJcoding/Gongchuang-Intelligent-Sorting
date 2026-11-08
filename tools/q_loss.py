import glob, time, numpy as np, cv2
files = sorted(glob.glob("/root/vision_ws/hsv_tune/f*.jpg"))[:8]
if not files:
    files = sorted(glob.glob("/root/vision_ws/*.jpg"))[:5]
print("files:", len(files))
def lap(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def roundtrip(buf, q):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, out, len(enc.tobytes())

rows = []
for f in files:
    buf = open(f, "rb").read()
    img, o92, s92 = roundtrip(buf, 92)
    _, o80, s80 = roundtrip(buf, 80)
    _, o95, s95 = roundtrip(buf, 95)
    rows.append((len(buf), s92, s95, s80, lap(img), lap(o92), lap(o95), lap(o80)))
n = len(rows)
med = lambda i: sorted(r[i] for r in rows)[n // 2]
print("orig bytes=%d  q92=%d (-%.1f%%)  q95=%d  q80=%d" % (
    med(0), med(1), 100.0 * (1 - med(1) / med(0)), med(2), med(3)))
print("lap orig=%.1f  q92=%.1f (%.1f%%)  q95=%.1f (%.1f%%)  q80=%.1f (%.1f%%)" % (
    med(4), med(5), 100.0 * (med(5) / med(4) - 1), med(6), 100.0 * (med(6) / med(4) - 1),
    med(7), 100.0 * (med(7) / med(4) - 1)))

# 单帧耗时（这个 CPU 上）
buf = open(files[0], "rb").read()
arr = np.frombuffer(buf, np.uint8)
N = 30
t = time.time()
for _ in range(N):
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
t_dec = (time.time() - t) / N * 1000
t = time.time()
for _ in range(N):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
t_enc = (time.time() - t) / N * 1000
h, w = img.shape[:2]
import sys
sys.path.insert(0, "/root/vision_ws/vision_sorter")
try:
    from yolov8_bpu import letterbox_image, bgr_to_nv12
    t = time.time()
    for _ in range(N):
        canvas, r, dw, dh = letterbox_image(img, 640, 640)
    t_lb = (time.time() - t) / N * 1000
    t = time.time()
    for _ in range(N):
        nv12 = bgr_to_nv12(canvas)
    t_nv = (time.time() - t) / N * 1000
    print("letterbox %.2f ms, bgr_to_nv12 %.2f ms, canvas=%s ratio=%.3f pad=(%.0f,%.0f)" % (t_lb, t_nv, canvas.shape, r, dw, dh))
except Exception as e:
    print("letterbox skip:", e)
print("imdecode %.2f ms, imencode(q92) %.2f ms, 源图 %dx%d" % (t_dec, t_enc, w, h))
print("30fps 预算 = 33.3 ms")
