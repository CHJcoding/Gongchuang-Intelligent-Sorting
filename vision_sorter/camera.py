# -*- coding: utf-8 -*-
"""多线程USB相机采集，C100免驱直接OpenCV读。"""
import threading
import time
import cv2
import config as C


class USBCamera:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.latest = None
        self.running = False
        self.thread = None

    def open(self):
        # RDK X5 Linux 用 V4L2，不要加 CAP_DSHOW
        self.cap = cv2.VideoCapture(C.CAM_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
        # 关键：手动补光，必须关掉相机自动AE/AWB，否则颜色会飘
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25=手动
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)             # 关自动白平衡
        if not self.cap.isOpened():
            raise RuntimeError("相机/dev/video%d打开失败，先 ls /dev/video*" % C.CAM_INDEX)
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()
        time.sleep(C.CAMERA_WARMUP_S)

    def _grab_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame

    def read(self):
        """返回最新一帧，没读到返回None。"""
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    cam = USBCamera()
    cam.open()
    while True:
        f = cam.read()
        if f is None:
            continue
        cv2.imshow("C100 preview", f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.close()
    cv2.destroyAllWindows()
