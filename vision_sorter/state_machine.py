# -*- coding: utf-8 -*-
"""
上位机主状态机：
WAIT_MATERIAL -> TRIGGER_CAPTURE -> DETECT -> COLOR -> TRANSFORM -> SEND_GRAB -> WAIT_DONE -> RESET -> ...
异常分支：识别失败重试 / 通信失败重试 / 置信度低报警
"""
import time
import cv2
from enum import Enum, auto
import config as C
from camera import USBCamera
from detector import ShapeDetector
from color_classifier import classify_color
from hand_eye import pixel_to_robot
from comm import Comm


class State(Enum):
    WAIT_MATERIAL = auto()
    CAPTURE = auto()
    DETECT = auto()
    JUDGE_COLOR = auto()
    TRANSFORM = auto()
    SEND_GRAB = auto()
    WAIT_DONE = auto()
    ERROR = auto()


class Sorter:
    def __init__(self, use_cam=True, use_comm=True):
        self.cam = USBCamera() if use_cam else None
        self.detector = ShapeDetector()
        self.comm = Comm() if use_comm else None
        self.state = State.WAIT_MATERIAL
        self.current = None      # 当前识别到的目标
        self.retry = 0
        self.max_retry = 3
        self.running = True

    def setup(self):
        if self.cam:
            self.cam.open()
        if self.comm:
            self.comm.open()

    def step(self):
        if self.state == State.WAIT_MATERIAL:
            # 比赛里通常由光电开关/下位机发信号触发；这里改成等按键或自动循环
            print("[WAIT] 等待物料到位...")
            time.sleep(0.5)
            self.state = State.CAPTURE

        elif self.state == State.CAPTURE:
            frame = self.cam.read() if self.cam else None
            if frame is None:
                self._err("相机读不到帧")
                return
            self.frame = frame
            self.state = State.DETECT

        elif self.state == State.DETECT:
            dets = self.detector.detect(self.frame)
            if not dets:
                self.retry += 1
                if self.retry >= self.max_retry:
                    self._err("未检测到物料")
                    self.retry = 0
                    return
                self.state = State.CAPTURE
                return
            # 取置信度最高的那个（竞赛场景通常一次只抓一个）
            self.current = max(dets, key=lambda d: d["conf"])
            print(f"[DETECT] shape={self.current['shape']} conf={self.current['conf']:.2f} "
                  f"bbox={self.current['bbox']}")
            self.state = State.JUDGE_COLOR

        elif self.state == State.JUDGE_COLOR:
            color, conf = classify_color(self.current["roi"])
            if color is None or conf < 0.6:
                self.retry += 1
                if self.retry >= self.max_retry:
                    self._err(f"颜色识别失败 conf={conf}")
                    self.retry = 0
                    return
                self.state = State.CAPTURE
                return
            self.current["color"] = color
            self.current["color_conf"] = conf
            print(f"[COLOR] {color} conf={conf}")
            self.state = State.TRANSFORM

        elif self.state == State.TRANSFORM:
            x, y, z = pixel_to_robot(self.current["cx"], self.current["cy"])
            self.current["robot_xy"] = (x, y, z)
            print(f"[TRANSFORM] pixel=({self.current['cx']:.0f},{self.current['cy']:.0f}) "
                  f"-> robot=({x},{y},{z})")
            self.state = State.SEND_GRAB

        elif self.state == State.SEND_GRAB:
            x, y, z = self.current["robot_xy"]
            if self.comm:
                ok = self.comm.send_grab(self.current["shape"], self.current["color"], x, y)
            else:
                ok = True
                print("[SIM] 不连下位机，模拟发送成功")
            if not ok:
                self._err("串口发送失败/无ACK")
                return
            self.state = State.WAIT_DONE

        elif self.state == State.WAIT_DONE:
            print(f"[WAIT] 等机械臂抓取完成 {C.GRAB_DONE_WAIT_S}s ...")
            time.sleep(C.GRAB_DONE_WAIT_S)
            self.retry = 0
            self.state = State.WAIT_MATERIAL

        elif self.state == State.ERROR:
            print(f"[ERROR] {self.err_msg}，3秒后重试")
            time.sleep(3.0)
            self.state = State.WAIT_MATERIAL

    def _err(self, msg):
        self.err_msg = msg
        print(f"[ERROR] {msg}")
        self.state = State.ERROR

    def run(self):
        self.setup()
        print("=== 上位机启动，按 q 退出 ===")
        while self.running:
            self.step()
            # 可视化
            if self.cam:
                f = self.cam.read()
                if f is not None and self.current:
                    x1, y1, x2, y2 = self.current["bbox"]
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"{self.current['shape']}/{self.current.get('color','?')}"
                    cv2.putText(f, txt, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("sorter", f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
        self.shutdown()

    def shutdown(self):
        if self.cam:
            self.cam.close()
        if self.comm:
            self.comm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 调试时 use_comm=False 先不连下位机
    s = Sorter(use_cam=True, use_comm=False)
    s.run()
