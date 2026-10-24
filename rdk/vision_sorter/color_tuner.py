# -*- coding: utf-8 -*-
"""
现场调HSV阈值用：跑起来后拖trackbar，框ROI看mask，调好后把数值填回config.py。
用法: python color_tuner.py
按 q 退出，按 c 截图保存当前帧到 captures/
"""
import cv2
import numpy as np
import os
import config as C


def nothing(x):
    pass


def main():
    os.makedirs("captures", exist_ok=True)
    cap = cv2.VideoCapture(C.CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    cv2.namedWindow("bars")
    cv2.createTrackbar("H_l", "bars", 0, 179, nothing)
    cv2.createTrackbar("H_h", "bars", 179, 179, nothing)
    cv2.createTrackbar("S_l", "bars", 50, 255, nothing)
    cv2.createTrackbar("S_h", "bars", 255, 255, nothing)
    cv2.createTrackbar("V_l", "bars", 50, 255, nothing)
    cv2.createTrackbar("V_h", "bars", 255, 255, nothing)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hl = cv2.getTrackbarPos("H_l", "bars")
        hh = cv2.getTrackbarPos("H_h", "bars")
        sl = cv2.getTrackbarPos("S_l", "bars")
        sh = cv2.getTrackbarPos("S_h", "bars")
        vl = cv2.getTrackbarPos("V_l", "bars")
        vh = cv2.getTrackbarPos("V_h", "bars")

        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])
        mask = cv2.inRange(hsv, lower, upper)
        # 红色跨180度时手动再加一段
        if hl > 160:
            m2 = cv2.inRange(hsv, np.array([0, sl, vl]), np.array([10, sh, vh]))
            mask = cv2.bitwise_or(mask, m2)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.putText(res, f"H:[{hl},{hh}] S:[{sl},{sh}] V:[{vl},{vh}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('c'):
            p = f"captures/frame_{idx}.jpg"
            cv2.imwrite(p, frame)
            print(f"saved {p}")
            idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
