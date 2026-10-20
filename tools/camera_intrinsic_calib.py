# -*- coding: utf-8 -*-
"""相机内参标定（棋盘格）

用法：
    python camera_intrinsic_calib.py                  # 默认相机 0
    python camera_intrinsic_calib.py --source 1       # 相机 1
    python camera_intrinsic_calib.py --cols 9 --rows 6 --square 25

按键：
    空格    拍一张（检测到棋盘格角点才保存）
    c       开始解算内参
    d       切换显示去畸变效果
    q/ESC   退出
q
结果保存到 camera_intrinsic.json：
    {mtx: [[...]], dist: [...], reproj_error: 0.1, size: [w,h]}
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=9, help="棋盘格内角点列数（宽方向角点数）")
    parser.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数（高方向角点数）")
    parser.add_argument("--square", type=float, default=25.0, help="格子边长 mm")
    parser.add_argument("--min-samples", type=int, default=15, help="最少采多少张才解算")
    parser.add_argument("--out", default="camera_intrinsic.json")
    args = parser.parse_args()

    # 棋盘格 3D 点（Z=0 平面）
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square  # mm

    objpoints = []  # 3D 真实点
    imgpoints = []  # 2D 像素点

    # 打开相机
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_sub = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    mtx, dist = None, None
    show_undistort = False

    print(f"棋盘格 {args.cols}x{args.rows} 内角点，格子 {args.square}mm")
    print("空格拍照，c 解算，d 切换去畸变，q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 实时检测角点
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        disp = frame.copy()

        if found:
            cv2.drawChessboardCorners(disp, (args.cols, args.rows), corners, found)
            cv2.putText(disp, "Chessboard found! Press SPACE to save",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "Show chessboard to camera...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(disp, f"Samples: {len(imgpoints)}/{args.min_samples}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if mtx is not None and show_undistort:
            # 实时显示去畸变
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            undist = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            disp = undist
            cv2.putText(disp, "UNDISTORTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Camera Calibration (SPACE=save, C=calc, D=undistort, Q=quit)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" ") and found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria_sub)
            objpoints.append(objp.copy())
            imgpoints.append(corners_sub.copy())
            print(f"saved {len(imgpoints)}")

        elif key == ord("c"):
            if len(imgpoints) < args.min_samples:
                print(f"need at least {args.min_samples}, have {len(imgpoints)}")
                continue
            print("calibrating...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, (w, h), None, None)
            # 重投影误差
            total_err = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_err += err
            mean_err = total_err / len(objpoints)
            print(f"reproj error: {mean_err:.4f} px")
            print(f"mtx:\n{mtx}")
            print(f"dist: {dist.ravel()}")
            # 保存
            data = {
                "mtx": mtx.tolist(),
                "dist": dist.ravel().tolist(),
                "reproj_error": float(mean_err),
                "size": [int(w), int(h)],
                "chessboard": [args.cols, args.rows, args.square],
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"saved to {args.out}")

        elif key == ord("d"):
            if mtx is not None:
                show_undistort = not show_undistort
            else:
                print("press C first")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
