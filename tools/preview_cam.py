# -*- coding: utf-8 -*-
"""纯相机预览：逐个打开指定相机，只显示画面不做检测，用来确认哪个编号是物料相机"""
import sys
import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"打开相机 {idx} ...")

# 不指定分辨率，用相机默认值（640x480）
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"相机 {idx} 打不开")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"分辨率 {w}x{h}，按 q 退出")

while True:
    ok, frame = cap.read()
    if not ok:
        print("读不到帧（黑屏/断开）")
        break
    cv2.putText(frame, f"Camera {idx}  {w}x{h}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow(f"Preview Camera {idx} (q=exit)", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
