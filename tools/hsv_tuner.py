import cv2
import numpy as np

def empty(a):
    pass

# 这里的 0 表示打开默认摄像头，如果是视频就填 '你的视频.mp4'
cap = cv2.VideoCapture(0) 

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# 注意：OpenCV 中 H 的范围是 0-179，S 和 V 是 0-255
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    success, img = cap.read()
    if not success:
        break

    # 将 BGR 转换为 HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 获取滑动条的当前值
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # 定义上下限
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # 根据阈值生成掩码 (Mask)，符合范围的地方变成白色，不符合变黑
    mask = cv2.inRange(imgHSV, lower, upper)

    # 用掩码提取原图中的颜色（可选，方便看效果）
    result = cv2.bitwise_and(img, img, mask=mask)

    # 将原图、掩码图、提取结果水平拼接显示
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转成3通道方便拼接
    stacked = np.hstack([img, mask_3ch, result])
    stacked = cv2.resize(stacked, (0, 0), fx=0.5, fy=0.5) # 缩小一点方便显示

    cv2.imshow("HSV Tuner", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()