# 视觉分拣上位机 (RDK X5 + C100 USB + YOLOv5 + HSV)

## 文件说明
| 文件 | 作用 |
|---|---|
| `config.py` | 所有参数集中地：相机/HSV阈值/手眼矩阵/串口/状态机，现场只改这个 |
| `camera.py` | C100 USB相机多线程采集，已关自动曝光/白平衡 |
| `color_tuner.py` | 现场调HSV阈值工具，拖trackbar看mask，调好后填回config.py |
| `color_classifier.py` | ROI内颜色投票，自动剔除高光/灰像素，深红vs暗橙二次判别 |
| `detector.py` | YOLOv5检测封装，PC用.pt，板端替换成地平线.bin |
| `hand_eye.py` | 像素→机械臂坐标转换，附带标定脚本 |
| `comm.py` | 串口二进制协议，带CRC8和ACK重试 |
| `state_machine.py` | 主状态机，串起拍照→识别→发指令→等待完成 |

## 上手顺序
1. **PC上先跑通视觉**：
   ```
   pip install opencv-python numpy torch pyserial
   python camera.py          # 先看相机画面
   python color_tuner.py    # 现场补光下调10种颜色的HSV区间，填回config.py
   python hand_eye.py        # 打印标定矩阵，填回config.py
   python state_machine.py   # use_comm=False 先不连下位机，看检测框对不对
   ```
2. **连下位机**：把 `state_machine.py` 最后一行 `use_comm=False` 改成 `True`，`config.py` 里串口口址改成实际的。
3. **上RDK X5**：
   - 模型转 `.bin`（地平线 hb_toolchain），替换 `detector.py` 里的 `ShapeDetector`
   - `camera.py` 里把 `cv2.CAP_DSHOW` 删掉（Linux不支持），`SERIAL_PORT` 改成 `/dev/ttyUSB0`

## 针对你提的三个难点，代码里已做的防护
- **反光**：`color_classifier.py` 自动剔除 V>230 高光像素和 S<40 灰像素，只对"可靠像素"投票；相机关了自动曝光。
- **重叠遮挡**：状态机里取置信度最高的单个目标抓，抓完复位再拍第二张，不硬算被挡目标。
- **深红/暗橙**：投票置信度<0.75时自动用 R/(R+G) 比值二次判别。

## 现场必做检查清单
- [ ] 相机固定后，手动补光角度调到物料表面无明显光斑
- [ ] 相机自动曝光/自动白平衡已关
- [ ] 每个颜色在比赛现场光下重新过一遍 `color_tuner.py`
- [ ] 手眼标定实测平均误差<5mm，否则重标
- [ ] 连续跑20次抓取，记录成功率，误判率>5%的颜色/形状单独加样本
