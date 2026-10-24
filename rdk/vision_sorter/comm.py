# -*- coding: utf-8 -*-
"""
与下位机(STM32/ESP32)的串口通信协议。
帧格式:
  [0] 0xAA
  [1] 0x55
  [2] shape_id   形状编号 0-8
  [3] color_id   颜色编号 0-9
  [4:6] x_int16  机械臂目标X(mm)，小端
  [6:8] y_int16  机械臂目标Y(mm)，小端
  [8]   z_byte   抓取高度档(0=低,1=中,2=高)
  [9]   crc8     从frame_head到z的校验
  [10] 0x0D      帧尾
下位机收到后回 0x06 (ACK)。
"""
import serial
import struct
import time
import config as C


# 形状/颜色和编号的映射，要和下位机固件里的表一致
SHAPE_ID = {name: i for i, name in enumerate(C.SHAPE_NAMES)}
COLOR_ID = {name: i for i, name in enumerate(C.COLOR_RANGES_HSV.keys())}


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Comm:
    def __init__(self):
        self.ser = None

    def open(self):
        self.ser = serial.Serial(
            C.SERIAL_PORT, C.SERIAL_BAUD, timeout=C.TIMEOUT_S
        )
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send_grab(self, shape_name, color_name, x_mm, y_mm, z_byte=0):
        """
        发一次抓取指令，带ACK等待和重试。
        返回: True 下位机确认收到
        """
        shape_id = SHAPE_ID.get(shape_name, 255)
        color_id = COLOR_ID.get(color_name, 255)
        x_i = int(round(x_mm))
        y_i = int(round(y_mm))
        body = bytes(C.FRAME_HEAD) + bytes([shape_id, color_id]) + \
               struct.pack("<hh", x_i, y_i) + bytes([z_byte])
        crc = _crc8(body)
        frame = body + bytes([crc, C.FRAME_TAIL])

        for attempt in range(C.RETRY_TIMES):
            self.ser.write(frame)
            self.ser.flush()
            # 等ACK
            ack = self.ser.read(1)
            if ack and ack[0] == C.ACK_BYTE:
                return True
            time.sleep(0.1)
        return False

    def close(self):
        if self.ser:
            self.ser.close()


if __name__ == "__main__":
    # 自检：发个假指令
    c = Comm()
    c.open()
    ok = c.send_grab("circle", "red", 200, 300, 0)
    print(f"send result: {ok}")
    c.close()
