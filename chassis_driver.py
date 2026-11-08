#!/usr/bin/env python3
"""
chassis_driver 骨架版
----------------------
职责：
  - 订阅上层处理节点的结果（速度、扫码、视觉）
  - 按协议打包
  - 通过串口发送给 STM32
  - 可选：接收 STM32 反馈并转发到 ROS 话题

待填：
  - pack_motion() / pack_servo() / pack_extend()  
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int32

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class ChassisDriver(Node):
    def __init__(self):
        super().__init__('chassis_driver')

        # ============================================================
        # 配置区（等对接时改）
        # ============================================================
        self.SERIAL_PORT = '/dev/ttyUSB0'
        self.BAUD_RATE = 115200
        self.SEND_HZ = 20                # 定时发送频率
        self.FRAME_HEAD = 0x6C           # 'l'
        self.FRAME_TAIL = 0x2F           # '/'

        # 任务类型（等协议确定后填）
        self.TASK_MOTION = 0x01          # 运动控制
        self.TASK_SERVO = 0x02           # 舵机
        self.TASK_EXTEND = 0x03          # 伸缩

        # ============================================================
        # 串口初始化（失败不退出，走调试模式）
        # ============================================================
        self.ser = None
        if HAS_SERIAL:
            try:
                self.ser = serial.Serial(
                    self.SERIAL_PORT, self.BAUD_RATE, timeout=0.0)
                self.get_logger().info(f'串口已打开: {self.SERIAL_PORT}')
            except Exception as e:
                self.get_logger().warn(f'串口打开失败（调试模式）: {e}')
        else:
            self.get_logger().warn('未安装 pyserial，调试模式')

        # ============================================================
        # 状态缓存（订阅到的所有数据都存这里）
        # ============================================================
        self.state = {
            # 运动（来自 /cmd_vel_muxed）
            'vx': 0.0,
            'vy': 0.0,
            'wz': 0.0,
            # 舵机（预留）
            'servo_angle': 0,
            # 伸缩（预留）
            'extend_pos': 0,
            # 上层状态（如果协议要回传给 STM32）
            'scanner': '',
            'vision': '',
        }

        # ============================================================
        # 订阅：从各处理节点拿数据
        # ============================================================
           # 1. 主运动指令（由仲裁节点输出）
        self.create_subscription(
            Twist, '/cmd_vel_muxed', self.cb_cmd_vel, 10)

        # 2. 扫码结果
        self.create_subscription(
            String, '/scanner/parsed', self.cb_scanner, 10)

        # 3. 视觉结果
        self.create_subscription(
            String, '/vision/result', self.cb_vision, 10)

        # 4. 舵机控制
        self.create_subscription(
            Int32, '/servo/target', self.cb_servo, 10)

        # 5. 伸缩控制
        self.create_subscription(
            Int32, '/extend/target', self.cb_extend, 10)

        # ============================================================
        # 发布：调试用（让虚拟机/其他节点看收发内容）
        # ============================================================
        self.pub_tx_debug = self.create_publisher(String, '/driver/tx_debug', 10)
        self.pub_rx_raw = self.create_publisher(String, '/driver/rx_raw', 10)

        # ============================================================
        # 定时器：周期性发送
        # ============================================================
        self.create_timer(1.0 / self.SEND_HZ, self.tick_send)
        self.create_timer(1.0 / 50.0, self.tick_recv)

        self.get_logger().info(
            f'chassis_driver started | {self.SEND_HZ}Hz | port={self.SERIAL_PORT}')

    # ================================================================
    # 订阅回调：只更新缓存，不做发送
    # ================================================================
    def cb_cmd_vel(self, msg: Twist):
        self.state['vx'] = msg.linear.x
        self.state['vy'] = msg.linear.y
        self.state['wz'] = msg.angular.z

    def cb_scanner(self, msg: String):
        self.state['scanner'] = msg.data

    def cb_vision(self, msg: String):
        self.state['vision'] = msg.data

    def cb_servo(self, msg: Int32):
        self.state['servo_angle'] = msg.data

    def cb_extend(self, msg: Int32):
        self.state['extend_pos'] = msg.data

    # ================================================================
    # 定时发送：打包当前状态 → 串口
    # ================================================================
    def tick_send(self):
        # 1. 运动帧
        frame = self.pack_motion(
            self.state['vx'], self.state['vy'], self.state['wz'])
        self.write_serial(frame)

        # 2. 舵机帧（如果值有变化才发，或每次都发，看协议）
        # frame = self.pack_servo(self.state['servo_angle'])
        # self.write_serial(frame)

        # 3. 伸缩帧
        # frame = self.pack_extend(self.state['extend_pos'])
        # self.write_serial(frame)

    # ================================================================
    # 协议打包
    # ================================================================
    def pack_motion(self, vx: float, vy: float, wz: float) -> bytes:
        """
        运动控制帧 —— 待填
        参数：
            vx: m/s
            vy: m/s
            wz: rad/s
        返回：
            bytes 类型的完整帧
        """
        # 临时实现（占位）：把 vx/vy/wz 转成整数放进帧里
        import struct
        vx_i = max(-32768, min(32767, int(vx * 1000)))  # mm/s
        vy_i = max(-32768, min(32767, int(vy * 1000)))
        wz_i = max(-32768, min(32767, int(wz * 1000)))  # mrad/s
        data_len = 6
        frame = struct.pack('<BBhhhBB',
                            self.FRAME_HEAD, self.TASK_MOTION,
                            vx_i, vy_i, wz_i,
                            data_len, 0x00)
        frame += bytes([self.FRAME_TAIL])
        return frame

    def pack_servo(self, angle: int) -> bytes:
        """舵机控制帧 —— 待填"""
        # TODO: 等协议
        return b''

    def pack_extend(self, pos: int) -> bytes:
        """伸缩控制帧 —— 待填"""
        # TODO: 等协议
        return b''

    # ================================================================
    # 串口读写（通用，不用改）
    # ================================================================
    def write_serial(self, frame: bytes):
        if not frame:
            return
        if self.ser is not None:
            try:
                self.ser.write(frame)
            except Exception as e:
                self.get_logger().error(f'串口写失败: {e}')
        # 调试话题
        msg = String()
        msg.data = frame.hex()
        self.pub_tx_debug.publish(msg)

    def tick_recv(self):
        """读 STM32 反馈，原样转发到 /driver/rx_raw"""
        if self.ser is None:
            return
        try:
            n = self.ser.in_waiting
            if n > 0:
                data = self.ser.read(n)
                msg = String()
                msg.data = data.hex()
                self.pub_rx_raw.publish(msg)
        except Exception as e:
            self.get_logger().error(f'串口读失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ChassisDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()