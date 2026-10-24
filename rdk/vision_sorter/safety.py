# -*- coding: utf-8 -*-
"""下发前的安全检查（纯逻辑，不 import ROS，方便离线单测）。

为什么要单独一个文件：这些判断决定"要不要把坐标发给龙门"，
属于安全相关逻辑，必须能被单测覆盖；放在 rclpy 节点里就只能上板才能验。

两道闸：

1. **手眼标定闸**：没标定时 `pixel_to_robot()` 用的是 `config.HAND_EYE_M`
   的占位值，坐标是假的。以前只打 ERROR 日志然后照发 —— 龙门装好后一旦
   忘了标定就会满场乱走。
2. **串口闸**：USB 转串口会掉（热插拔、供电、驱动）。掉线时如果静默进模拟
   模式，日志看起来"抓得很正常"，其实一个字节都没发出去。

两道闸都是"默认关闭、显式放开"：台架调试要开，用 ROS 参数或 config 显式打开。
"""

# ---- 拒绝原因码（同时作为 /sort/state 的状态字符串）----
REASON_UNCALIBRATED = "BLOCKED_UNCALIBRATED"
REASON_SERIAL_DOWN = "BLOCKED_SERIAL_DOWN"

_HUMAN = {
    REASON_UNCALIBRATED: "手眼未标定，坐标是占位矩阵算出来的假值",
    REASON_SERIAL_DOWN: "串口未就绪，坐标发不到 STM32",
}


def check_downlink(calibrated, allow_uncalibrated=False,
                   serial_ok=True, serial_sim=False):
    """判断能否下发一次抓取指令。

    参数:
        calibrated          : hand_eye.is_calibrated()
        allow_uncalibrated  : 显式绕过标定闸（台架调试）
        serial_ok           : 串口当前是否可用
        serial_sim          : 是否显式处于模拟模式（用户主动选的，不是故障）

    返回:
        None            —— 允许下发
        (code, 原因中文) —— 拒绝，code 可直接当状态字符串发出去
    """
    if not calibrated and not allow_uncalibrated:
        return (REASON_UNCALIBRATED, _HUMAN[REASON_UNCALIBRATED])
    # serial_sim=True 是用户主动选的模拟，属于合法运行方式，不算故障
    if not serial_ok and not serial_sim:
        return (REASON_SERIAL_DOWN, _HUMAN[REASON_SERIAL_DOWN])
    return None


def human(code):
    """把原因码翻译成人话（未知码原样返回）。"""
    return _HUMAN.get(code, str(code))


def describe_gates(require_calib, allow_uncalibrated, calibrated,
                   serial_ok, serial_sim, sim_on_fail=False):
    """启动时打印的安全状态，逐条列出（方便一眼看出"现在到底能不能动"）。

    返回 [(level, 文本)]，level 是 "warn" / "info"。
    ⚠ 必须把 level 一起返回、由调用方在两个固定的代码行上分别调用
    logger.warn / logger.info：rclpy 按【调用点】缓存 severity，
    同一行代码用不同 severity 调用会抛
    `ValueError: Logger severity cannot be changed between calls`。
    """
    lines = []

    if not require_calib:
        lines.append(("warn", "安全闸[标定]: ⚠ 已关闭（SAFETY_REQUIRE_CALIB=False，"
                              "未标定也会下发 —— 只允许在台架上这么干）"))
    elif calibrated:
        lines.append(("info", "安全闸[标定]: ✅ 已通过（hand_eye.json 已加载）"))
    elif allow_uncalibrated:
        lines.append(("warn", "安全闸[标定]: ⚠ 被 allow_uncalibrated=true 绕过，"
                              "下发的是占位矩阵算出的假坐标"))
    else:
        lines.append(("warn", "安全闸[标定]: 🔒 关闭中（未标定，拒绝下发；"
                              "要演示请加 -p allow_uncalibrated:=true）"))

    if serial_sim:
        lines.append(("info", "安全闸[串口]: 模拟模式（用户显式指定 serial_sim=true，"
                              "只打印不发送）"))
    elif serial_ok:
        lines.append(("info", "安全闸[串口]: ✅ 已连接"))
    elif sim_on_fail:
        lines.append(("warn", "安全闸[串口]: ⚠ 打不开，按 SERIAL_SIM_ON_FAIL=True 退化为模拟"
                              "（日志会假装在抓，注意别误判）"))
    else:
        lines.append(("warn", "安全闸[串口]: 🔒 故障（打不开，拒绝下发并周期性重试）"))

    return lines