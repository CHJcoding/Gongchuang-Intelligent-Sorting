#ifndef __MOTOR_CONFIG_H__
#define __MOTOR_CONFIG_H__


//手眼标定是否打开
#define Handeye_calibration 1


/* ========== 电机地址 ========== */
#define A_MOTOR_ADDR    1U
#define B_MOTOR_ADDR    2U
#define Z_MOTOR_ADDR    3U
#define CHASSIS_MOTOR_COUNT 3U

/* ============================================================
 *  驱动器 / 电机基础参数（所有轴共用）
 * ============================================================ */
 
/* 1.8° 两相步进电机：整步 200 步/圈 */
#define MOTOR_FULL_STEPS      200U
/* 张大头默认 16 细分；用 Emm_V5_Modify_MicroStep 改过就同步改这里 */
#define MOTOR_MICROSTEP       16U
/* 电机转一圈需要的脉冲数 = 200 * 16 = 3200 */
#define MOTOR_PULSES_PER_REV  (MOTOR_FULL_STEPS * MOTOR_MICROSTEP)

/* S_CPOS 反馈单位：编码器位置每转 65536 个计数 */
#define MOTOR_POSITION_UNITS_PER_REV  65536.0f

/* 驱动器协议速度上限 5000RPM；这里填机械/工艺允许的上限 */
#define MOTOR_MAX_RPM         3000U

/* ============================================================
 *  方向极性：+1 或 -1
 *  用法：int32_t s = t.steps_A * A_DIR_SIGN;
 *  实测方向反了就把对应宏改成 -1，不要动逆解公式
 * ============================================================ */
#define A_DIR_SIGN            (+1)
#define B_DIR_SIGN            (+1)
#define Z_DIR_SIGN            (+1)


/* ============================================================
 *  XY 轴机械参数（CoreXY，同步带 + 同步轮）
 * ============================================================ */
/* 同步轮齿数：20T 常见；换成 16T/24T 只改这里 */
#define XY_PULLEY_TEETH       20U
 /* GT2 同步带节距 2.0mm */
#define XY_BELT_PITCH_MM      2.0f
/* 节圆周长 = 齿数 × 节距 = 20 × 2 = 40mm */
#define XY_PULLEY_CIRCUM_MM   ((float)XY_PULLEY_TEETH * XY_BELT_PITCH_MM)
/* 同步轮与电机直连填 1.0f；带减速箱填实际减速比 */
#define XY_GEAR_RATIO         1.0f
/* 每毫米脉冲数 = 3200 / 40 = 80 */
#define COREXY_STEPS_PER_MM   \
    ((float)MOTOR_PULSES_PER_REV * XY_GEAR_RATIO / XY_PULLEY_CIRCUM_MM)

    
/* ============================================================
 *  运动默认值（Chassis_MoveTo 不传参时用）
 * ============================================================ */
#define XY_VEL_RPM_DEF   300U    /* 走得远那一轴的速度上限 */
#define XY_ACC_DEF       50U     /* 0~255，0 是“直接启动”，不要用 0 */
#define Z_VEL_RPM_DEF         300U
#define Z_ACC_DEF             50U
    
    
/* ============================================================
 *  数学常量
 * ============================================================ */
#define MOTOR_PI              3.14159265358979f


#endif
