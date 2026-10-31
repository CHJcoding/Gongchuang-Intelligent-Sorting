#ifndef __MOTOR_CONFIG_H__
#define __MOTOR_CONFIG_H__


//手眼标定是否打开
#define Handeye_calibration 1

/* ============================================================
 *  实际位置参数
 * ============================================================ */
 /* 分拣动作物理参数 */
#define Z_SAFE_HEIGHT        0.0f   // Z轴安全高度（抬起，负数代表向上）
#define Z_PICK_HEIGHT        43.0f   // Z轴抓取高度（下探，正数代表向下）
#define Z_PLACE_HEIGHT       20.0f   // Z轴放置高度

#define XY_VEL              150U     // XY 移动速度 (RPM)
#define XY_ACC              40U      // XY 加速度
#define Z_VEL               200U     // Z 移动速度 (RPM)
#define Z_ACC               40U      // Z 加速度

//点位x 和点位y 放置点
#define PUT_PLACE_X            0.0f    // 放置区中心 X 坐标
#define PUT_PLACE_Y            0.0f    // 放置区中心 Y 坐标

//过度点
#define PUT_TRANSITION_PLACE_X  0.0f
#define PUT_TRANSITION_PLACE_Y  160.0f 

/* ============================================================
 *  XY 轴物理安全边界 (单位: mm)
 *  根据你的机械行程修改，防止视觉发来非法坐标导致撞机
 * ============================================================ */
#define XY_X_MIN   0.0f
#define XY_X_MAX   150.0f   // 假设 X 轴最大 300mm
#define XY_Y_MIN   0.0f
#define XY_Y_MAX   150.0f   // 假设 Y 轴最大 300mm

/* ========== 电机地址 ========== */
#define A_MOTOR_ADDR    1U
#define B_MOTOR_ADDR    2U
#define Z_MOTOR_ADDR    3U
#define SPIN_MOTOR_ADDR     4U  //用于旋转
#define CHASSIS_MOTOR_COUNT 4U
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
#define XY_PULLEY_TEETH       20U       //20U
 /* GT2 同步带节距 2.0mm */
#define XY_BELT_PITCH_MM      2.0f      //2.0f
/* 节圆周长 = 齿数 × 节距 = 20 × 2 = 40mm */
#define XY_PULLEY_CIRCUM_MM   ((float)XY_PULLEY_TEETH * XY_BELT_PITCH_MM)
/* 同步轮与电机直连填 1.0f；带减速箱填实际减速比 */
#define XY_GEAR_RATIO         1.0f
/* 每毫米脉冲数 = 3200 / 40 = 80 */
#define COREXY_STEPS_PER_MM   \
    ((float)MOTOR_PULSES_PER_REV * XY_GEAR_RATIO / XY_PULLEY_CIRCUM_MM)

/* ============================================================
 *  Z 轴机械参数（齿轮齿条）
 * ============================================================ */
/* 请根据你实际买的齿轮修改下面两个值 */
#define Z_GEAR_MODULE         1.0f      // 齿轮模数
#define Z_GEAR_TEETH          34.0f     // 齿轮齿数 36 * pai 

/* 齿轮节圆直径 = 模数 × 齿数 */
#define Z_GEAR_DIAMETER_MM    (Z_GEAR_MODULE * Z_GEAR_TEETH)
/* 齿轮转一圈齿条走的距离（节圆周长） = π × 直径 */
#define Z_GEAR_CIRCUM_MM      (MOTOR_PI * Z_GEAR_DIAMETER_MM)

/* Z轴减速比（电机直连齿轮填1.0） */
#define Z_GEAR_RATIO          1.0f

/* Z 轴每毫米对应的脉冲数 */
#define Z_STEPS_PER_MM        ((float)MOTOR_PULSES_PER_REV * Z_GEAR_RATIO / Z_GEAR_CIRCUM_MM)

    
/* ============================================================
 *  数学常量
 * ============================================================ */
#define MOTOR_PI              3.14159265358979f


#endif
