#ifndef __MOTOR_CONFIG_H__
#define __MOTOR_CONFIG_H__

/* ========== 电机地址 ========== */
#define X_MOTOR_ADDR    1U
#define Y_MOTOR_ADDR    2U
#define Z_MOTOR_ADDR    3U
#define CHASSIS_MOTOR_COUNT 3U

/* ---------------- Mechanical dimensions ---------------- */
#define MOTOR_LEAD_MM             200U  //【需要测试】电机转一圈 执行机构走多少，可以通过张大头位置模式测试
#define AXIS_PULSES_PER_REV  3200.0f   /* 默认16细分，建议读回确认 */
#define AXIS_GEAR_RATIO      1.0f      /* 直连填1.0，带减速箱必须改！ */

/* ---------------- Motor pulses ---------------- */
#define MOTOR_PULSES_PER_REV          3200U               /* 电机转一圈需要的脉冲数 200 步 × 16 细分 = 3200 */
#define MOTOR_MAX_RPM                 3000U               //电机最大转速

/* ---------------- Logical forward direction: 0=CW, 1=CCW（电机装反了就改这） ---------------- */
#define MOTOR_X_DIR          0U          
#define MOTOR_Y_DIR          1U
#define MOTOR_Z_DIR          0U

/* ------------------- Math ------------------- */
#define MOTOR_PI          3.14159265358979f


#endif
