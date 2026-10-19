#ifndef __SERVO_CONFIG_H__
#define __SERVO_CONFIG_H__

/* Pulse limits are expressed in microseconds. */
#define LIMIT_PULSE 1

#if LIMIT_PULSE == 0
#define SERVO_PULSE_MIN        500U
#define SERVO_PULSE_MAX        2500U
#endif

#define SERVO_PERIOD_US        20000U
#define SERVO_TIMER_PRESCALER  ((SERVO_TIMER_CLOCK_HZ / SERVO_TIMER_TICK_HZ) - 1U)
#define SERVO_TIMER_PERIOD     (SERVO_PERIOD_US - 1U)


//舵机20kg款，PWM500US - 2500US   
//逆时针！对应：500作为0°，1850则为180°，2500为270°（最大）1175为90° 不是算的很精细   
#define SERVO_ADDR1            1U
#define SERVO_ADDR2            2U
#define VALVE_SERVO3_ID        3U
#define PUMP_SERVO4_ID         4U

#define SERVO_COUNT            4U

/* Servo 1/2 return to the middle position at startup. */
#define SERVO1_SAFE_PULSE      500U            //后期调整
#define SERVO2_SAFE_PULSE      1500U            //后期调整

/* 气泵安全脉冲参数（初始化调用）. */
#define SERVO3_SAFE_PULSE      PNEUMATIC_OFF_PULSE_US
#define SERVO4_SAFE_PULSE      PNEUMATIC_OFF_PULSE_US


/* 气泵打开关闭参数. */
#define PNEUMATIC_OFF_PULSE_US 544U
#define PNEUMATIC_ON_PULSE_US  2400U



#endif
