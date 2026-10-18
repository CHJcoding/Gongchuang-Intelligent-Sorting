#ifndef __SERVO_CONFIG_H__
#define __SERVO_CONFIG_H__

/* 脉宽范围（us），按你舵机手册可改。
 * 常见：500~2500 为扩展范围，1000~2000 为保守范围 */
#define SERVO_PULSE_MIN    500U
#define SERVO_PULSE_MID    1500U
#define SERVO_PULSE_MAX    2500U
#define SERVO_PERIOD_US    20000U   /* 20ms -> 50Hz，不要改 */

#define SERVO_COUNT         4       //舵机数量
/*后期改名，下面是舵机具体控制的是什么部分*/
#define SERVO_ADDR1         1
#define SERVO_ADDR2         1
#define SERVO_ADDR3         1
#define SERVO_ADDR4         1

/*每个舵机的安全也是初始化时候的位姿脉冲*/
#define SERVO1_SAFE_PULSE    1500U
#define SERVO2_SAFE_PULSE    1500U
#define SERVO3_SAFE_PULSE    1500U
#define SERVO4_SAFE_PULSE    1500U

#endif
