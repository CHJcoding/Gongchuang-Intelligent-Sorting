#include "motor.h"

/* ============================================================
 *                        内部数据
 * ============================================================ */
//用于for循环统一命令电机
static const uint8_t s_motor_addr[CHASSIS_MOTOR_COUNT] = {
    X_MOTOR_ADDR,    //x轴
    Y_MOTOR_ADDR,    //y轴
    Z_MOTOR_ADDR     //z轴
};

//存储方向
static const uint8_t s_forward_dir[CHASSIS_MOTOR_COUNT] = {
    MOTOR_X_DIR,       //x轴
    MOTOR_Y_DIR,       //y轴
    MOTOR_Z_DIR        //z轴
};

/* ============================================================
 *                        工具函数
 * ============================================================ */
 /**
  * @brief    绝对值函数
  * @param    v  ：传参
  * @Introduction   如果传参为负数，返回他的负值，也就变为正数
  */
//但是不用这个 fabsf可以库函数可以直接代替
static float Chassis_Abs(float v)
{
    return (v < 0.0f) ? -v : v;
}

 /**
  * @brief    毫米/s 转化为 脉冲数
  * @param    mm_s  ：实际需要的速度
  * @Intro    mm/s为电机实际运动距离，输出为RPM表示电机每分钟转多少圈
  */
/* mm/s → RPM（有4舍5入功能）速度模式 */
static uint16_t Chassis_SpeedToRpm(float mm_s)
{
    float rpm = fabsf(mm_s) * 60.0f / MOTOR_LEAD_MM;
    
    if(rpm > (float)MOTOR_MAX_RPM){
        rpm = (float)MOTOR_MAX_RPM;
    }
    return (uint16_t)(rpm + 0.5f);  
}

/* ============================================================
 *                        初始化
 * ============================================================ */

//一次性使能3个电机
HAL_StatusTypeDef Chassis_EnableAll(void)
{
    uint8_t i;

    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        /* 每个电机 snF=1，等待广播同步触发 */
        Emm_V5_En_Control(s_motor_addr[i], true, true);
        HAL_Delay(10U);
    }
    /* 广播同步 */
    Emm_V5_Synchronous_motion(0U);

    return HAL_OK;
}

//一次性失能
HAL_StatusTypeDef Chassis_DisableAll(void)
{
    uint8_t i;

    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_En_Control(s_motor_addr[i], false, false);
    }
    return HAL_OK;
}

//此处停止 是直接让其速度将为0 非失能
HAL_StatusTypeDef Chassis_Stop(void)
{
    uint8_t i;

    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        /* snF = true，等广播 */
        Emm_V5_Stop_Now(s_motor_addr[i], true);
    }
    Emm_V5_Synchronous_motion(0U);

    return HAL_OK;
}

//电机回零（直接设置无运动）！
/* Set all wheel positions to zero for absolute rotation. */
HAL_StatusTypeDef Chassis_SetZeroPositions(void)
{
    uint8_t i;

    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_Reset_CurPos_To_Zero(s_motor_addr[i]);
    }
    HAL_Delay(20U);
    return HAL_OK;
}

HAL_StatusTypeDef Chassis_Init(void)
{
    uint8_t i;

    /* 1. 使能四个电机 */
    Chassis_EnableAll();
    HAL_Delay(100U);

    /* 2. 清堵转保护 */
    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_Reset_Clog_Pro(s_motor_addr[i]);
    }
    HAL_Delay(50U);

    /* 3. 停止 */
    Chassis_Stop();

    /* 4. 把当前姿态记为绝对旋转零点。 */
    (void)Chassis_SetZeroPositions();

    return HAL_OK;
}

/* ============================================================
 *                        预设动作
 * ============================================================ */

/* 距离 → 脉冲数 */
static uint32_t Chassis_DistanceToPulses(uint32_t distance_mm)
{
    float pulses = distance_mm / MOTOR_LEAD_MM * AXIS_PULSES_PER_REV * AXIS_GEAR_RATIO;
    return (uint32_t)(fabsf(pulses) + 0.5f);
}