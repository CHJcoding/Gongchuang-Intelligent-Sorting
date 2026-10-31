#include "motor.h"
#include <math.h>

#if Handeye_calibration
/* 存储驱动器反馈的真实脉冲数 */
volatile int32_t g_real_pulse_A = 0;
volatile int32_t g_real_pulse_B = 0;
volatile int32_t g_real_pulse_Z = 0;
volatile uint8_t g_real_position_valid_A = 0U;
volatile uint8_t g_real_position_valid_B = 0U;
volatile uint8_t g_real_position_valid_Z = 0U;
#endif

/* ============================================================
 *                        内部数据
 * ============================================================ */
static const uint8_t s_motor_addr[CHASSIS_MOTOR_COUNT] = {
    A_MOTOR_ADDR,
    B_MOTOR_ADDR,
    Z_MOTOR_ADDR
};

/* H7 端维护的当前绝对坐标（mm），代表驱动器内部的绝对坐标系 */
static float s_cur_x_mm = 0.0f;       
static float s_cur_y_mm = 0.0f;
static float s_cur_z_mm = 0.0f;

#if Handeye_calibration
/* ============================================================ */
/*                 CAN 反馈解析回调                             */
/* ============================================================ */
static void Chassis_CAN_Rx_Callback(uint32_t id, const uint8_t *data, uint8_t len)
{
    // CAN 数据：功能码、符号、32位位置(大端)、校验字节
    if (len >= 7U && data[0] == 0x36U && data[6] == 0x6BU)
    {
        uint8_t addr = (uint8_t)(id >> 8U);
        uint32_t value = ((uint32_t)data[2] << 24U) |
                         ((uint32_t)data[3] << 16U) |
                         ((uint32_t)data[4] << 8U)  |
                         (uint32_t)data[5];
        int32_t pos = (data[1] != 0U) ? -(int32_t)value : (int32_t)value;
        
        if (addr == A_MOTOR_ADDR) {
            g_real_pulse_A = pos;
            g_real_position_valid_A = 1U;
        } else if (addr == B_MOTOR_ADDR) {
            g_real_pulse_B = pos;
            g_real_position_valid_B = 1U;
        } else if (addr == Z_MOTOR_ADDR) {
            g_real_pulse_Z = pos;
            g_real_position_valid_Z = 1U;
        }
    }
}
#endif

/* ============================================================
 *                        初始化与控制
 * ============================================================ */

HAL_StatusTypeDef Chassis_EnableAll(void)
{
    uint8_t i;
    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_En_Control(s_motor_addr[i], true, true);
        HAL_Delay(10U);
    }
    Emm_V5_Synchronous_motion(0U);
    return HAL_OK;
}

HAL_StatusTypeDef Chassis_DisableAll(void)
{
    uint8_t i;
    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_En_Control(s_motor_addr[i], false, false);
    }
    return HAL_OK;
}

HAL_StatusTypeDef Chassis_Stop(void)
{
    uint8_t i;
    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_Stop_Now(s_motor_addr[i], true);
    }
    Emm_V5_Synchronous_motion(0U);
    return HAL_OK;
}

/* 
 * 【绝对模式核心】每次上电或回零后，必须调用此函数！
 * 强制驱动器把当前物理位置设为 0，并同步 H7 软件坐标。
 */
HAL_StatusTypeDef Chassis_SetZeroPositions(void)
{
    uint8_t i;
    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_Reset_CurPos_To_Zero(s_motor_addr[i]);
    }
    HAL_Delay(20U);
    
    s_cur_x_mm = 0.0f;
    s_cur_y_mm = 0.0f;
    s_cur_z_mm = 0.0f;
    
    return HAL_OK;
}

HAL_StatusTypeDef Chassis_Init(void)
{
    CAN_RegisterExtRxCallback(Chassis_CAN_Rx_Callback);
    
    uint8_t i;
    Chassis_EnableAll();
    HAL_Delay(100U);

    for (i = 0U; i < CHASSIS_MOTOR_COUNT; i++) {
        Emm_V5_Reset_Clog_Pro(s_motor_addr[i]);
    }
    HAL_Delay(50U);

    Chassis_Stop();
    (void)Chassis_SetZeroPositions();

    return HAL_OK;
}


/* ============================================================
 *                        核心算法（绝对模式）
 * ============================================================ */
typedef struct {
    int32_t steps_A;
    int32_t steps_B;
} CoreXY_Target_t;

/* 逆解：XY目标（mm）→ A/B电机目标绝对脉冲数(带符号) */
CoreXY_Target_t corexy_inverse(float X_mm, float Y_mm, float steps_per_mm)
{
    CoreXY_Target_t t;
    t.steps_A = (int32_t)lroundf((X_mm + Y_mm) * steps_per_mm);
    t.steps_B = (int32_t)lroundf((X_mm - Y_mm) * steps_per_mm);
    return t;
}

/**
 * @brief 控制 XY 轴走到绝对坐标 (x_mm, y_mm)
 */
HAL_StatusTypeDef Chassis_MoveTo(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc)
{
    if (x_mm < 0.0f || x_mm > XY_X_MAX || y_mm < 0.0f || y_mm > XY_Y_MAX) 
    {
        return HAL_ERROR; 
    }
    
    // 如果目标与当前记录的坐标完全一致，直接返回，不动作
    if (x_mm == s_cur_x_mm && y_mm == s_cur_y_mm) {
        return HAL_OK;
    }

    /* ==========================================================
     * 1. 计算【目标绝对位置】 (发给驱动器)
     * ========================================================== */
    CoreXY_Target_t target_t = corexy_inverse(x_mm, y_mm, COREXY_STEPS_PER_MM);
    int32_t sA = target_t.steps_A * A_DIR_SIGN;
    int32_t sB = target_t.steps_B * B_DIR_SIGN;
    
    uint8_t  dir_A = (sA >= 0) ? 1U : 0U;
    uint8_t  dir_B = (sB >= 0) ? 1U : 0U;
    uint32_t clk_A = (uint32_t)((sA >= 0) ? sA : -sA);
    uint32_t clk_B = (uint32_t)((sB >= 0) ? sB : -sB);
    
    // 【============ 新增的核心代码：加在这里 ============】
    // 防止绝对模式下发 0 脉冲被张大头固件吞掉，强制给 1 个脉冲
    if (clk_A == 0) clk_A = 1;
    if (clk_B == 0) clk_B = 1;
    
    /* ==========================================================
     * 2. 计算【增量距离】 (仅用于分配速度，防止回0时速度变成1)
     * ========================================================== */
    float dx = x_mm - s_cur_x_mm;
    float dy = y_mm - s_cur_y_mm;
    CoreXY_Target_t delta_t = corexy_inverse(dx, dy, COREXY_STEPS_PER_MM);
    
    uint32_t delta_A = (uint32_t)abs(delta_t.steps_A);
    uint32_t delta_B = (uint32_t)abs(delta_t.steps_B);
    
    /* ==========================================================
     * 3. 按增量距离分配速度与加速度
     * ========================================================== */
    uint32_t d_max = (delta_A > delta_B) ? delta_A : delta_B;
    if (d_max == 0U) d_max = 1U; 
    
    uint16_t v_A = (uint16_t)(((uint32_t)vel_rpm * delta_A) / d_max);
    uint16_t v_B = (uint16_t)(((uint32_t)vel_rpm * delta_B) / d_max);
    if (v_A == 0U) v_A = 1U;
    if (v_B == 0U) v_B = 1U;

    uint8_t acc_A = (uint8_t)(((uint32_t)acc * delta_A) / d_max);
    uint8_t acc_B = (uint8_t)(((uint32_t)acc * delta_B) / d_max);
    if (acc_A == 0U) acc_A = 1U;
    if (acc_B == 0U) acc_B = 1U;

    /* ==========================================================
     * 4. 发送绝对位置指令 (raF = 1)
     * ========================================================== */
    Emm_V5_Pos_Control(A_MOTOR_ADDR, dir_A, v_A, acc_A, clk_A, 1, true);
    Emm_V5_Pos_Control(B_MOTOR_ADDR, dir_B, v_B, acc_B, clk_B, 1, true);

    Emm_V5_Synchronous_motion(0U);
    
    // 5. 更新 H7 端记录的绝对坐标
    s_cur_x_mm = x_mm;
    s_cur_y_mm = y_mm;

    return HAL_OK;
}

/**
 * @brief 获取当前 H7 记录的绝对坐标
 */
void Chassis_GetCurrentPos(float *x_mm, float *y_mm, float *z_mm)
{
    if (x_mm != NULL) *x_mm = s_cur_x_mm;
    if (y_mm != NULL) *y_mm = s_cur_y_mm;
    if (z_mm != NULL) *z_mm = s_cur_z_mm;
}

/**
 * @brief Z 轴走到绝对坐标 z_mm（0 为原点，正数下降，负数上升）
 */
HAL_StatusTypeDef Z_MoveTo(float z_mm, uint16_t vel_rpm, uint8_t acc)
{
    if (z_mm == s_cur_z_mm) return HAL_OK;

    // 目标绝对脉冲
    float pulses_float = fabsf(z_mm) * Z_STEPS_PER_MM;
    int32_t steps = (int32_t)lroundf(pulses_float);

    int32_t signed_steps = (z_mm >= 0.0f) ? steps : -steps;
    if (Z_DIR_SIGN < 0) signed_steps = -signed_steps;

    uint8_t dir = (signed_steps >= 0) ? 1U : 0U;
    uint32_t clk = (uint32_t)((signed_steps >= 0) ? signed_steps : -signed_steps);

    // 发送绝对位置指令
    Emm_V5_Pos_Control(Z_MOTOR_ADDR, dir, vel_rpm, acc, clk, 1, false);
    
    s_cur_z_mm = z_mm;
    return HAL_OK;
}


/* ============================================================
 *                        运动时间估算与等待
 * ============================================================ */
uint32_t Motion_EstimateTimeMs(float distance_mm, uint16_t vel_rpm, uint8_t acc, float mm_per_rev)
{
    if (distance_mm < 0.1f || vel_rpm == 0 || mm_per_rev <= 0.0f) {
        return 0;
    }

    float v_mm_s = ((float)vel_rpm / 60.0f) * mm_per_rev;
    float t_cruise = fabsf(distance_mm) / v_mm_s;

    float t_acc = 0.3f; 
    if (acc > 0) {
        t_acc = 51.0f / (float)acc; 
    } else {
        t_acc = 0.0f; 
    }
    
    float t_total = t_cruise + t_acc;
    uint32_t ms = (uint32_t)(t_total * 1.2f * 1000.0f) + 800U;

    if (ms < 150U) ms = 150U;
    return ms;
}

/**
 * @brief XY 轴绝对移动并阻塞等待
 */
HAL_StatusTypeDef Chassis_MoveToAndWait(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc)
{
    float start_x, start_y;
    Chassis_GetCurrentPos(&start_x, &start_y, NULL);
    
    // 1. 先发送绝对位置移动指令
    if (Chassis_MoveTo(x_mm, y_mm, vel_rpm, acc) != HAL_OK) 
    {
        // 【新增】如果越界导致发送失败，不延时，直接返回
        return HAL_ERROR;
    }
    
    // 2. 只有在发送成功后，才计算增量距离和延时
    float dx = x_mm - start_x;
    float dy = y_mm - start_y;
    
    CoreXY_Target_t t = corexy_inverse(dx, dy, COREXY_STEPS_PER_MM);
    
    float dist_a = fabsf((float)t.steps_A) / COREXY_STEPS_PER_MM;
    float dist_b = fabsf((float)t.steps_B) / COREXY_STEPS_PER_MM;
    float motor_dist = (dist_a > dist_b) ? dist_a : dist_b;
    
    // 3. 延时等待
    uint32_t wait_ms = Motion_EstimateTimeMs(motor_dist, vel_rpm, acc, XY_PULLEY_CIRCUM_MM);
    osDelay(wait_ms);
    
    return HAL_OK;
}

/**
 * @brief Z 轴绝对移动并阻塞等待
 */
void Z_MoveToAndWait(float z_mm, uint16_t vel_rpm, uint8_t acc)
{
    float start_z;
    Chassis_GetCurrentPos(NULL, NULL, &start_z);
    
    // 1. 算出增量物理距离，仅用于估算时间
    float dz = z_mm - start_z;
    float dist = fabsf(dz);
    
    // 2. 发送绝对位置移动指令
    Z_MoveTo(z_mm, vel_rpm, acc);
    
    // 3. 延时等待
    uint32_t wait_ms = Motion_EstimateTimeMs(dist, vel_rpm, acc, Z_GEAR_CIRCUM_MM);
    osDelay(wait_ms);
}

/**
 * @brief 旋转电机绝对角度控制
 */
void Abs_Pluse2Angle(float angle)
{
    float pluse = 3200.0f / 360.0f * angle;
    // raF = 1 绝对位置运动
    Emm_V5_Pos_Control(SPIN_MOTOR_ADDR, 1, 100, 50, pluse, 1, 0);
}

HAL_StatusTypeDef Spin_position(int position)
{
    if (position > 6 || position < 0) 
    {
        return HAL_ERROR;
    }
    Abs_Pluse2Angle(position * 60);
    return HAL_OK;
}


/* ============================================================
 *                        真实位置反馈打印
 * ============================================================ */
#if Handeye_calibration

void Z_GetRealPosition(float *z_mm)
{
    if (z_mm == NULL) return;
    float logical_Z = (float)g_real_pulse_Z * Z_DIR_SIGN;
    *z_mm = logical_Z * Z_GEAR_CIRCUM_MM / MOTOR_POSITION_UNITS_PER_REV;
}

void Chassis_GetRealPosition(float *x_mm, float *y_mm)
{
    float logical_A = (float)g_real_pulse_A * A_DIR_SIGN;
    float logical_B = (float)g_real_pulse_B * B_DIR_SIGN;
    
    float mm_A = logical_A * XY_PULLEY_CIRCUM_MM /
                 (MOTOR_POSITION_UNITS_PER_REV * XY_GEAR_RATIO);
    float mm_B = logical_B * XY_PULLEY_CIRCUM_MM /
                 (MOTOR_POSITION_UNITS_PER_REV * XY_GEAR_RATIO);
    
    *x_mm = (mm_A + mm_B) / 2.0f;
    *y_mm = (mm_A - mm_B) / 2.0f;
}

void Chassis_ReportRealPos(void)
{
    float real_x = 0.0f, real_y = 0.0f, real_z = 0.0f;
    float angle_A = 0.0f, angle_B = 0.0f, angle_Z = 0.0f;
    uint8_t valid_mask;
    static char usb_buf[160];
    int len = 0;
    
    g_real_position_valid_A = 0U;
    g_real_position_valid_B = 0U;
    g_real_position_valid_Z = 0U;
    
    Emm_V5_Read_Sys_Params(A_MOTOR_ADDR, S_CPOS);
    Emm_V5_Read_Sys_Params(B_MOTOR_ADDR, S_CPOS);
    Emm_V5_Read_Sys_Params(Z_MOTOR_ADDR, S_CPOS);
    
    osDelay(20U);
    
    Chassis_GetRealPosition(&real_x, &real_y);
    Z_GetRealPosition(&real_z);
    
    angle_A = (float)g_real_pulse_A * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    angle_B = (float)g_real_pulse_B * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    angle_Z = (float)g_real_pulse_Z * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    
    valid_mask = (uint8_t)((g_real_position_valid_A != 0U ? 1U : 0U) |
                          (g_real_position_valid_B != 0U ? 2U : 0U) |
                          (g_real_position_valid_Z != 0U ? 4U : 0U));
    
    len = snprintf(usb_buf, sizeof(usb_buf),
                   "POS X:%.2f Y:%.2f Z:%.2f | A:%.1f° B:%.1f° Z:%.1f° | Valid:%u\r\n",
                   real_x, real_y, real_z, 
                   angle_A, angle_B, angle_Z, 
                   (unsigned int)valid_mask);
    
    if (len > 0) {
        (void)CDC_Transmit_HS((uint8_t*)usb_buf, (uint16_t)len);
    }
}

void Chassis_ManualMove_TestTask(void)
{
    osDelay(1000); 
    Chassis_DisableAll();
    
    static const char ready_msg[] = "Motor Disabled. Ready for manual move.\r\n";
    CDC_Transmit_HS((uint8_t*)ready_msg, (uint16_t)(sizeof(ready_msg) - 1U));
    
    for(;;)
    {
        Chassis_ReportRealPos();
        osDelay(100); 
    }
}

#endif
