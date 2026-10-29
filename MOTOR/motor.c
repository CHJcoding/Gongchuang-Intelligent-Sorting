#include "motor.h"

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
//用于for循环统一命令电机
static const uint8_t s_motor_addr[CHASSIS_MOTOR_COUNT] = {
    A_MOTOR_ADDR,    //A轴
    B_MOTOR_ADDR,    //B轴
    Z_MOTOR_ADDR     //z轴
};

/* H7 端维护的当前绝对坐标（mm）当前坐标！ */
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
        }
        else if (addr == Z_MOTOR_ADDR) { // 匹配 Z 轴
            g_real_pulse_Z = pos;
            g_real_position_valid_Z = 1U;
        }
    }
}
#endif

/* ============================================================
 *                        初始化
 * ============================================================ */

//一次性使能4个电机
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
    
    /* 关键：软件坐标也清零 */
    s_cur_x_mm = 0.0f;
    s_cur_y_mm = 0.0f;
    s_cur_z_mm = 0.0f;
    
    return HAL_OK;
}

HAL_StatusTypeDef Chassis_Init(void)
{
    /* 注册 CAN 接收回调 */
    CAN_RegisterExtRxCallback(Chassis_CAN_Rx_Callback);
    
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
 *                        核心算法
 * ============================================================ */
//CoreXY 逆解结果，单位：脉冲
typedef struct {
    int32_t steps_A;
    int32_t steps_B;
} CoreXY_Target_t;

/* 逆解：XY目标（mm）→ A/B电机脉冲数(带符号) */
CoreXY_Target_t corexy_inverse(float X_mm, float Y_mm, float steps_per_mm)
{
    CoreXY_Target_t t;
    t.steps_A = (int32_t)lroundf((X_mm + Y_mm) * steps_per_mm);
    t.steps_B = (int32_t)lroundf((X_mm - Y_mm) * steps_per_mm);
    return t;
}

//使用：Chassis_MoveTo(100.0f, 50.0f, 200U, 50U); 走到(100mm, 50mm)，电机转速上限 200RPM，加速度 50
HAL_StatusTypeDef Chassis_MoveTo(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc)
{
    
     /* 1. 目标 - 当前 = 增量 */
    float dx = x_mm - s_cur_x_mm;
    float dy = y_mm - s_cur_y_mm;
    
    /* 2. 增量逆解 */
    CoreXY_Target_t t = corexy_inverse(dx, dy, COREXY_STEPS_PER_MM);

    /* 3. 方向极性修正（只影响符号，不影响运动距离） */
    int32_t sA = t.steps_A * A_DIR_SIGN;
    int32_t sB = t.steps_B * B_DIR_SIGN;
    
    /* 4. 带符号脉冲 → 方向 + 绝对值 */
    uint8_t  dir_A = (sA >= 0) ? 1U : 0U;      /* 1=CCW，0=CW，按电机接线定 */
    uint8_t  dir_B = (sB >= 0) ? 1U : 0U;
    uint32_t clk_A = (uint32_t)((sA >= 0) ? sA : -sA);
    uint32_t clk_B = (uint32_t)((sB >= 0) ? sB : -sB);
    
    /* 5. 已经到目标位置，不动，但坐标要同步 */
    if (clk_A == 0U && clk_B == 0U) {
        s_cur_x_mm = x_mm;
        s_cur_y_mm = y_mm;
        return HAL_OK;
    }
    
    /* 6. 按距离比例分配速度/加速度，保证两轴同时到位 */
    uint32_t d_max = (clk_A > clk_B) ? clk_A : clk_B;
    if (d_max == 0U) d_max = 1U;   /* 防除零；理论上走不到这里 */
    
    uint16_t v_A = (uint16_t)(((uint32_t)vel_rpm * clk_A) / d_max);
    uint16_t v_B = (uint16_t)(((uint32_t)vel_rpm * clk_B) / d_max);
    if (v_A == 0U) v_A = 1U;       /* 速度不能为 0 */
    if (v_B == 0U) v_B = 1U;

    uint8_t acc_A = (uint8_t)(((uint32_t)acc * clk_A) / d_max);
    uint8_t acc_B = (uint8_t)(((uint32_t)acc * clk_B) / d_max);
    if (acc_A == 0U) acc_A = 1U;   /* acc=0 是“直接启动”，必须兜底 */
    if (acc_B == 0U) acc_B = 1U;

    /* 7. 两轴都带 snF=true，等广播；
     *    raF=2：相对“当前实时位置”，避免上一段未走完时叠加误差
     */
    //位置模式！
    Emm_V5_Pos_Control(A_MOTOR_ADDR, dir_A, v_A, acc_A, clk_A, 2, true);
    Emm_V5_Pos_Control(B_MOTOR_ADDR, dir_B, v_B, acc_B, clk_B, 2, true);

    /* 8. 广播同步，两轴同一时刻启动 */
    Emm_V5_Synchronous_motion(0U);
    
    /* 9. 更新 H7 端坐标 */
    s_cur_x_mm = x_mm;
    s_cur_y_mm = y_mm;


    return HAL_OK;
    
}

/**
 * @brief 获取当前 H7 记录的绝对坐标
 * @param x_mm 输出 X 坐标
 * @param y_mm 输出 Y 坐标
 */
void Chassis_GetCurrentPos(float *x_mm, float *y_mm, float *z_mm)
{
    if (x_mm != NULL) *x_mm = s_cur_x_mm;
    if (y_mm != NULL) *y_mm = s_cur_y_mm;
    if (z_mm != NULL) *z_mm = s_cur_z_mm;
}


/**
 * @brief Z 轴绝对位置控制（齿轮齿条）
 * @param z_mm      目标绝对位置（0 为原点，正数下降，负数上升）
 * @param vel_rpm   电机转速 (RPM)
 * @param acc       加速度档位
 */
HAL_StatusTypeDef Z_MoveTo(float z_mm, uint16_t vel_rpm, uint8_t acc)
{
    // 1. 计算增量
    float dz = z_mm - s_cur_z_mm;
    
    // 2. 距离转脉冲
    float pulses_float = fabsf(dz) * Z_STEPS_PER_MM;
    int32_t steps = (int32_t)lroundf(pulses_float);
    
    if (steps == 0) {
        s_cur_z_mm = z_mm;
        return HAL_OK; // 距离太近，不需要动
    }

    // 3. 方向判定：dz >= 0 代表向下，dz < 0 代表向上
    uint8_t dir = (dz >= 0.0f) ? 1U : 0U;
    
    // 如果 Z_DIR_SIGN 是 -1，说明电机接线反了，翻转方向
    if (Z_DIR_SIGN < 0) {
        dir = !dir;
    }

    uint32_t clk = (uint32_t)steps;

    // 4. 下发位置指令 (raF=2 相对当前实时位置)
    Emm_V5_Pos_Control(Z_MOTOR_ADDR, dir, vel_rpm, acc, clk, 2, false);
    
    // 5. 更新软件坐标
    s_cur_z_mm = z_mm;

    return HAL_OK;
}


/**
 * @brief 通用运动时间估算（基于梯形曲线近似）
 * @param distance_mm 移动的物理距离（传绝对值即可）
 * @param vel_rpm     电机速度 (RPM)
 * @param acc         加速度档位 (0~255)
 * @param mm_per_rev  电机转一圈对应的物理距离 (同步轮周长 / 齿轮周长)
 * @return 建议的延时时间 (毫秒)
 */
uint32_t Motion_EstimateTimeMs(float distance_mm, uint16_t vel_rpm, uint8_t acc, float mm_per_rev)
{
    if (distance_mm < 0.1f || vel_rpm == 0 || mm_per_rev <= 0.0f) {
        return 0;
    }

    // 1. 计算线速度 (mm/s)
    float v_mm_s = ((float)vel_rpm / 60.0f) * mm_per_rev;

    // 2. 计算纯匀速运动所需时间 (秒)
    float t_cruise = fabsf(distance_mm) / v_mm_s;

    // 3. 估算加减速补偿时间
    // 张大头的梯形曲线中，acc 越大，加速越快，耗时越少。
    // 假设 acc=255 时，加速 + 减速总共耗时约 0.2 秒。
    // 那么耗时与 acc 成反比：t_acc ≈ 51.0 / acc
    float t_acc = 0.3f; // 默认给一个保守值
    if (acc > 0) {
        t_acc = 51.0f / (float)acc; 
    } else {
        t_acc = 0.0f; // acc=0 是直接启动，无加减速时间
    }
    
    // 4. 纯理论总时间 = 匀速时间 + 加减速时间
    float t_total = t_cruise + t_acc;

    // 5. 加上 20% 的安全余量（防止丢步、机械阻力） + 基础响应延时 300ms
    uint32_t ms = (uint32_t)(t_total * 1.2f * 1000.0f) + 800U;

    // 兜底：即使距离极短，驱动器也需要时间响应，至少延时 150ms
    if (ms < 150U) ms = 150U;

    return ms;
}

//整合函数1：对应xy轴运动
void Chassis_MoveToAndWait(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc)
{
    float start_x, start_y;
    Chassis_GetCurrentPos(&start_x, &start_y, NULL);
    
    // 1. 算出物理直线距离
    float dx = x_mm - start_x;
    float dy = y_mm - start_y;
    
        /* 增量逆解 → A/B 轴各自要走的脉冲 */
    CoreXY_Target_t t = corexy_inverse(dx, dy, COREXY_STEPS_PER_MM);
    
    /* 把脉冲换回"该电机需要走的等效物理距离(mm)"，
       取两轴中更远的那一个作为运动时间估算基准 */
    float dist_a = fabsf((float)t.steps_A) / COREXY_STEPS_PER_MM;
    float dist_b = fabsf((float)t.steps_B) / COREXY_STEPS_PER_MM;
    float motor_dist = (dist_a > dist_b) ? dist_a : dist_b;
    
    // 2. 发送移动指令
    Chassis_MoveTo(x_mm, y_mm, vel_rpm, acc);
    
    // 3. 用通用函数算延时（XY 转一圈是同步轮周长）
    uint32_t wait_ms = Motion_EstimateTimeMs(motor_dist, vel_rpm, acc, XY_PULLEY_CIRCUM_MM);
    osDelay(wait_ms);
}


void Z_MoveToAndWait(float z_mm, uint16_t vel_rpm, uint8_t acc)
{
    float start_z;
    Chassis_GetCurrentPos(NULL, NULL, &start_z);
    
    // 1. 算出物理直线距离
    float dz = z_mm - start_z;
    float dist = fabsf(dz);
    
    // 2. 发送移动指令
    Z_MoveTo(z_mm, vel_rpm, acc);
    
    // 3. 用通用函数算延时（Z 转一圈是齿轮周长）
    uint32_t wait_ms = Motion_EstimateTimeMs(dist, vel_rpm, acc, Z_GEAR_CIRCUM_MM);
    osDelay(wait_ms);
}

void Abs_Pluse2Angle(float angle)
{
    float pluse = 3200.0f / 360.0f * angle;
    
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



//用于实时发送给上位机当前坐标
#if Handeye_calibration

/* 获取 Z 轴真实的物理坐标 (基于编码器反馈) */
void Z_GetRealPosition(float *z_mm)
{
    if (z_mm == NULL) return;

    // 1. 还原逻辑脉冲（乘上 Z 轴极性方向）
    float logical_Z = (float)g_real_pulse_Z * Z_DIR_SIGN;
    
    // 2. 编码器计数转毫米（使用齿轮周长 Z_GEAR_CIRCUM_MM）
    // 假设 Z 轴是直连齿轮齿条，没有额外的减速比
    *z_mm = logical_Z * Z_GEAR_CIRCUM_MM / MOTOR_POSITION_UNITS_PER_REV;
}

/* 获取真实的物理坐标 (基于编码器反馈) */
void Chassis_GetRealPosition(float *x_mm, float *y_mm)
{
    // 1. 还原逻辑脉冲（处理方向极性）
    float logical_A = (float)g_real_pulse_A * A_DIR_SIGN;
    float logical_B = (float)g_real_pulse_B * B_DIR_SIGN;
    
    // 2. 编码器计数转毫米（每转 65536 计数）
    float mm_A = logical_A * XY_PULLEY_CIRCUM_MM /
                 (MOTOR_POSITION_UNITS_PER_REV * XY_GEAR_RATIO);
    float mm_B = logical_B * XY_PULLEY_CIRCUM_MM /
                 (MOTOR_POSITION_UNITS_PER_REV * XY_GEAR_RATIO);
    
    // 3. CoreXY 正解公式
    *x_mm = (mm_A + mm_B) / 2.0f;
    *y_mm = (mm_A - mm_B) / 2.0f;
}

/**
 * @brief 单次读取真实坐标并通过 USB 发送
 * @note 电机使能或失能状态下均可读取
 */
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
    // 1. 向 A/B 电机发送读取实时位置 (S_CPOS) 的 CAN 指令
    Emm_V5_Read_Sys_Params(A_MOTOR_ADDR, S_CPOS);
    Emm_V5_Read_Sys_Params(B_MOTOR_ADDR, S_CPOS);
    Emm_V5_Read_Sys_Params(Z_MOTOR_ADDR, S_CPOS);
    
    // 2. 给 CAN 总线时间回传数据 (根据波特率给 10~20ms)
    osDelay(20U);
    
    // 3. 计算真实坐标
    Chassis_GetRealPosition(&real_x, &real_y);
    Z_GetRealPosition(&real_z);
    
    angle_A = (float)g_real_pulse_A * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    angle_B = (float)g_real_pulse_B * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    angle_Z = (float)g_real_pulse_Z * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    valid_mask = (uint8_t)((g_real_position_valid_A != 0U ? 1U : 0U) |
                          (g_real_position_valid_B != 0U ? 2U : 0U) |
                          (g_real_position_valid_Z != 0U ? 4U : 0U));
    
    // 4. 通过 USB 发送出来
    len = snprintf(usb_buf, sizeof(usb_buf),
                   "POS X:%.2f Y:%.2f Z:%.2f | A:%.1f° B:%.1f° Z:%.1f° | Valid:%u\r\n",
                   real_x, real_y, real_z, 
                   angle_A, angle_B, angle_Z, 
                   (unsigned int)valid_mask);
    
    if (len > 0) {
        // 非阻塞发送；USB 忙时丢弃本帧，避免卡住电机任务
        (void)CDC_Transmit_HS((uint8_t*)usb_buf, (uint16_t)len);
    }
}


/**
 * @brief 完整的失能手推测试任务 (带死循环)
 * @note 直接把这个函数的内容放进你的 IndicatorTask 即可
 */
void Chassis_ManualMove_TestTask(void)
{
    // 1. 初始化后立刻失能，方便手推
    osDelay(1000); 
    Chassis_DisableAll();
    
    // 向串口/USB发个提示
    static const char ready_msg[] = "Motor Disabled. Ready for manual move.\r\n";
    CDC_Transmit_HS((uint8_t*)ready_msg, (uint16_t)(sizeof(ready_msg) - 1U));
    
    for(;;)
    {
        // 调用单次读取打印函数
        Chassis_ReportRealPos();
        
        // 控制刷新率，10Hz 左右，既能看清手推轨迹，又不会刷屏太快
        osDelay(100); 
    }
}


#endif
