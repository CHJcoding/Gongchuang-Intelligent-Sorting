#include "motor.h"

#if Handeye_calibration
/* 存储驱动器反馈的真实脉冲数 */
volatile int32_t g_real_pulse_A = 0;
volatile int32_t g_real_pulse_B = 0;
volatile uint8_t g_real_position_valid_A = 0U;
volatile uint8_t g_real_position_valid_B = 0U;

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

/* H7 端维护的当前绝对坐标（mm） */
static float s_cur_x_mm = 0.0f;
static float s_cur_y_mm = 0.0f;


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
    }
}
#endif


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
//static uint16_t Chassis_SpeedToRpm(float mm_s)
//{
//    float rpm = fabsf(mm_s) * 60.0f / MOTOR_LEAD_MM;
//    
//    if(rpm > (float)MOTOR_MAX_RPM){
//        rpm = (float)MOTOR_MAX_RPM;
//    }
//    return (uint16_t)(rpm + 0.5f);  
//}

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
    
    /* 关键：软件坐标也清零 */
    s_cur_x_mm = 0.0f;
    s_cur_y_mm = 0.0f;
    
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


#if Handeye_calibration


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
    float real_x = 0.0f, real_y = 0.0f;
    float angle_A = 0.0f, angle_B = 0.0f;
    uint8_t valid_mask;
    static char usb_buf[128];
    int len = 0;
    
    g_real_position_valid_A = 0U;
    g_real_position_valid_B = 0U;
    // 1. 向 A/B 电机发送读取实时位置 (S_CPOS) 的 CAN 指令
    Emm_V5_Read_Sys_Params(A_MOTOR_ADDR, S_CPOS);
    Emm_V5_Read_Sys_Params(B_MOTOR_ADDR, S_CPOS);
    
    // 2. 给 CAN 总线时间回传数据 (根据波特率给 10~20ms)
    osDelay(20U);
    
    // 3. 计算真实坐标
    Chassis_GetRealPosition(&real_x, &real_y);
    angle_A = (float)g_real_pulse_A * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    angle_B = (float)g_real_pulse_B * 360.0f / MOTOR_POSITION_UNITS_PER_REV;
    valid_mask = (uint8_t)((g_real_position_valid_A != 0U ? 1U : 0U) |
                          (g_real_position_valid_B != 0U ? 2U : 0U));
    
    // 4. 通过 USB 发送出来
    len = snprintf(usb_buf, sizeof(usb_buf),
                   "POS A:%d B:%d X:%.2f Y:%.2f Adeg:%.2f Bdeg:%.2f valid:%u\r\n",
                   -(int)g_real_pulse_A, -(int)g_real_pulse_B,
                   real_x, real_y, angle_A, angle_B, (unsigned int)valid_mask);
    
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
