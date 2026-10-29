#include "BUTTON.h"

/**
 * @brief 按键处理函数（放在任务循环中，非阻塞轮询）
 */
void Button_Handler(void)
{
    static uint8_t last_state = 1;      // 上一次状态 (1为松开)
    static uint32_t press_time = 0;     // 按下的时间戳
    uint8_t current_state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_15); // 读取 PA15

    // 1. 检测到下降沿（按下）
    if (current_state == 0 && last_state == 1) 
    {
        press_time = osKernelGetTickCount(); // 记录按下时间
    }
    // 2. 检测到上升沿（松开）
    else if (current_state == 1 && last_state == 0) 
    {
        uint32_t duration = osKernelGetTickCount() - press_time;
        
        // 软件消抖：按下时间大于 50ms 且小于 2000ms 才算有效按键
        if (duration > 50U && duration < 2000U) 
        {
            /* ========== 执行动作 ========== */
            
            // 动作1：发送 USB "OK"
            uint8_t ack_msg[] = "OK\r\n"; 
            // 简单判断一下 USB 是否忙碌，避免卡死
            if (CDC_Transmit_HS(ack_msg, sizeof(ack_msg)-1) == USBD_BUSY) {
                // 如果忙，可以考虑用个标志位稍后发，或者直接丢弃
            }

            // 动作2：灯换绿灯
            // 根据你的底层驱动调用。假设 WS2812_SetColor(红, 绿, 蓝)
            // 亮度给个适中值，比如 100
            WS2812_SetColor(0, 255, 0, 100); 
            
            /* ============================== */
        }
    }
    
    last_state = current_state; // 更新状态
}
