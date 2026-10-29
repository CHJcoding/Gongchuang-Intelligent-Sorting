#include "BUTTON.h"
#include "usbd_cdc_if.h"
#include "led.h"
#include "cmsis_os2.h"

void Button_Handler(void)
{
    // 静态变量：只初始化一次，其他任何地方都碰不到它们
    static uint8_t last_state = 1;       // 假设默认高电平（松开）
    static uint32_t last_trigger_time = 0; // 上次触发时间
    
    // 读取当前 PA15 电平（按下为 0，松开为 1）
    uint8_t current_state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_15);
    uint32_t now = osKernelGetTickCount();

    // 核心逻辑：检测下降沿（上次是1松开，这次是0按下）
    if (last_state == 1 && current_state == 0) 
    {
        // 防抖拦截：距离上次触发必须大于 300ms
        if ((now - last_trigger_time) > 300) 
        {
            /* ========== 执行你的动作（只执行一次） ========== */
            uint8_t ack_msg[] = "OK\r\n"; 
            CDC_Transmit_HS(ack_msg, sizeof(ack_msg)-1);

            WS2812_SetColor(1, 0, 255, 0, 100); // 变绿灯
            /* ============================================== */
               
            // 记录本次触发时间
            last_trigger_time = now; 
        }
    }
    
    // 【极其关键】记录上一次的状态，防止长按一直触发
    last_state = current_state; 
}
