#include "sensor.h"

#define SENSOR_DROP_PORT  GPIOD
#define SENSOR_DROP_PIN   GPIO_PIN_2

/**
 * @brief 检测物料是否落下（低电平有效，瞬时状态）
 */
uint8_t Sensor_IsMaterialDropped(void)
{
    if (HAL_GPIO_ReadPin(SENSOR_DROP_PORT, SENSOR_DROP_PIN) == GPIO_PIN_RESET) {
        return 1;
    }
    return 0;
}

/**
 * @brief 阻塞等待物料掉落（带防重复触发和超时）
 * @param timeout_ms 最长等待时间（毫秒）
 * @return 1: 成功检测到掉落  0: 超时未检测到
 */
uint8_t Sensor_WaitForDrop(uint32_t timeout_ms)
{
    uint32_t start_time = HAL_GetTick();
    uint8_t last_state = HAL_GPIO_ReadPin(SENSOR_DROP_PORT, SENSOR_DROP_PIN); // 记录初始状态

    // 在超时时间内不断轮询
    while ((HAL_GetTick() - start_time) < timeout_ms) 
    {
        uint8_t current_state = HAL_GPIO_ReadPin(SENSOR_DROP_PORT, SENSOR_DROP_PIN);
        
        // 【核心逻辑】检测下降沿：上一次是高电平(未遮挡)，这一次是低电平(遮挡)
        if (last_state == GPIO_PIN_SET && current_state == GPIO_PIN_RESET) 
        {
            return 1; // 捕捉到掉落瞬间，立刻返回成功
        }
        
        // 更新上一次状态
        last_state = current_state;
        
        // 每次查询间隔 20ms，让出 CPU 给其他任务（比如屏幕刷新和电机控制）
        osDelay(20); 
    }
    
    return 0; // 超时了都没检测到
}

