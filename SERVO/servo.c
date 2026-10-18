#include "servo.h"

static const struct { uint8_t addr; uint16_t safe_pulse; } s_cfg[SERVO_COUNT] = {
    { SERVO_ADDR1, SERVO1_SAFE_PULSE },
    { SERVO_ADDR2, SERVO2_SAFE_PULSE },
    { SERVO_ADDR3, SERVO3_SAFE_PULSE },
    { SERVO_ADDR4, SERVO4_SAFE_PULSE },
};


void Servo_SetPulse(uint8_t id, uint16_t us)
{

    switch (id) {
        case 1: __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, us); break;
        case 2: __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, us); break;
        case 3: __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, us); break;
        case 4: __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, us); break;
        default: break;
    }
}

void Servo_Init(void)
{
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);       //1ºÅ¶æ»ú
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);       //2ºÅ¶æ»ú
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);       //3ºÅ¶æ»ú
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);       //4ºÅ¶æ»ú
    
    for (int i = 0; i < SERVO_COUNT; i++){
        Servo_SetPulse(s_cfg[i].addr, s_cfg[i].safe_pulse); 
    }
}
