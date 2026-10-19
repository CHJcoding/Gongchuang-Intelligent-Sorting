#include "servo.h"

static const struct { uint8_t addr; uint16_t safe_pulse; } s_cfg[SERVO_COUNT] = {
    { SERVO_ADDR1, SERVO1_SAFE_PULSE },
    { SERVO_ADDR2, SERVO2_SAFE_PULSE },
    { VALVE_SERVO3_ID, SERVO3_SAFE_PULSE },
    { PUMP_SERVO4_ID, SERVO4_SAFE_PULSE },
};

void Servo_SetPulse(uint8_t id, uint16_t us)
{
    #if LIMIT_PULSE == 0
    if (us < SERVO_PULSE_MIN)
    {
        us = SERVO_PULSE_MIN;
    }
    else if (us > SERVO_PULSE_MAX)
    {
        us = SERVO_PULSE_MAX;
    }
    #endif

    switch (id)
    {
        case SERVO_ADDR1:
            __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, us);
            break;
        case SERVO_ADDR2:
            __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, us);
            break;
        case VALVE_SERVO3_ID:
            __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_3, us);
            break;
        case PUMP_SERVO4_ID:
            __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, us);
            break;
        default:
            break;
    }
}

void Servo_Init(void)
{
    /* Load safe positions before enabling the outputs. */
    for (uint8_t i = 0U; i < SERVO_COUNT; i++)
    {
        Servo_SetPulse(s_cfg[i].addr, s_cfg[i].safe_pulse);
    }

    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
}


/* 气泵部分 */
void Pneumatic_ValveOpen(void)
{
    Servo_SetPulse(VALVE_SERVO3_ID, PNEUMATIC_ON_PULSE_US);
}

void Pneumatic_ValveClose(void)
{
    Servo_SetPulse(VALVE_SERVO3_ID, PNEUMATIC_OFF_PULSE_US);
}

void Pneumatic_PumpOn(void)
{
    Servo_SetPulse(PUMP_SERVO4_ID, PNEUMATIC_ON_PULSE_US);
}

void Pneumatic_PumpOff(void)
{
    Servo_SetPulse(PUMP_SERVO4_ID, PNEUMATIC_OFF_PULSE_US);
}

//开启气泵
void Pneumatic_On(void)
{
    Pneumatic_ValveClose();
    Pneumatic_PumpOn();
}

//关闭气泵
void Pneumatic_Off(void)
{
    Pneumatic_PumpOff();
    Pneumatic_ValveOpen();
}

//整体关闭气泵
void Pneumatic_AllOff(void)
{
    Pneumatic_ValveClose();
    Pneumatic_PumpOff();
}
