#include "buzzer.h"

void buzzer_Init(void)
{
    HAL_TIM_PWM_Start(&htim12, TIM_CHANNEL_2);
    TIM12->CCR2 = 0;               /* 上电先静音 */
}

void buzzer_Open(void)
{
    TIM12->CCR2 = 100;
}

void buzzer_Close(void)
{
    TIM12->CCR2 = 0;
}

//慢慢变响慢慢变弱
void buzzer_PWM(uint8_t high, uint8_t low)
{
    for(int i = low; i < high; i++)
      {
        TIM12->CCR2 = i;
          HAL_Delay(10);
       }
      for(int j = high; j > low; j--)
      {
        TIM12->CCR2 = j;
          HAL_Delay(10);
       }
}


