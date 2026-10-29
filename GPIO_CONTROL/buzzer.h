#ifndef __BUZZER_H__
#define __BUZZER_H__

#include "tim.h"
#include "cmsis_os2.h"

void buzzer_Init(void);
void buzzer_Open(void);
void buzzer_Close(void);
void buzzer_PWM(uint8_t high, uint8_t low);

void buzzer_finish_flag(int j);
#endif
