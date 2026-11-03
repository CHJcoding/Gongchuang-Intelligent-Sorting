#ifndef __BUZZER_H__
#define __BUZZER_H__

#include "tim.h"
#include "cmsis_os2.h"

void buzzer_Init(void);
void buzzer_Open(void);
void buzzer_Close(void);
void buzzer_PWM(uint8_t high, uint8_t low);

void buzzer_finish_flag(int j);


//这个不用
void Buzzer_SetFreq(uint32_t freq);
void Buzzer_PlayMelody(const uint16_t *notes, const uint16_t *durs,
                       uint16_t len, uint8_t loop);

//选歌
void Buzzer_ChooseSong(uint8_t i);

#endif
