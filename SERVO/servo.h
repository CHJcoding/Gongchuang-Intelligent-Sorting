#ifndef __SERVO_H__
#define __SERVO_H__

#include "tim.h"
#include "servo_config.h"

void Servo_Init(void);
void Servo_SetPulse(uint8_t id, uint16_t us);

#endif
