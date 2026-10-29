#ifndef __SERVO_H__
#define __SERVO_H__

#include "tim.h"
#include "servo_config.h"
#include "cmsis_os2.h"
#include "motor.h"

void Servo_Init(void);
void Servo_SetPulse(uint8_t id, uint16_t us);

//À¬»øÍ°Ðý×ª£¨postition - ÌîÎ»ÖÃ£©
HAL_StatusTypeDef Spin_servo_position(int position);

void Pneumatic_On(void);
void Pneumatic_Off(void);

/* Close the valve and stop the pump. */
void Pneumatic_AllOff(void);

#endif
