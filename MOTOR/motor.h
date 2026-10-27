#ifndef __MOTOR_H__
#define __MOTOR_H__

#include "main.h"
#include "motor_config.h"
#include "Emm_V5.h"
#include <string.h>
#include "usbd_cdc_if.h"



/* 记得后期给函数重命名 chassis为底盘 */
HAL_StatusTypeDef Chassis_EnableAll(void);
HAL_StatusTypeDef Chassis_DisableAll(void);
HAL_StatusTypeDef Chassis_Stop(void);
HAL_StatusTypeDef Chassis_SetZeroPositions(void);
HAL_StatusTypeDef Chassis_Init(void);
static uint32_t Chassis_DistanceToPulses(uint32_t distance_mm);

HAL_StatusTypeDef Chassis_MoveTo(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc);

#if Handeye_calibration
/* 查询 A/B 实时位置，并通过 USB CDC 发送一行 ASCII 数据 */
/* 获取真实的物理坐标 (基于编码器反馈) */
void Chassis_GetRealPosition(float *x_mm, float *y_mm);
void Chassis_ReportRealPos(void);
void Chassis_ManualMove_TestTask(void);
#endif

#endif
