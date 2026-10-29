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

//用于测试和中间放置耦合 freertos若调用 是为了测试用
HAL_StatusTypeDef Chassis_MoveTo(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc);
HAL_StatusTypeDef Z_MoveTo(float z_mm, uint16_t vel_rpm, uint8_t acc);

//拥有运动+延迟的功能！实际调用这个
void Chassis_MoveToAndWait(float x_mm, float y_mm, uint16_t vel_rpm, uint8_t acc);
void Z_MoveToAndWait(float z_mm, uint16_t vel_rpm, uint8_t acc);

//垃圾桶旋转
void Abs_Pluse2Angle(float angle);
HAL_StatusTypeDef Spin_position(int position);

#if Handeye_calibration
/* 查询 A/B 实时位置，并通过 USB CDC 发送一行 ASCII 数据 */
/* 获取真实的物理坐标 (基于编码器反馈) */
void Chassis_GetRealPosition(float *x_mm, float *y_mm);
void Chassis_ReportRealPos(void);
void Chassis_ManualMove_TestTask(void);
#endif

#endif
