#ifndef __BATTERY_DRIVER_H
#define __BATTERY_DRIVER_H

#include "main.h"

// 外部可以随时读取这个变量来获取最新电压
extern uint16_t battery_adc_raw;   // 新增：暴露给外部使用
extern volatile float g_battery_voltage; 

// 声明初始化函数（用于启动 ADC 和 DMA）
void Battery_Init(void);

// 声明获取电压函数（返回值单位：伏特 V）
float Battery_GetVoltage(void);

#endif
