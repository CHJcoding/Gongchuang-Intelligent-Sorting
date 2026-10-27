#include "battery_driver.h"

// 引用 CubeMX 自动生成的 ADC 句柄
extern ADC_HandleTypeDef hadc1;

// 定义静态全局变量，专门用来存 DMA 搬过来的原始 ADC 数据
// 注意：由于是单通道，只需要一个 uint16_t 变量
uint16_t battery_adc_raw = 0;

// 全局变量，用来记录中断进了多少次
volatile uint32_t tim13_irq_count = 0; 

// 【修改2】在这里定义全局电压变量
volatile float g_battery_voltage = 0.0f; 

/**
  * @brief  电池检测模块初始化
  * @note   必须在 main 函数中 MX_ADC1_Init() 之后调用
  */
void Battery_Init(void)
{
    // 1. H7 芯片必须进行 ADC 校准，否则精度极差
    HAL_ADCEx_Calibration_Start(&hadc1, ADC_CALIB_OFFSET_LINEARITY, ADC_SINGLE_ENDED);
    HAL_ADCEx_Calibration_Start(&hadc1, ADC_CALIB_OFFSET, ADC_SINGLE_ENDED);
    
    // 2. 启动 ADC 的 DMA 连续采集，长度为 1
    HAL_ADC_Start_DMA(&hadc1, (uint32_t *)&battery_adc_raw, 1);
}

/**
  * @brief  获取电池输入电压
  * @retval 真实电压值（单位：V，如 12.35）
  */
float Battery_GetVoltage(void)
{
    // 计算公式： (ADC原始值 / 16位满量程65535) * 3.3V参考电压 * 11倍分压比
    // (达妙板子 R86=100k, R87=10k，分压比为 11)
    float voltage = ((float)battery_adc_raw / 65535.0f) * 3.3f * 11.0f;

    return voltage;
}
