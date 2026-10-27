#include "OLED.h"

void OLED_Init()
{
    char display_buf[32];
    
    // 1. 初始化 OLED (使用默认的 I2C1 和引脚)
    ssd1306_Init();
    
    // 2. 开机显示测试
    ssd1306_Fill(Black);
    
    //显示字体
    ssd1306_SetCursor(20, 25);
    ssd1306_WriteString("Init OK", Font_11x18, White); // 稍微用大字体显示一下
    
    ssd1306_UpdateScreen();
    
    osDelay(1000); // 等待 1 秒展示开机画面

}

void OLED_UpdataScreen()
{
    char display_buf[32]; // 【修复】在内部定义局部变量
    uint32_t current_tick = HAL_GetTick(); // 【修复】实时获取当前时间

    // 【修复】刷新前必须清屏，否则会有数字残影
    ssd1306_Fill(Black); 

    // ================= 第一行：标题 =================
    ssd1306_SetCursor(0, 0);
    ssd1306_WriteString("BATTERY MONITOR", Font_6x8, White);
    
    // ================= 第二行：核心电压 =================
    // g_battery_voltage 需要在 battery_driver.h 中声明为 extern volatile float
    snprintf(display_buf, sizeof(display_buf), "Voltage: %.2f V", g_battery_voltage);
    ssd1306_SetCursor(0, 16);
    ssd1306_WriteString(display_buf, Font_7x10, White);
  
    // ================= 第三行：系统时间 =================
    snprintf(display_buf, sizeof(display_buf), "Uptime : %lu s", current_tick / 1000);
    ssd1306_SetCursor(0, 32);
    ssd1306_WriteString(display_buf, Font_6x8, White);
    
    // ================= 第四行：状态栏 =================
    ssd1306_SetCursor(0, 48);
    ssd1306_WriteString("Status : RUNNING", Font_6x8, White);
    
    // 将显存数据推送到屏幕
    ssd1306_UpdateScreen();
}
    
