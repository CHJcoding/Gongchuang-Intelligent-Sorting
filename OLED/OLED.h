#ifndef __OLED_H__
#define __OLED_H__

#include "ssd1306.h"
#include "ssd1306_fonts.h"
#include <stdio.h>
#include "cmsis_os2.h"
#include "battery_driver.h"

void OLED_Init(void);
void OLED_UpdataScreen(void);
    
#endif
