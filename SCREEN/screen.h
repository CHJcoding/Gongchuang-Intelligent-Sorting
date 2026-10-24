#ifndef __SCREEN_H__
#define __SCREEN_H__

#include "cmsis_os2.h"
#include "usart.h"   /* CubeMX生成的UART10句柄 */
#include "usb_processing.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* 发送控件 */
void screen_send_text(const char *obj, const char *txt);
void screen_change_page(const char *page_name);

void screen_update_row(uint8_t row, const char *name, uint32_t pic_id, int32_t count);
//下面是主要函数
void screen_update_vision(uint8_t target_row, int32_t count);
#endif
