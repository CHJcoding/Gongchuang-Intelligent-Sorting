#include "usart.h"   /* CubeMX生成的UART10句柄 */
#include <stdio.h>
#include <string.h>

/**
 * @brief 发送文本到陶晶驰串口屏的文本控件
 * @param obj  控件名称，例如 "t0"
 * @param txt  要显示的字符串
 */
void screen_send_text(const char *obj, const char *txt)
{
    char cmd[32];

    /* 指令格式：t0.txt="内容" + 0xFF 0xFF 0xFF */
    int len = snprintf(cmd, sizeof(cmd), "%s.txt=\"%s\"", obj, txt);

    /* 发送指令部分 */
    HAL_UART_Transmit(&huart10, (uint8_t*)cmd, len, 100);

    /* 发送结束符 0xFF 0xFF 0xFF */
    uint8_t tail[3] = {0xFF, 0xFF, 0xFF};
    HAL_UART_Transmit(&huart10, tail, 3, 100);
}

/**
 * @brief 切换到指定画面
 * @param page_name  画面名称，例如 "page0"、"main"、"scanner"
 */
void screen_change_page(const char *page_name)
{
    char cmd[32];

    /* 指令格式：page 画面名 + 0xFF 0xFF 0xFF */
    int len = snprintf(cmd, sizeof(cmd), "page %s", page_name);

    HAL_UART_Transmit(&huart10, (uint8_t*)cmd, len, 100);

    uint8_t tail[3] = {0xFF, 0xFF, 0xFF};
    HAL_UART_Transmit(&huart10, tail, 3, 100);
}
