#include "usart.h"   /* CubeMX生成的UART10句柄 */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/**
 * @brief 发送文本到陶晶驰串口屏的文本控件
 * @param obj  控件名称，例如 "t0"
 * @param txt  要显示的字符串
 */
 //screen_send_text("t0", "三角形"); 让t0显示"三角形"
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
 * @brief 向文本控件发送数字（整数或浮点）
 * @param obj      控件名，如 "t0" 或 "main.t0"
 * @param value    要显示的数字
 * @param decimals 小数位数：0=整数，1~6=浮点
 */
void screen_send_number(const char *obj, float value, uint8_t decimals)
{
    char cmd[64];
    int n;

    if (decimals > 6U) decimals = 6U;

    if (decimals == 0U) {
        /* 整数：四舍五入后按 long 输出 */
        long iv = (long)(value >= 0.0f ? value + 0.5f : value - 0.5f);
        n = snprintf(cmd, sizeof(cmd) - 3, "%s.txt=\"%ld\"", obj, iv);
    } else {
        /* 浮点：%.*f 动态精度 */
        n = snprintf(cmd, sizeof(cmd) - 3, "%s.txt=\"%.*f\"",
                     obj, (int)decimals, (double)value);
    }

    if (n < 0 || n >= (int)(sizeof(cmd) - 3)) {
        return;   /* 截断或出错，不发垃圾 */
    }

    cmd[n]     = (char)0xFF;
    cmd[n + 1] = (char)0xFF;
    cmd[n + 2] = (char)0xFF;

    HAL_UART_Transmit(&huart10, (uint8_t*)cmd, (uint16_t)(n + 3), 100);
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



/**
 * @brief 万能屏幕发送接口（支持字符串、整数、浮点数）
 * @param obj  控件名，如 "t0"
 * @param fmt  格式化字符串，如 "%s" (文本), "%ld" (整数), "%.1f" (浮点)
 * @param ...  要发送的变量
 */
void screen_send_fmt(const char *obj, const char *fmt, ...)
{
    char content[64];
    char cmd[128];
    va_list ap;
    int cn, n;

    // 1. 将可变参数格式化为字符串
    va_start(ap, fmt);
    cn = vsnprintf(content, sizeof(content), fmt, ap);
    va_end(ap);
    if (cn < 0 || cn >= (int)sizeof(content)) return;

    // 2. 拼接指令头
    n = snprintf(cmd, sizeof(cmd) - 3, "%s.txt=\"%s\"", obj, content);
    if (n < 0 || n >= (int)(sizeof(cmd) - 3)) return;

    // 3. 拼接结束符 0xFF 0xFF 0xFF
    cmd[n] = (char)0xFF; cmd[n+1] = (char)0xFF; cmd[n+2] = (char)0xFF;

    // 4. 一次性发送，防打断
    HAL_UART_Transmit(&huart10, (uint8_t*)cmd, (uint16_t)(n + 3), 100);
}

/**
 * @brief 一次性更新一整行（名称、图片、数量）
 * @param row       行号（1 到 6）
 * @param name      货物名称（如 "三角形"）
 * @param pic_id    图片库中的图片ID（如 0, 1, 2...）
 * @param count     分拣成功总数量
 */
void screen_update_row(uint8_t row, const char *name, uint32_t pic_id, int32_t count)
{
    if (row < 1 || row > 6) return; // 越界保护

    char cmd[256]; // 容量要大，装 3 条指令，防溢出
    int n = 0;
    int len = 0;

    // ============ 1. 更新名称 (t0 ~ t5) ============
    len = snprintf(cmd + n, sizeof(cmd) - n - 3, "t%d.txt=\"%s\"", row - 1, name);
    if (len > 0) {
        n += len;
        cmd[n++] = (char)0xFF; cmd[n++] = (char)0xFF; cmd[n++] = (char)0xFF;
    }

    // ============ 2. 更新图片 (p1 ~ p6) ============
    // 淘晶驰图片控件使用 .pic 属性
    len = snprintf(cmd + n, sizeof(cmd) - n - 3, "p%d.pic=%lu", row, (unsigned long)pic_id);
    if (len > 0) {
        n += len;
        cmd[n++] = (char)0xFF; cmd[n++] = (char)0xFF; cmd[n++] = (char)0xFF;
    }

    // ============ 3. 更新数量 (t6 ~ t11) ============
    len = snprintf(cmd + n, sizeof(cmd) - n - 3, "t%d.txt=\"%ld\"", row + 5, (long)count);
    if (len > 0) {
        n += len;
        cmd[n++] = (char)0xFF; cmd[n++] = (char)0xFF; cmd[n++] = (char)0xFF;
    }

    // ============ 4. 一次性发给屏幕 ============
    if (n > 0) {
        HAL_UART_Transmit(&huart10, (uint8_t*)cmd, (uint16_t)n, 100);
    }
}
//调用案例：
//screen_update_row(1, "三角形", 1, 0);
//分别是：第几行，什么名字，图片id，抓取数量
