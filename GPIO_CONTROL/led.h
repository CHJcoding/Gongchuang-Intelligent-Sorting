#ifndef LED_H
#define LED_H

#include "main.h"
#include "spi.h"
#include <stdint.h>

/* 你的 SPI 句柄，按实际修改 */
#define WS2812_SPI_UNIT     hspi6
extern SPI_HandleTypeDef WS2812_SPI_UNIT;

/* WS2812 码元（6MHz SPI 下：0x60=比特0，0x78=比特1） */
#define WS2812_LOW    0x60U     // 0码
#define WS2812_HIGH   0x78U     // 1码

void WS2812_Init(void);

/* 底层：立即写颜色，满亮度。保留你原来的名字 */
void WS2812_Ctrl(uint8_t r, uint8_t g, uint8_t b);

/* 固定颜色常亮。brightness: 0(灭) ~ 255(最亮) */
void WS2812_SetColor(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness);

/* 只改亮度，保持当前颜色 */
void WS2812_SetBrightness(uint8_t brightness);

/* 闪烁：亮 on_ms，灭 off_ms，无限循环 */
void WS2812_Blink(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                  uint16_t on_ms, uint16_t off_ms);

/* 呼吸：亮度 0→brightness→0，一个完整周期 period_ms */
void WS2812_Breathe(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                    uint16_t period_ms);

/* 渐变：从当前颜色过渡到目标颜色，历时 duration_ms，结束后停在目标色 */
void WS2812_FadeTo(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                   uint16_t duration_ms);

/* 常闭 */
void WS2812_Off(void);

/* 周期调用，建议 osDelay(10) 或 (20) */
void WS2812_Update(void);

#endif
