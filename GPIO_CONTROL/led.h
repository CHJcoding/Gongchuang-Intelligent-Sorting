#ifndef LED_H
#define LED_H

#include "main.h"
#include "spi.h"
#include <stdint.h>

/* 两个 SPI 句柄（确认你 CubeMX 里生成了这两个） */
#define WS2812_SPI_1        hspi6    // 灯环1 (6.0MHz)
#define WS2812_SPI_2        hspi3    // 灯环2 (7.5MHz)

extern SPI_HandleTypeDef WS2812_SPI_1;
extern SPI_HandleTypeDef WS2812_SPI_2;

/* 速度差异导致的两套时序 */
#define WS2812_LOW_6M    0x60U
#define WS2812_HIGH_6M   0x78U

#define WS2812_LOW_7M5   0xC0U
#define WS2812_HIGH_7M5  0xF8U

/* 每个灯环有 16 颗灯珠，每颗灯珠需要 24 个字节 */
#define WS2812_LED_COUNT     16U
#define WS2812_NUM_BYTES     (WS2812_LED_COUNT * 24U)  // 384 字节

/* 初始化 */
void WS2812_Init(void);

/* 核心操作 API（参数 id: 1 代表灯环1，2 代表灯环2） */
void WS2812_SetColor(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness);
void WS2812_SetBrightness(uint8_t id, uint8_t brightness);
void WS2812_Blink(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness, uint16_t on_ms, uint16_t off_ms);
void WS2812_Breathe(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness, uint16_t period_ms);
void WS2812_FadeTo(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness, uint16_t duration_ms);
void WS2812_Off(uint8_t id);

/* 周期更新（状态机核心，放在 FreeRTOS 任务里每 10~20ms 调用一次） */
void WS2812_Update(void);

/* 保留兼容旧代码：同时设置两个灯环同色 */
void WS2812_SetColorAll(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness);

#endif