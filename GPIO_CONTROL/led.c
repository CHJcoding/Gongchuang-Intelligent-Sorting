#include "led.h"
#include "FreeRTOS.h"
#include "task.h"
#include <string.h>

#define WS2812_RESET_BYTES   100U
#define WS2812_REFRESH_MS    20U
#define WS2812_SPI_TIMEOUT   50U   // 384字节传输耗时，超时给大一点

typedef enum {
    WS_MODE_OFF = 0,
    WS_MODE_SOLID,
    WS_MODE_BLINK,
    WS_MODE_BREATHE,
    WS_MODE_FADE,
} ws_mode_t;

/* 将状态打包成结构体，每个灯环独立 */
typedef struct {
    ws_mode_t mode;
    uint8_t r, g, b, brightness;
    uint8_t cur_r, cur_g, cur_b, cur_br;
    uint32_t phase_start;
    uint32_t last_write_ms;
    uint16_t on_ms, off_ms;
    uint8_t last_on;
    uint32_t period_ms;
    uint32_t fade_start;
    uint16_t fade_ms;
    uint8_t fade_from_r, fade_from_g, fade_from_b, fade_from_br;
    uint8_t fade_to_r, fade_to_g, fade_to_b, fade_to_br;
} ws_led_state_t;

/* 定义两个灯环的状态数组（索引 0=灯环1，1=灯环2） */
static ws_led_state_t s_leds[2];

static const uint8_t s_reset_buf[WS2812_RESET_BYTES] = {0};

/* ------------------------- 底层发送 ------------------------- */
static void WS2812_Ctrl(uint8_t id, uint8_t r, uint8_t g, uint8_t b)
{
    // 【关键】必须加 static！否则 384 字节会直接撑爆 FreeRTOS 任务栈导致死机
    static uint8_t txbuf[WS2812_NUM_BYTES]; 
    
    uint8_t i, led_idx;
    uint8_t low_code, high_code;
    SPI_HandleTypeDef *hspi;

    if (id == 1) {
        hspi = &WS2812_SPI_1;
        low_code = WS2812_LOW_6M;
        high_code = WS2812_HIGH_6M;
    } else if (id == 2) {
        hspi = &WS2812_SPI_2;
        low_code = WS2812_LOW_7M5;
        high_code = WS2812_HIGH_7M5;
    } else {
        return; // 无效 ID
    }

    // 循环 16 次，填满 16 个灯珠的数据
    for (led_idx = 0U; led_idx < WS2812_LED_COUNT; led_idx++) {
        for (i = 0U; i < 8U; i++) {
            uint16_t offset = led_idx * 24U; // 每颗灯珠占 24 字节
            
            // WS2812 物理顺序是 G R B
            txbuf[offset + 7U - i]  = ((g >> i) & 1U) ? high_code : low_code;
            txbuf[offset + 15U - i] = ((r >> i) & 1U) ? high_code : low_code;
            txbuf[offset + 23U - i] = ((b >> i) & 1U) ? high_code : low_code;
        }
    }

    // 关中断发送，防止波形被打断（注意：384字节约耗时0.5ms，可能会影响CAN通信）
    taskENTER_CRITICAL(); 
    
    (void)HAL_SPI_Transmit(hspi, txbuf, WS2812_NUM_BYTES, WS2812_SPI_TIMEOUT);
    (void)HAL_SPI_Transmit(hspi, (uint8_t *)s_reset_buf, WS2812_RESET_BYTES, WS2812_SPI_TIMEOUT);
    
    taskEXIT_CRITICAL();
}

/* ------------------------- 渲染 ------------------------- */
static uint8_t ws_scale(uint8_t v, uint8_t br)
{
    return (uint8_t)(((uint16_t)v * br + 127U) / 255U);
}

static void ws_render(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t br)
{
    uint8_t idx = id - 1;
    s_leds[idx].cur_r  = r;
    s_leds[idx].cur_g  = g;
    s_leds[idx].cur_b  = b;
    s_leds[idx].cur_br = br;
    
    WS2812_Ctrl(id, ws_scale(r, br), ws_scale(g, br), ws_scale(b, br));
    s_leds[idx].last_write_ms = HAL_GetTick();
}

/* ------------------------- 初始化 ------------------------- */
void WS2812_Init(void)
{
    for (int i = 0; i < 2; i++) {
        s_leds[i].mode = WS_MODE_OFF;
        s_leds[i].r = s_leds[i].g = s_leds[i].b = 0;
        s_leds[i].brightness = 255;
        s_leds[i].cur_r = s_leds[i].cur_g = s_leds[i].cur_b = s_leds[i].cur_br = 0;
    }
    WS2812_Off(1);
    WS2812_Off(2);
}

/* ------------------------- 模式设置 ------------------------- */
void WS2812_Off(uint8_t id)
{
    if (id < 1 || id > 2) return;
    uint8_t idx = id - 1;
    s_leds[idx].mode = WS_MODE_OFF;
    WS2812_Ctrl(id, 0, 0, 0);
    s_leds[idx].cur_r = s_leds[idx].cur_g = s_leds[idx].cur_b = s_leds[idx].cur_br = 0;
}

void WS2812_SetColor(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness)
{
    if (id < 1 || id > 2) return;
    uint8_t idx = id - 1;
    s_leds[idx].mode = WS_MODE_SOLID;
    s_leds[idx].r = r; s_leds[idx].g = g; s_leds[idx].b = b;
    s_leds[idx].brightness = brightness;
    ws_render(id, r, g, b, brightness);
}

/* 兼容旧代码：两灯环同时变同一颜色 */
void WS2812_SetColorAll(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness)
{
    WS2812_SetColor(1, r, g, b, brightness);
    WS2812_SetColor(2, r, g, b, brightness);
}

void WS2812_SetBrightness(uint8_t id, uint8_t brightness)
{
    if (id < 1 || id > 2) return;
    uint8_t idx = id - 1;
    s_leds[idx].brightness = brightness;
    if (s_leds[idx].mode == WS_MODE_SOLID || s_leds[idx].mode == WS_MODE_BLINK) {
        ws_render(id, s_leds[idx].r, s_leds[idx].g, s_leds[idx].b, 
                  (s_leds[idx].mode == WS_MODE_BLINK && s_leds[idx].last_on == 0U) ? 0U : brightness);
    }
}

void WS2812_Blink(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                  uint16_t on_ms, uint16_t off_ms)
{
    if (id < 1 || id > 2) return;
    if (on_ms == 0U) on_ms = 1U;
    if (off_ms == 0U) off_ms = 1U;

    uint8_t idx = id - 1;
    s_leds[idx].mode = WS_MODE_BLINK;
    s_leds[idx].r = r; s_leds[idx].g = g; s_leds[idx].b = b;
    s_leds[idx].brightness = brightness;
    s_leds[idx].on_ms = on_ms;
    s_leds[idx].off_ms = off_ms;
    s_leds[idx].phase_start = HAL_GetTick();
    s_leds[idx].last_on = 2U;
    ws_render(id, r, g, b, brightness);
}

void WS2812_Breathe(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                    uint16_t period_ms)
{
    if (id < 1 || id > 2) return;
    if (period_ms < 100U) period_ms = 100U;

    uint8_t idx = id - 1;
    s_leds[idx].mode = WS_MODE_BREATHE;
    s_leds[idx].r = r; s_leds[idx].g = g; s_leds[idx].b = b;
    s_leds[idx].brightness = brightness;
    s_leds[idx].period_ms = period_ms;
    s_leds[idx].phase_start = HAL_GetTick();
    ws_render(id, r, g, b, 0U);
}

void WS2812_FadeTo(uint8_t id, uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                   uint16_t duration_ms)
{
    if (id < 1 || id > 2) return;
    uint8_t idx = id - 1;
    
    if (duration_ms == 0U) {
        WS2812_SetColor(id, r, g, b, brightness);
        return;
    }

    s_leds[idx].fade_from_r = s_leds[idx].cur_r;
    s_leds[idx].fade_from_g = s_leds[idx].cur_g;
    s_leds[idx].fade_from_b = s_leds[idx].cur_b;
    s_leds[idx].fade_from_br = s_leds[idx].cur_br;

    s_leds[idx].fade_to_r = r;
    s_leds[idx].fade_to_g = g;
    s_leds[idx].fade_to_b = b;
    s_leds[idx].fade_to_br = brightness;

    s_leds[idx].fade_ms = duration_ms;
    s_leds[idx].fade_start = HAL_GetTick();
    s_leds[idx].mode = WS_MODE_FADE;
    s_leds[idx].last_write_ms = HAL_GetTick() - WS2812_REFRESH_MS;
}

/* ------------------------- 周期更新 ------------------------- */
void WS2812_Update(void)
{
    uint32_t now = HAL_GetTick();

    for (uint8_t i = 0; i < 2; i++) {
        uint8_t id = i + 1;
        ws_led_state_t *led = &s_leds[i];

        switch (led->mode) {
        case WS_MODE_OFF:
        case WS_MODE_SOLID:
            break;

        case WS_MODE_BLINK: {
            uint32_t period = (uint32_t)led->on_ms + (uint32_t)led->off_ms;
            uint32_t pos    = (now - led->phase_start) % period;
            uint8_t  on     = (pos < led->on_ms) ? 1U : 0U;

            if (on != led->last_on) {
                led->last_on = on;
                ws_render(id, led->r, led->g, led->b, on ? led->brightness : 0U);
            }
            break;
        }

        case WS_MODE_BREATHE: {
            uint32_t elapsed = (now - led->phase_start) % led->period_ms;
            uint32_t half = led->period_ms / 2U;
            uint8_t  level;

            if (elapsed < half) {
                level = (uint8_t)((elapsed * 255U) / half);
            } else {
                level = (uint8_t)(((led->period_ms - elapsed) * 255U) / half);
            }

            if ((now - led->last_write_ms) >= WS2812_REFRESH_MS) {
                uint8_t br = (uint8_t)(((uint16_t)level * led->brightness + 127U) / 255U);
                ws_render(id, led->r, led->g, led->b, br);
            }
            break;
        }

        case WS_MODE_FADE: {
            uint32_t elapsed = now - led->fade_start;

            if (elapsed >= led->fade_ms) {
                led->r = led->fade_to_r;
                led->g = led->fade_to_g;
                led->b = led->fade_to_b;
                led->brightness = led->fade_to_br;
                led->mode = WS_MODE_SOLID;
                ws_render(id, led->r, led->g, led->b, led->brightness);
                break;
            }

            if ((now - led->last_write_ms) >= WS2812_REFRESH_MS) {
                int32_t t = (int32_t)elapsed;
                int32_t d = (int32_t)led->fade_ms;

                uint8_t r = (uint8_t)((int32_t)led->fade_from_r + ((int32_t)led->fade_to_r - (int32_t)led->fade_from_r) * t / d);
                uint8_t g = (uint8_t)((int32_t)led->fade_from_g + ((int32_t)led->fade_to_g - (int32_t)led->fade_from_g) * t / d);
                uint8_t b = (uint8_t)((int32_t)led->fade_from_b + ((int32_t)led->fade_to_b - (int32_t)led->fade_from_b) * t / d);
                uint8_t br = (uint8_t)((int32_t)led->fade_from_br + ((int32_t)led->fade_to_br - (int32_t)led->fade_from_br) * t / d);

                ws_render(id, r, g, b, br);
            }
            break;
        }

        default:
            break;
        }
    }
}
