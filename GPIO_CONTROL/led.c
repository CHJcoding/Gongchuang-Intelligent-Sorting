#include "led.h"

#define WS2812_NUM_BYTES     24U
#define WS2812_RESET_BYTES   100U
#define WS2812_REFRESH_MS    20U     /* 呼吸/渐变最小刷新间隔 */
#define WS2812_SPI_TIMEOUT   10U     /* ms */

typedef enum {
    WS_MODE_OFF = 0,
    WS_MODE_SOLID,
    WS_MODE_BLINK,
    WS_MODE_BREATHE,
    WS_MODE_FADE,
} ws_mode_t;

static ws_mode_t s_mode = WS_MODE_OFF;

/* 目标颜色（SetColor/Blink/Breathe 使用） */
static uint8_t s_r = 0U, s_g = 0U, s_b = 0U;
static uint8_t s_brightness = 255U;

/* 当前实际渲染出的颜色/亮度（供 FadeTo 作为起点） */
static uint8_t s_cur_r = 0U, s_cur_g = 0U, s_cur_b = 0U, s_cur_br = 0U;

/* 时间基准 */
static uint32_t s_phase_start = 0U;   /* Blink/Breathe 起点 */
static uint32_t s_last_write_ms = 0U; /* 上次真正写 SPI 的时刻 */

/* Blink 参数 */
static uint16_t s_on_ms = 500U, s_off_ms = 500U;
static uint8_t  s_last_on = 2U;       /* 2=未知，强制第一帧写入 */

/* Breathe 参数 */
static uint32_t s_period_ms = 1000U;

/* Fade 参数 */
static uint32_t s_fade_start = 0U;
static uint16_t s_fade_ms = 0U;
static uint8_t  s_fade_from_r = 0U, s_fade_from_g = 0U, s_fade_from_b = 0U;
static uint8_t  s_fade_from_br = 0U;
static uint8_t  s_fade_to_r = 0U, s_fade_to_g = 0U, s_fade_to_b = 0U;
static uint8_t  s_fade_to_br = 0U;

/* reset 帧放 flash，省 RAM */
static const uint8_t s_reset_buf[WS2812_RESET_BYTES] = {0};

/* ------------------------- 底层 ------------------------- */

void WS2812_Ctrl(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t txbuf[WS2812_NUM_BYTES];
    uint8_t i;

    /* WS2812 顺序：G R B */
    for (i = 0U; i < 8U; i++) {
        txbuf[7U - i]  = ((g >> i) & 1U) ? WS2812_HIGH : WS2812_LOW;
        txbuf[15U - i] = ((r >> i) & 1U) ? WS2812_HIGH : WS2812_LOW;
        txbuf[23U - i] = ((b >> i) & 1U) ? WS2812_HIGH : WS2812_LOW;
    }

    (void)HAL_SPI_Transmit(&WS2812_SPI_UNIT, txbuf, WS2812_NUM_BYTES, WS2812_SPI_TIMEOUT);
    /* 一次性发 100 字节 0 作为 reset（6MHz 下约 133μs，> 50μs 要求） */
    (void)HAL_SPI_Transmit(&WS2812_SPI_UNIT, (uint8_t *)s_reset_buf,
                           WS2812_RESET_BYTES, WS2812_SPI_TIMEOUT);
}

/* ------------------------- 渲染 ------------------------- */

static uint8_t ws_scale(uint8_t v, uint8_t br)
{
    return (uint8_t)(((uint16_t)v * br + 127U) / 255U);
}

static void ws_render(uint8_t r, uint8_t g, uint8_t b, uint8_t br)
{
    s_cur_r  = r;
    s_cur_g  = g;
    s_cur_b  = b;
    s_cur_br = br;
    WS2812_Ctrl(ws_scale(r, br), ws_scale(g, br), ws_scale(b, br));
    s_last_write_ms = HAL_GetTick();
}
/* ------------------------- 初始化 ------------------------- */

void WS2812_Init(void)
{
    s_mode = WS_MODE_OFF;
    s_r = s_g = s_b = 0U;
    s_brightness = 255U;
    s_cur_r = s_cur_g = s_cur_b = s_cur_br = 0U;
    WS2812_Ctrl(0U, 0U, 0U);
}

/* ------------------------- 模式设置 ------------------------- */

void WS2812_Off(void)
{
    s_mode = WS_MODE_OFF;
    WS2812_Ctrl(0U, 0U, 0U);
    s_cur_r = s_cur_g = s_cur_b = s_cur_br = 0U;
}

void WS2812_SetColor(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness)
{
    s_mode = WS_MODE_SOLID;
    s_r = r; s_g = g; s_b = b;
    s_brightness = brightness;
    ws_render(r, g, b, brightness);
}

void WS2812_SetBrightness(uint8_t brightness)
{
    s_brightness = brightness;
    if (s_mode == WS_MODE_SOLID || s_mode == WS_MODE_BLINK) {
        ws_render(s_r, s_g, s_b, (s_mode == WS_MODE_BLINK && s_last_on == 0U) ? 0U : brightness);
    }
}

//r - 红色分量，g - 绿色分量， b - 蓝色分量， brightness - 整体亮度(0-255), on_ms 亮持续多少毫秒，off_ms灭持续多少毫秒
void WS2812_Blink(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                  uint16_t on_ms, uint16_t off_ms)
{
    if (on_ms == 0U)  on_ms = 1U;
    if (off_ms == 0U) off_ms = 1U;

    s_mode = WS_MODE_BLINK;
    s_r = r; s_g = g; s_b = b;
    s_brightness = brightness;
    s_on_ms = on_ms;
    s_off_ms = off_ms;
    s_phase_start = HAL_GetTick();
    s_last_on = 2U;                 /* 强制第一帧写入 */
    ws_render(r, g, b, brightness); /* 立即亮 */
}

void WS2812_Breathe(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                    uint16_t period_ms)
{
    if (period_ms < 100U) period_ms = 100U;

    s_mode = WS_MODE_BREATHE;
    s_r = r; s_g = g; s_b = b;
    s_brightness = brightness;
    s_period_ms = period_ms;
    s_phase_start = HAL_GetTick();
    ws_render(r, g, b, 0U);          /* 从暗开始 */
}

void WS2812_FadeTo(uint8_t r, uint8_t g, uint8_t b, uint8_t brightness,
                   uint16_t duration_ms)
{
    if (duration_ms == 0U) {
        WS2812_SetColor(r, g, b, brightness);
        return;
    }

    /* 从当前实际渲染值开始，视觉上连续 */
    s_fade_from_r = s_cur_r;
    s_fade_from_g = s_cur_g;
    s_fade_from_b = s_cur_b;
    s_fade_from_br = s_cur_br;

    s_fade_to_r = r;
    s_fade_to_g = g;
    s_fade_to_b = b;
    s_fade_to_br = brightness;

    s_fade_ms = duration_ms;
    s_fade_start = HAL_GetTick();
    s_mode = WS_MODE_FADE;
    s_last_write_ms = HAL_GetTick() - WS2812_REFRESH_MS;  /* 允许立即写 */
}

/* ------------------------- 周期更新 ------------------------- */

void WS2812_Update(void)
{
    uint32_t now = HAL_GetTick();

    switch (s_mode) {
    case WS_MODE_OFF:
    case WS_MODE_SOLID:
        /* 只在设置时写过，这里不动作 */
        return;

    case WS_MODE_BLINK: {
        uint32_t period = (uint32_t)s_on_ms + (uint32_t)s_off_ms;
        uint32_t pos    = (now - s_phase_start) % period;
        uint8_t  on     = (pos < s_on_ms) ? 1U : 0U;

        if (on != s_last_on) {
            s_last_on = on;
            ws_render(s_r, s_g, s_b, on ? s_brightness : 0U);
        }
        return;
    }

    case WS_MODE_BREATHE: {
        uint32_t period = s_period_ms;
        uint32_t elapsed = (now - s_phase_start) % period;
        uint32_t half = period / 2U;
        uint8_t  level;

        if (elapsed < half) {
            level = (uint8_t)((elapsed * 255U) / half);
        } else {
            level = (uint8_t)(((period - elapsed) * 255U) / half);
        }

        if ((now - s_last_write_ms) >= WS2812_REFRESH_MS) {
            uint8_t br = (uint8_t)(((uint16_t)level * s_brightness + 127U) / 255U);
            ws_render(s_r, s_g, s_b, br);
        }
        return;
    }

    case WS_MODE_FADE: {
        uint32_t elapsed = now - s_fade_start;

        if (elapsed >= s_fade_ms) {
            /* 完成，停在目标色 */
            s_r = s_fade_to_r;
            s_g = s_fade_to_g;
            s_b = s_fade_to_b;
            s_brightness = s_fade_to_br;
            s_mode = WS_MODE_SOLID;
            ws_render(s_r, s_g, s_b, s_brightness);
            return;
        }

        if ((now - s_last_write_ms) >= WS2812_REFRESH_MS) {
            int32_t t = (int32_t)elapsed;
            int32_t d = (int32_t)s_fade_ms;

            uint8_t r = (uint8_t)((int32_t)s_fade_from_r +
                        ((int32_t)s_fade_to_r - (int32_t)s_fade_from_r) * t / d);
            uint8_t g = (uint8_t)((int32_t)s_fade_from_g +
                        ((int32_t)s_fade_to_g - (int32_t)s_fade_from_g) * t / d);
            uint8_t b = (uint8_t)((int32_t)s_fade_from_b +
                        ((int32_t)s_fade_to_b - (int32_t)s_fade_from_b) * t / d);
            uint8_t br = (uint8_t)((int32_t)s_fade_from_br +
                        ((int32_t)s_fade_to_br - (int32_t)s_fade_from_br) * t / d);

            ws_render(r, g, b, br);
        }
        return;
    }

    default:
        return;
    }
}
