#include "screen.h" 
#include "screen_config.h"

// =========================================================
// 数组定义放在 .c 文件中，防止头文件被多次包含导致报错
// =========================================================

// 形状名称查找表：通过 shape_id 直接拿到中文名
const char* const shape_name_table[10] = {
    NAME_CUBOID, NAME_HOLLOW_CUBE, NAME_SPHERE, NAME_TRI_PRISM, NAME_CONE,
    NAME_CUBE, NAME_CYLINDER, NAME_PENT_PRISM, NAME_HEX_PRISM, NAME_TRI_PYRAMID
};

// 二维图片索引表: [形状ID][颜色ID] = 屏幕图片ID
const uint8_t shape_color_pic_table[10][10] = {
    // 颜色: 0(红) 1(橙) 2(黄) 3(绿) 4(棕) 5(蓝) 6(紫) 7(品) 8(黑) 9(白)
    /* 0 长方体     */  {   0,    1,    2,    3,    4,    5,    6,    7,    8,    9 },
    /* 1 镂空正方体 */  {  10,   11,   12,   13,   14,   15,   16,   17,   18,   19 },
    /* 2 球         */  {  20,   21,   22,   23,   24,   25,   26,   27,   28,   29 },
    /* 3 三棱柱     */  {  30,   31,   32,   33,   34,   35,   36,   37,   38,   39 },
    /* 4 圆锥       */  {  40,   41,   42,   43,   44,   45,   46,   47,   48,   49 },
    /* 5 正方体     */  {  50,   51,   52,   53,   54,   55,   56,   57,   58,   59 },
    /* 6 圆柱       */  {  60,   61,   62,   63,   64,   65,   66,   67,   68,   69 },
    /* 7 五棱柱     */  {  70,   71,   72,   73,   74,   75,   76,   77,   78,   79 },
    /* 8 六棱柱     */  {  80,   81,   82,   83,   84,   85,   86,   87,   88,   89 },
    /* 9 三棱锥     */  {  90,   91,   92,   93,   94,   95,   96,   97,   98,   99 }
};


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


// =========================================================
// 新增：自动根据视觉解析结果更新屏幕
// =========================================================
/**
 * @brief 根据视觉识别结果自动更新屏幕指定行
 * @param target_row 屏幕上的目标行 (1~6)
 * @param count 当前累计的分拣数量
 */
void screen_update_vision(uint8_t target_row, int32_t count)
{
    uint8_t s_id = g_vision_Data.M_shape_id;
    uint8_t c_id = g_vision_Data.M_color_id;
    

    // 1. 查表获取文字名称
    const char *name = shape_name_table[s_id];
    
    // 2. 查二维表获取组合图片 ID
    uint32_t pic_id = shape_color_pic_table[s_id][c_id];
    
    // 3. 调用你原有的函数更新整行
    screen_update_row(target_row, name, pic_id, count);
}
