#ifndef __SCREEN_CONFIG_H__
#define __SCREEN_CONFIG_H__

#include "cmsis_os.h"

// 形状/物料名称
#define NAME_CUBOID             "\xB3\xA4\xB7\xBD\xCC\xE5"                // 长方体
#define NAME_HOLLOW_CUBE        "\xE2\xCF\xBF\xD5\xD5\xFD\xB7\xBD\xCC\xE5" // 镂空正方体
#define NAME_SPHERE             "\xC7\xF2"                                // 球
#define NAME_TRI_PRISM          "\xC8\xFD\xC0\xE2\xD6\xF9"                // 三棱柱
#define NAME_CONE               "\xD4\xB2\xD7\xB6"                        // 圆锥
#define NAME_CUBE               "\xD5\xFD\xB7\xBD\xCC\xE5"                // 正方体
#define NAME_CYLINDER           "\xD4\xB2\xD6\xF9"                        // 圆柱
#define NAME_PENT_PRISM         "\xCE\xE5\xC0\xE2\xD6\xF9"                // 五棱柱
#define NAME_HEX_PRISM          "\xC1\xF9\xC0\xE2\xD6\xF9"                // 六棱柱
#define NAME_TRI_PYRAMID        "\xC8\xFD\xC0\xE2\xD7\xB6"                // 三棱锥

// 颜色定义
#define COLOR_MATERIAL_RED           0          //红色
#define COLOR_MATERIAL_ORANGE        1          //橙色
#define COLOR_MATERIAL_YELLOW        2          //黄色
#define COLOR_MATERIAL_GREEN         3          //绿色
#define COLOR_MATERIAL_CYAN          4          //棕色
#define COLOR_MATERIAL_BLUE          5          //蓝色
#define COLOR_MATERIAL_PURPLE        6          //紫色
#define COLOR_MATERIAL_MAGENTA       7          //品红
#define COLOR_MATERIAL_BLACK         8          //黑色
#define COLOR_MATERIAL_WHITE         9          //白色

#define COLOR_TOTAL        10
#define NAME_TOTAL         10

#endif
