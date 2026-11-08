#ifndef __SCREEN_H__
#define __SCREEN_H__

#include "cmsis_os2.h"
#include "usart.h"   /* CubeMX生成的UART10句柄 */
#include "usb_processing.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>


/*========== 后期删除 不需要 ==========*/
typedef enum {
    SCREEN_CMD_UPDATE_ROW,  // 更新整行（新物料）
    SCREEN_CMD_ADD_COUNT  // 抓取成功：只更新数量+1
} ScreenCmdType_t;
/*========== ================ ==========*/


/* 【修改】现在只需要传形状和颜色，业务逻辑交给 screen.c 内部处理 */
typedef struct {
    uint8_t shape_id;
    uint8_t color_id;
} ScreenMsg_t;


/* 发送控件 */
void screen_send_text(const char *obj, const char *txt);
void screen_change_page(const char *page_name);
void screen_update_row(uint8_t row, const char *name, uint32_t pic_id, int32_t count);



/*========== 后期删除 不需要 ==========*/
void screen_update_vision(uint8_t target_row, int32_t count);
/*========== ================ ==========*/


/* 【新增】主控调用的核心接口：记录并刷新屏幕 */
void screen_handle_material(uint8_t shape_id, uint8_t color_id);

#endif
