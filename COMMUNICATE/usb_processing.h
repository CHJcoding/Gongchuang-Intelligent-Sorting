#ifndef __USB_PROCESSING_H__
#define __USB_PROCESSING_H__

#include "cmsis_os.h"
#include <string.h>

typedef struct {
    uint8_t M_shape_id;
    uint8_t M_color_id;
    int16_t M_x;
    int16_t M_y;
    uint8_t row;    // 【新增】当前是第几行（供运动任务控制放置位）
}comm_vision_t;

extern comm_vision_t g_vision_Data;

void cmd_handle(uint8_t *buf, uint32_t len);

#endif
