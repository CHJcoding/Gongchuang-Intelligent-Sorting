#include "usb_processing.h"


comm_vision_t g_vision_Data; 

// 1. 修改 find_sub，传入 len 的指针，以便在挪动数据后更新有效长度
// 这样能避免 vision_handle 读取到越界的数据
//这个方式对比于strncmp，含有滤除噪声的效果
static int find_sub(uint8_t *buf, uint32_t *len, const char *token)
{
    uint32_t tlen = strlen(token);
    if(*len < tlen) return -1;
    
    for(uint32_t i = 0; i + tlen <= *len; i++)
    {
        if(memcmp(&buf[i], token, tlen) == 0)
        {
            if(i == 0) return 0;   // 已经在帧头位置，直接返回
            
            uint32_t remaining = *len - i;   
            memmove(buf, &buf[i], remaining);   // 重叠内存必须用 memmove
            *len = remaining;                   // 【关键修改】更新剩余的有效长度
            return 0;
        }
    }
    return -1;
}

//视觉解算
void vision_handle(uint8_t *buf, uint32_t len)
{

    ///校验帧尾（字节9：0x0D）且 长度小于10字节，直接返回
    if (len < 10 || buf[9] != 0x0D) {

        
        return; 
    }
    
    g_vision_Data.M_shape_id = buf[2];
    g_vision_Data.M_color_id = buf[3];
    
    //字节4-5：x(低字节在前)
    //将两个 uint8_t 拼为 int16_t
    g_vision_Data.M_x = (int16_t)(buf[4] | (buf[5] << 8));

    //字节6-7：y
    //将两个 uint8_t 拼为 int16_t
    g_vision_Data.M_y = (int16_t)(buf[6] | (buf[7] << 8));

}

// 3. 统一入口函数
void cmd_handle(uint8_t *buf, uint32_t len)
{
    
    
    // 使用 "\xAA\x55" 作为 token 去查找帧头
    // 这里传入 &len 是因为 find_sub 内部挪动数据后需要更新长度
    if(find_sub(buf, &len, "\xAA\x55") == 0)
    {
        vision_handle(buf, len);
    }
}

