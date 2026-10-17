#include "usb_processing.h"

//find_sub(uint8_t *buf, uint32_t len,需要判断一致的字符)
//这个方式对比于strncmp，含有滤除噪声的效果
static int find_sub(uint8_t *buf, uint32_t len, const char *token)
{
    uint32_t tlen = strlen(token);
    if(len < tlen) return -1;
    
    for(uint32_t i = 0; i + tlen <= len; i++)
    {
        if(memcmp(&buf[i], token, tlen) == 0)
        {
            if(i == 0)return 0;   //若已经在判断字符上了就返回 不进行数据挪位
            
            uint32_t remaining = len - i;   
            memmove(buf, &buf[i], remaining);   // 重叠内存必须用 memmove
            return 0;
        }
    }
    return -1;
}

//视觉解算
void vision_handle(uint8_t *buf, uint32_t len)
{
    
}

//统一入口函数
void cmd_handle(uint8_t *buf, uint32_t len)
{
    /* 视觉任务入口 */
//    if (strncmp((char*)buf, "vision", 6) == 0)
//    {
//        vision_handle(buf ,len);
//    }
    
    if(find_sub(buf, len, "vision") == 0)
    {
        vision_handle(buf, len);
    }
}

