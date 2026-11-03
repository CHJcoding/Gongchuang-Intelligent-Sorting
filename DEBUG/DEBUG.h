#ifndef __DEBUG_H
#define __DEBUG_H

#include "main.h"
#include "usart.h"  // 包含 huart8 的定义
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* 初始化 Debug 模块 (必须在 FreeRTOS 调度器启动后调用，因为要创建互斥锁) */
void Debug_Init(void);

/* 线程安全的串口发送函数 */
void Debug_SendData(const uint8_t *data, uint16_t len);

/* 线程安全的格式化打印函数 (替代 printf) */
void Debug_Printf(const char *fmt, ...);

#endif
