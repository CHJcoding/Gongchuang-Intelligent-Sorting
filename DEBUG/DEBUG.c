#include "debug.h"
#include "cmsis_os2.h" // 需要用到互斥锁和系统状态

/* 声明外部 UART8 句柄 */
extern UART_HandleTypeDef huart8;

/* 互斥锁句柄 */
static osMutexId_t debug_mutex = NULL;
static const osMutexAttr_t debug_mutex_attr = {
  .name = "DebugMutex"
};

/**
 * @brief 初始化 Debug 模块
 * @note 必须在 osKernelInitialize 之后，且调度器启动后调用
 */
void Debug_Init(void)
{
    if (debug_mutex == NULL) {
        debug_mutex = osMutexNew(&debug_mutex_attr);
    }
}

/**
 * @brief 线程安全的串口发送
 */
void Debug_SendData(const uint8_t *data, uint16_t len)
{
    if (len == 0) return;

    // 只有在互斥锁创建成功且系统正在运行时，才去获取锁
    if (debug_mutex != NULL && osKernelGetState() == osKernelRunning) {
        osMutexAcquire(debug_mutex, osWaitForever);
    }

    HAL_UART_Transmit(&huart8, (uint8_t *)data, len, 100);

    if (debug_mutex != NULL && osKernelGetState() == osKernelRunning) {
        osMutexRelease(debug_mutex);
    }
}

/**
 * @brief 线程安全的格式化打印
 */
void Debug_Printf(const char *fmt, ...)
{
    char buf[128]; // 控制单次打印长度，防止栈溢出
    va_list args;
    int len;

    va_start(args, fmt);
    len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (len > 0 && len < sizeof(buf)) {
        Debug_SendData((uint8_t *)buf, (uint16_t)len);
    }
}

/* ============================================================
 * 重定向 printf 到 UART8
 * ============================================================ */
#ifdef __GNUC__
    /* 适用于 GCC (STM32CubeIDE, CLion 等) */
    int _write(int file, char *ptr, int len)
    {
        Debug_SendData((uint8_t *)ptr, (uint16_t)len);
        return len;
    }
#else
    /* 适用于 Keil (AC5 / AC6) */
    int fputc(int ch, FILE *f)
    {
        uint8_t temp = (uint8_t)ch;
        Debug_SendData(&temp, 1);
        return ch;
    }
#endif
    