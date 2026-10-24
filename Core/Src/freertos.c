/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : freertos.c
  * Description        : Code for freertos applications
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "FreeRTOS.h"
#include "cmsis_os2.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "usbd_cdc_if.h"
#include "usb_processing.h"
#include "buzzer.h"
#include "led.h"
#include "servo.h"
#include "Emm_V5.h"
#include "motor.h"
#include "screen.h"
#include "screen_config.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

typedef enum {
    SCREEN_CMD_UPDATE_ROW,  // 更新整行（新物料）
    SCREEN_CMD_ADD_COUNT  // 抓取成功：只更新数量+1
} ScreenCmdType_t;

typedef struct {
    ScreenCmdType_t type; // 操作类型
    uint8_t row;          // 屏幕行号
    int32_t count;        // 当前的数量
} ScreenMsg_t;


/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PNEUMATIC_TEST_ENABLE   1U
#define CHASSIS_TEST_ENABLE     0U
#define COMM_TASK_TEST_ENABLE   0U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

/* USER CODE END Variables */
/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = {
  .name = "defaultTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* Definitions for Comm_Task */
osThreadId_t Comm_TaskHandle;
const osThreadAttr_t Comm_Task_attributes = {
  .name = "Comm_Task",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* Definitions for Indicator_Task */
osThreadId_t Indicator_TaskHandle;
const osThreadAttr_t Indicator_Task_attributes = {
  .name = "Indicator_Task",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityAboveNormal,
};
/* Definitions for Screen_Task */
osThreadId_t Screen_TaskHandle;
const osThreadAttr_t Screen_Task_attributes = {
  .name = "Screen_Task",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityLow,
};
/* Definitions for Main_Task */
osThreadId_t Main_TaskHandle;
const osThreadAttr_t Main_Task_attributes = {
  .name = "Main_Task",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityBelowNormal,
};
/* Definitions for UsbRxQueue */
osMessageQueueId_t UsbRxQueueHandle;
const osMessageQueueAttr_t UsbRxQueue_attributes = {
  .name = "UsbRxQueue"
};
/* Definitions for ScreenQueue */
osMessageQueueId_t ScreenQueueHandle;
const osMessageQueueAttr_t ScreenQueue_attributes = {
  .name = "ScreenQueue"
};
/* Definitions for MotionQueue */
osMessageQueueId_t MotionQueueHandle;
const osMessageQueueAttr_t MotionQueue_attributes = {
  .name = "MotionQueue"
};
/* Definitions for CommQueue */
osMessageQueueId_t CommQueueHandle;
const osMessageQueueAttr_t CommQueue_attributes = {
  .name = "CommQueue"
};
/* Definitions for MotionDoneSem */
osSemaphoreId_t MotionDoneSemHandle;
const osSemaphoreAttr_t MotionDoneSem_attributes = {
  .name = "MotionDoneSem"
};

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

/* USER CODE END FunctionPrototypes */

void StartDefaultTask(void *argument);
void CommTask(void *argument);
void IndicatorTask(void *argument);
void ScreenTask(void *argument);
void MainTask(void *argument);

extern void MX_USB_DEVICE_Init(void);
void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/**
  * @brief  FreeRTOS initialization
  * @param  None
  * @retval None
  */
void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* Create the semaphores(s) */
  /* creation of MotionDoneSem */
  MotionDoneSemHandle = osSemaphoreNew(1, 0, &MotionDoneSem_attributes);

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* Create the queue(s) */
  /* creation of UsbRxQueue */
  UsbRxQueueHandle = osMessageQueueNew (64, sizeof(USB_RxPacket_t), &UsbRxQueue_attributes);

  /* creation of ScreenQueue */
  ScreenQueueHandle = osMessageQueueNew (5, sizeof(ScreenMsg_t), &ScreenQueue_attributes);

  /* creation of MotionQueue */
  MotionQueueHandle = osMessageQueueNew (5, sizeof(comm_vision_t), &MotionQueue_attributes);

  /* creation of CommQueue */
  CommQueueHandle = osMessageQueueNew (5, sizeof(comm_vision_t), &CommQueue_attributes);

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of defaultTask */
  defaultTaskHandle = osThreadNew(StartDefaultTask, NULL, &defaultTask_attributes);

  /* creation of Comm_Task */
  Comm_TaskHandle = osThreadNew(CommTask, NULL, &Comm_Task_attributes);

  /* creation of Indicator_Task */
  Indicator_TaskHandle = osThreadNew(IndicatorTask, NULL, &Indicator_Task_attributes);

  /* creation of Screen_Task */
  Screen_TaskHandle = osThreadNew(ScreenTask, NULL, &Screen_Task_attributes);

  /* creation of Main_Task */
  Main_TaskHandle = osThreadNew(MainTask, NULL, &Main_Task_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

}

/* USER CODE BEGIN Header_StartDefaultTask */
/**
  * @brief  Function implementing the defaultTask thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument)
{
  /* init code for USB_DEVICE */
  MX_USB_DEVICE_Init();
  /* USER CODE BEGIN StartDefaultTask */

    WS2812_Init();

//    //蓝色亮度为100共255，亮200ms灭800ms
    WS2812_Blink(0, 0, 255, 100, 200, 800);

    Servo_Init();

#if PNEUMATIC_TEST_ENABLE
  for (uint32_t cycle = 0U; cycle < 10; cycle++)
  {
      //Servo_SetPulse(1, 500);  
      //osDelay(3000U);         
      Servo_SetPulse(1, 2500);  
      osDelay(3000U);

  }

  Pneumatic_AllOff();
#endif

  /* Test complete. Keep the task idle without changing the outputs. */
  for (;;)
  {
    osDelay(1000U);
  }
  /* USER CODE END StartDefaultTask */
}

/* USER CODE BEGIN Header_CommTask */
/**
* @brief Function implementing the Comm_Task thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_CommTask */
void CommTask(void *argument)
{
  /* USER CODE BEGIN CommTask */
    USB_RxPacket_t rxPacket;
    comm_vision_t v_data;


  /* Infinite loop */
  for(;;)
  {
      if(osMessageQueueGet(UsbRxQueueHandle, &rxPacket, NULL, 0) == osOK)
      {
        // 2. 解析协议
          cmd_handle(rxPacket.buffer, rxPacket.len);

          //3. 打包丢给逻辑大任务
          if(g_vision_Data.M_shape_id <= NAME_TOTAL && g_vision_Data.M_color_id <= COLOR_TOTAL)
          {
            
               v_data.M_shape_id = g_vision_Data.M_shape_id;
                v_data.M_color_id = g_vision_Data.M_color_id;
                v_data.M_x = g_vision_Data.M_x;
                v_data.M_y = g_vision_Data.M_y;
                    //放入视觉队列，供Main_task用
                    osMessageQueuePut(CommQueueHandle, &v_data, 0 ,0);         
                }
          }    
          
#if COMM_TASK_TEST_ENABLE          
          uint8_t txbuf[128];
//            int n = snprintf((char*)txbuf, sizeof(txbuf), "ECHO: %.*s\r\n",
//                             (int)rxPacket.len, rxPacket.buffer);
            int n = snprintf((char*)txbuf, sizeof(txbuf), 
                     "Vision -> Shape:%d, Color:%d, X:%d, Y:%d\r\n",
                     g_vision_Data.M_shape_id, 
                     g_vision_Data.M_color_id, 
                     g_vision_Data.M_x, 
                     g_vision_Data.M_y);
            CDC_Transmit_HS(txbuf, (uint16_t)n);
#endif
            
    osDelay(1);
  }
  /* USER CODE END CommTask */
}

/* USER CODE BEGIN Header_IndicatorTask */
/**
* @brief Function implementing the Indicator_Task thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_IndicatorTask */
void IndicatorTask(void *argument)
{
  /* USER CODE BEGIN IndicatorTask */
    
    comm_vision_t target_pos;
    
    #if CHASSIS_TEST_ENABLE
        Chassis_Init();
        osDelay(200);
        
         /* 从 (0,0) 开始 */
        Chassis_SetZeroPositions();
        osDelay(200U);
    #endif
    
  /* Infinite loop */
  for(;;)
  {
      //ZDTcontrol
      #if CHASSIS_TEST_ENABLE
      /* ---------- 1. 单轴方向：A 正转 10mm ----------
         * 预期：只有 A 电机转，B 不动
         * 反了：改 A_DIR_SIGN
         */
        Chassis_MoveTo(10.0f, 0.0f, 100U, 30U);   // A=B=800，两电机同向
        osDelay(1500U);

        /* ---------- 2. 回原点 ---------- */
        Chassis_MoveTo(0.0f, 0.0f, 100U, 30U);
        osDelay(1500U);
      #endif
      
      //1. 阻塞等待运动指令
      if(osMessageQueueGet(MotionQueueHandle, &target_pos, NULL, osWaitForever) == osOK)
      {
          // --- 放置实际电机、舵机动作 ---
          
          //1. 移动到目标位置
          //Chassis_MoveTo(target_pos.x, target_pos.y, 100U, 30U);
          
          //2. 下降 吸盘吸取并且舵机抓取
          // ...
          // Pneumatic_AllOn(); 
          
          osDelay(500); //模拟动作延迟
          
          //3. 抬起 / 归位
          // ...
          
          //4. 移动到放置区域
          //Chassis_MoveTo(放置位x, 放置位Y, 100U, 30U);
          
          //5. 关闭吸盘和舵机抓取
          // ...
          //Pneumatic_AllOff(); 
          
          //6.发送标志位 切计数成功
          // ...
          
          //7.可以加入释放信号量，告诉MainTask抓完了
          osSemaphoreRelease(MotionDoneSemHandle);
      }
      
    osDelay(1);
  }
  /* USER CODE END IndicatorTask */
}

/* USER CODE BEGIN Header_ScreenTask */
/**
* @brief Function implementing the Screen_Task thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_ScreenTask */
void ScreenTask(void *argument)
{
  /* USER CODE BEGIN ScreenTask */
    ScreenMsg_t msg;
    
  /* Infinite loop */
  for(;;)
  {
      if(osMessageQueueGet(ScreenQueueHandle, &msg, NULL, osWaitForever) == osOK)
      {
            screen_update_vision(msg.row, msg.count);
      }
      
    osDelay(1);
  }
  /* USER CODE END ScreenTask */
}

/* USER CODE BEGIN Header_MainTask */
/**
* @brief Function implementing the Main_Task thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_MainTask */
void MainTask(void *argument)
{
  /* USER CODE BEGIN MainTask */
    comm_vision_t current_vision;
    
    static uint8_t current_row = 1;         //第几行
    static int32_t current_count = 0;       //抓放成功数量
    static uint8_t last_shape = 0xFF;
    static uint8_t last_color = 0xFF; 
    static bool task_finished = false;  //任务完成标志位
    
  /* Infinite loop */
  for(;;)
  {
      if(osMessageQueueGet(CommQueueHandle, &current_vision, NULL, osWaitForever) == osOK)
      {
        if(task_finished)
        {
            continue;   //任务结束丢弃后续数据
        }
        
        //2. 业务判断：是否换行
        if(current_vision.M_shape_id != last_shape || current_vision.M_color_id != last_color)
        {
            if (last_shape != 0xFF) current_row++;
            
            if (current_row > 6) 
            {
                    task_finished = true;
                    // 可以给屏幕发个“任务完成”的提示
                    continue; 
            }
            
            last_shape = current_vision.M_shape_id;
            last_color = current_vision.M_color_id;
            current_count = 0; 
            
            // 通知屏幕：新物料进场 (数量 0)
            ScreenMsg_t smsg = {SCREEN_CMD_UPDATE_ROW, current_row, current_count};
            osMessageQueuePut(ScreenQueueHandle, &smsg, 0, 0);
        }
        
        //3.通知运动任务区抓取
        osMessageQueuePut(MotionQueueHandle, &current_vision, 0, 0);
        
        //4.大任务在这里挂起，等待运动完成
        //永远等待信号量，只有当MotionTask 抓完了才会被唤醒
        osSemaphoreAcquire(MotionDoneSemHandle, osWaitForever);
        
        //5.抓取成功，更新计数，通知屏幕
        current_count++;
        ScreenMsg_t smsg_add = {SCREEN_CMD_ADD_COUNT, current_row, current_count};
        osMessageQueuePut(ScreenQueueHandle, &smsg_add, 0, 0);
        
        //6. 向RDK发送一个ACK，允许发下一个物料
        uint8_t ack_msg[] = "OK\r\n"; 
        while(CDC_Transmit_HS(ack_msg, sizeof(ack_msg)-1) == USBD_BUSY) 
        {
            osDelay(1); 
        }
      }
      
    osDelay(1);
  }
  /* USER CODE END MainTask */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */

