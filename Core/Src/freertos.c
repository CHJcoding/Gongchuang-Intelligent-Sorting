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
#include "battery_driver.h"
#include "DEBUG.h"
#include "OLED.h"
#include "BUTTON.h"
#include "SENSOR.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */


//typedef struct {
//    ScreenCmdType_t type; // 操作类型
//    uint8_t row;          // 屏幕行号
//    int32_t count;        // 当前的数量
//} ScreenMsg_t;


/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PNEUMATIC_TEST_ENABLE   0U
#define CHASSIS_TEST_ENABLE     0U
#define COMM_TASK_TEST_ENABLE   0U
#define MOTOR_POSITION_REPORT_ENABLE 0U
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
volatile uint8_t g_motion_aborted = 0;  //用于保护（业务保护，比如坐标越界就会1）
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
    Servo_Init();
    OLED_Init();
    Chassis_Init();
    buzzer_Init();
    Debug_Init(); 
    //* 2. 初始状态：亮蓝灯闪烁（等待按键） *//
    WS2812_Blink(1, 0, 0, 255, 50, 200, 800); // 亮度50，亮500ms灭500ms
    WS2812_SetColor(2, 255, 255, 255, 50);
    #if PNEUMATIC_TEST_ENABLE
      for (uint32_t cycle = 0U; cycle < 1; cycle++)
      {


//          Servo_SetPulse(1, SERVO1_GRIP);  
//          osDelay(3000U); 
//          Servo_SetPulse(1, SERVO1_RELEASE);  
//          osDelay(3000U);
//          for(uint8_t i = 0; i < 6; i++)
//          {
//            Spin_servo_position(i);
//             osDelay(3000U);
//          }
//         Pneumatic_On(); 
//          osDelay(3000);
//         Pneumatic_Off();
//          

      }

      //Pneumatic_AllOff();
    #endif

  /* Test complete. Keep the task idle without changing the outputs. */
  for (;;)
  {
      Button_Handler();
      WS2812_Update();
      OLED_UpdataScreen();
      osDelay(10U);
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
          if(g_vision_Data.M_shape_id < NAME_TOTAL && g_vision_Data.M_color_id < COLOR_TOTAL)
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

    
  /* Infinite loop */
  for(;;)
  {
      //ZDTcontrol
      #if CHASSIS_TEST_ENABLE
      
        // 【优雅的一行调用】等待物料掉落，最长等 1500 毫秒
          if (Sensor_WaitForDrop(1500)) 
          {
              // 成功检测到物料穿过传感器掉进桶里
              g_motion_aborted = 0; 
              buzzer_finish_flag(2); // 滴滴，成功提示
          } 
          else 
          {
              // 1.5 秒了还没掉下去，肯定卡住了或者没抓起来
              g_motion_aborted = 1; 
          }
          
      /* ---------- 1. 单轴方向：A 正转 10mm ----------
         * 预期：只有 A 电机转，B 不动
         * 反了：改 A_DIR_SIGN
         */
        //Z_MoveTo(30.0f, 100U, 30U);   // A=B=800，两电机同向
//        osDelay(3000U);

//        /* ---------- 2. 回原点 ---------- */
//        osDelay(2000U);
//      osDelay(1500U);
//      Chassis_MoveToAndWait(PUT_PLACE_X, PUT_PLACE_Y, XY_VEL, XY_ACC);
//        Chassis_MoveToAndWait(130.0f, 120.0f, XY_VEL, XY_ACC);
//        osDelay(5000U);
//        Chassis_MoveToAndWait(0.0f, 0.0f, XY_VEL, XY_ACC);
//        osDelay(5000U);
//      Chassis_MoveTo(130.0f, 120.0f, 300, 225);
//      osDelay(5000U);
//      Chassis_MoveTo(0.0f, 0.0f, 300, 225);
//      osDelay(5000U);
      
//      Z_MoveToAndWait(Z_PICK_HEIGHT+1.0f, Z_VEL, Z_ACC);
        //Servo_SetPulse(SERVO_ADDR1, SERVO1_GRIP);
      //osDelay(1000);
      //Servo_SetPulse(SERVO_ADDR1, SERVO1_RELEASE);
      //Chassis_ReportRealPos();
//      Z_MoveToAndWait(Z_PICK_HEIGHT, Z_VEL, Z_ACC);
//      Debug_Printf("Hello World");
      #endif
      
      //1. 阻塞等待运动指令
      if(osMessageQueueGet(MotionQueueHandle, &target_pos, NULL, 100U) == osOK)
      {
          // --- 放置实际电机、舵机动作 ---
          // ==========================================
          // 阶段 1：准备抓取
          // ==========================================
          // 1. 确保 Z 轴在安全高度（防止平移时撞机）
          Z_MoveToAndWait(Z_SAFE_HEIGHT, Z_VEL, Z_ACC);
          
          // 2. XY 移动到物料上方
          if (Chassis_MoveToAndWait(target_pos.M_x, target_pos.M_y, XY_VEL, XY_ACC) != HAL_OK)
          {
              g_motion_aborted = 1; 
              osSemaphoreRelease(MotionDoneSemHandle); 
              continue; // 直接进入下一轮循环，Z轴完全不动
          }
          Pneumatic_On();
          //2. 下降 吸盘吸取并且舵机抓取
          Z_MoveToAndWait(Z_PICK_HEIGHT, Z_VEL, Z_ACC);
          
          // ==========================================
          // 阶段 2：抓取动作
          // ==========================================
          // 打开吸盘 / 舵机夹紧
          Servo_SetPulse(SERVO_ADDR1, SERVO1_GRIP);
          //osDelay(1000);
           
          //osDelay(500);     //加一点延迟放置吸不上
          
          //3. 抬起 / 归位
           Z_MoveToAndWait(Z_SAFE_HEIGHT, Z_VEL, Z_ACC);
          
          //4. 移动到放置区域（需要考虑过度点）
          // Chassis_MoveToAndWait(PUT_TRANSITION_PLACE_X, PUT_TRANSITION_PLACE_Y, XY_VEL, XY_ACC);
          Chassis_MoveToAndWait(PUT_PLACE_X, PUT_PLACE_Y, XY_VEL, XY_ACC);
          
          //5. 垃圾桶旋转
          //Spin_position(target_pos.row - 1);  这个是张大头
          Spin_servo_position(target_pos.row - 1);    //这个是舵机
          
          //6. z轴下降到放置高度
          //Z_MoveToAndWait(Z_PLACE_HEIGHT, Z_VEL, Z_ACC);
         
          
          //7. 关闭吸盘 / 舵机松开
          Servo_SetPulse(SERVO_ADDR1, SERVO1_RELEASE);
          osDelay(500);
          Pneumatic_Off();
          //osDelay(500);     //加一点延迟放置吸不上
          
          //8. 最后抬起 Z 轴，回到安全高度，准备下次任务
           //Z_MoveToAndWait(Z_SAFE_HEIGHT, Z_VEL, Z_ACC);
          //Servo_SetPulse(SERVO_ADDR1, SERVO1_RELEASE);
          //10.可以滴滴两声（灯光变色等等）
          buzzer_finish_flag(2);
          
          //11.可以加入释放信号量，告诉MainTask抓完了ww
          osSemaphoreRelease(MotionDoneSemHandle);
         
      }
      
      #if MOTOR_POSITION_REPORT_ENABLE
      //测试阶段专用：如果你要读位置，就把这句加上，平时务必关掉
        //Chassis_DisableAll();
      
        //Chassis_ReportRealPos();
      #endif

      
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
          //若接收到了上位机下发的视觉消息，则进行判断
            screen_handle_material(msg.shape_id, msg.color_id);
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
    
  /* Infinite loop */
  for(;;)
  {
      if(osMessageQueueGet(CommQueueHandle, &current_vision, NULL, osWaitForever) == osOK)
      {
        // 1. 通知运动任务区抓取
            osMessageQueuePut(MotionQueueHandle, &current_vision, 0, 0);
            
            // 2. 等待抓取完成 (强烈建议加上 20000U 超时)
          if (osSemaphoreAcquire(MotionDoneSemHandle, 20000U) != osOK)    //感觉后期可以不加，调试加上即可(osWaitForever)
            {       
                uint8_t err_msg[] = "ERR\r\n"; 
                CDC_Transmit_HS(err_msg, sizeof(err_msg)-1);
                continue; 
            }
            
            // 3. 越界/异常检查（这里也不用加）
            if (g_motion_aborted) 
            {
                uint8_t err_msg[] = "ERR\r\n"; 
                while(CDC_Transmit_HS(err_msg, sizeof(err_msg)-1) == USBD_BUSY) osDelay(1); 
                continue; 
            }

            // 4. 【核心精简】抓取成功，把物料信息丢给屏幕任务去处理
            ScreenMsg_t smsg;
            smsg.shape_id = current_vision.M_shape_id;
            smsg.color_id = current_vision.M_color_id;
            osMessageQueuePut(ScreenQueueHandle, &smsg, 0, 0);
            
            // 5. 发送 OK
            uint8_t ack_msg[] = "OK\r\n"; 
            while(CDC_Transmit_HS(ack_msg, sizeof(ack_msg)-1) == USBD_BUSY) osDelay(1);
      }
    osDelay(1);
  }
  /* USER CODE END MainTask */
}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */

