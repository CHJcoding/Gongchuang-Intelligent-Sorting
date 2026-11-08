#ifndef __SENSOR_H
#define __SENSOR_H

#include "main.h"
#include "cmsis_os2.h"

uint8_t Sensor_WaitForDrop(uint32_t timeout_ms);

#endif
