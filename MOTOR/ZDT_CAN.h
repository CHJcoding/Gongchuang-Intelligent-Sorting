#ifndef __ZDT_CAN_H__
#define __ZDT_CAN_H__

#include "main.h"
#include "fdcan.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

typedef void (*CAN_ExtRxCallback_t)(uint32_t id, const uint8_t *data, uint8_t len);

void USER_CAN1_Init(void);
void can_SendCmd(__IO uint8_t *cmd, uint8_t len);
void CAN_RegisterExtRxCallback(CAN_ExtRxCallback_t cb);

#endif
