#ifndef __USB_PROCESSING_H__
#define __USB_PROCESSING_H__

#include "cmsis_os.h"
#include <string.h>

void cmd_handle(uint8_t *buf, uint32_t len);

#endif
