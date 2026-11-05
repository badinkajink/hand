#ifndef USBD_CONF_H
#define USBD_CONF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stm32h7xx_hal.h"

#define USBD_MAX_NUM_INTERFACES     1U
#define USBD_MAX_NUM_CONFIGURATION  1U
#define USBD_MAX_STR_DESC_SIZ       512U
#define USBD_SELF_POWERED           0U
#define USBD_DEBUG_LEVEL            0U
#define USBD_LPM_ENABLED            0U

/* OTG_HS peripheral run in Full-Speed mode via its internal embedded PHY --
 * this board has no external ULPI HS PHY populated, matching the standard
 * "USB_OTG_HS in FS mode" pattern used on Nucleo-H723ZG and similar. */
#define USE_USB_HS
#define USBD_CDC_INTERVAL           1U

#define USBD_malloc  malloc
#define USBD_free    free
#define USBD_memset  memset
#define USBD_memcpy  memcpy

#define __USBD_MALLOC   malloc
#define __USBD_FREE     free
#define __USBD_MEMCPY   memcpy

#endif /* USBD_CONF_H */
