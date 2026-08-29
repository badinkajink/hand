#ifndef STM32H7xx_HAL_CONF_H
#define STM32H7xx_HAL_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

/* Module enable ------------------------------------------------------------*/
#define HAL_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED
#define HAL_TIM_MODULE_ENABLED
#define HAL_PCD_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED

/* Oscillator values ---------------------------------------------------------*/
#if !defined  (HSE_VALUE)
#define HSE_VALUE    25000000U   /* Manta M8P V2.0 crystal, confirmed 25MHz per BTT's own Klipper build config */
#endif

#if !defined  (HSE_STARTUP_TIMEOUT)
/* HAL's stock default (100ms) was left in place through the intermittent
 * USB enumeration debugging documented in docs/bringup.md -- a settle
 * delay before USBD_Start() (up to 800ms) made no difference at all, which
 * argues against a USB-pull-up-timing explanation and points further back:
 * if HSE occasionally takes just over 100ms to stabilize on a cold boot
 * (real crystals vary boot to boot, especially without ideal load-cap
 * tuning), SystemClock_Config()'s HAL_RCC_OscConfig() call times out and
 * drops straight into Error_Handler()'s silent infinite hang -- BEFORE
 * USB, or anything else, ever initializes. That failure mode is
 * indistinguishable from "USB never came up" when the only way to check is
 * "does /dev/ttyACM0 exist", which is exactly what's been observed. A
 * longer timeout costs nothing on a healthy boot (a good crystal still
 * locks in a few ms either way) and only matters for the failure case. */
#define HSE_STARTUP_TIMEOUT    5000U
#endif

#if !defined  (CSI_VALUE)
#define CSI_VALUE    4000000U
#endif

#if !defined  (HSI_VALUE)
#define HSI_VALUE    64000000U
#endif

#if !defined  (LSI_VALUE)
#define LSI_VALUE  32000U
#endif

#if !defined  (LSE_VALUE)
#define LSE_VALUE  32768U
#endif

#if !defined  (LSE_STARTUP_TIMEOUT)
#define LSE_STARTUP_TIMEOUT    5000U
#endif

#if !defined  (EXTERNAL_CLOCK_VALUE)
#define EXTERNAL_CLOCK_VALUE    12288000U
#endif

/* System configuration ------------------------------------------------------*/
#define  VDD_VALUE                    3300U
#define  TICK_INT_PRIORITY            0x0FU
#define  USE_RTOS                     0U
#define  PREFETCH_ENABLE               0U
#define  USE_SD_TRANSCEIVER            0U
#define  USE_SPI_CRC                   0U

/* Assert -----------------------------------------------------------------*/
#define USE_FULL_ASSERT    0U

#include "stm32h7xx_hal_rcc.h"
#include "stm32h7xx_hal_gpio.h"
#include "stm32h7xx_hal_dma.h"
#include "stm32h7xx_hal_cortex.h"
#include "stm32h7xx_hal_pwr.h"
#include "stm32h7xx_hal_pwr_ex.h"
#include "stm32h7xx_hal_uart.h"
#include "stm32h7xx_hal_tim.h"
#include "stm32h7xx_hal_pcd.h"
#include "stm32h7xx_hal_flash.h"
#include "stm32h7xx_hal_flash_ex.h"
#include "stm32h7xx_hal_exti.h"

#if USE_FULL_ASSERT
#define assert_param(expr) ((expr) ? (void)0U : assert_failed((uint8_t *)__FILE__, __LINE__))
void assert_failed(uint8_t *file, uint32_t line);
#else
#define assert_param(expr) ((void)0U)
#endif

#ifdef __cplusplus
}
#endif

#endif /* STM32H7xx_HAL_CONF_H */
