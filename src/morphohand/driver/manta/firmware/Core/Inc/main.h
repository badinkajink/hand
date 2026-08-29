#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h7xx_hal.h"

void Error_Handler(void);

/* Onboard status LED is not confirmed against the V2.0 schematic yet --
 * this pin is a guess (commonly PA... varies by board) and MUST be checked
 * before relying on it. Until then, treat any Error_Handler() call as
 * "halted" and check via a debug probe or by probing STAT over USB. */
/* #define STATUS_LED_PORT GPIOA */
/* #define STATUS_LED_PIN  GPIO_PIN_0 */

#ifdef __cplusplus
}
#endif

#endif /* MAIN_H */
