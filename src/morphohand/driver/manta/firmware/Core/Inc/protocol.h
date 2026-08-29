#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

void Protocol_Init(void);

/* Fed one byte at a time from the USB CDC receive callback (usbd_cdc_if.c).
 * Cheap and ISR-safe: just pushes into a ring buffer. */
void Protocol_OnUsbRxByte(uint8_t byte);

/* Call from the main loop: parses any complete lines received and executes
 * them, writing replies back out over USB CDC. */
void Protocol_Poll(void);

#endif /* PROTOCOL_H */
