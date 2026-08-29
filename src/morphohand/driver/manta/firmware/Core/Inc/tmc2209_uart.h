#ifndef TMC2209_UART_H
#define TMC2209_UART_H

#include <stdint.h>

/* TMC2209 register addresses we actually use */
#define TMC_REG_GCONF      0x00
#define TMC_REG_IOIN       0x06  /* read-only, has a version field -- good "is this driver alive" probe */
#define TMC_REG_IHOLD_IRUN 0x10
#define TMC_REG_TPOWERDOWN 0x11
#define TMC_REG_TPWMTHRS   0x13
#define TMC_REG_CHOPCONF   0x6C

void TMC_Init(void);

/* Returns 0 on success, negative on UART/CRC/timeout error. */
int TMC_WriteReg(uint8_t axis, uint8_t reg, uint32_t data);
int TMC_ReadReg(uint8_t axis, uint8_t reg, uint32_t *data_out);

/* DIAGNOSTIC: read with an explicit slave-address byte instead of the
 * fixed TMC_ADDR_TX. See the definition in tmc2209_uart.c for why this
 * exists -- suspected MS1/MS2 address-strapping mismatch. */
int TMC_ReadRegAddr(uint8_t axis, uint8_t slave_addr, uint8_t reg, uint32_t *data_out);

/* run_ma/hold_ma are RMS current in milliamps; converted to TMC2209's
 * 5-bit CS (current scale) field using the standard Rsense=0.11ohm formula
 * used on BTT's TMC2209 driver boards. */
int TMC_SetCurrent(uint8_t axis, uint16_t run_ma, uint16_t hold_ma);
int TMC_SetMicrosteps(uint8_t axis, uint16_t usteps); /* 1,2,4,8,16,32,64,128,256 */

#endif /* TMC2209_UART_H */
