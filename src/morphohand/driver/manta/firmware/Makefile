# CLI Makefile build for the manta-hand firmware (STM32H723ZET6).
# Intended to run directly on the CB1 (install the cross toolchain with
# `sudo apt install gcc-arm-none-eabi` on Armbian) -- see docs/bringup.md.

TARGET = manta_hand_fw
BUILD_DIR = build

PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
SZ = $(PREFIX)size

CPU = -mcpu=cortex-m7
FPU = -mfpu=fpv5-d16
FLOAT = -mfloat-abi=hard
MCU = $(CPU) -mthumb $(FPU) $(FLOAT)

CORE_DIR       = Core
CMSIS_DIR      = Drivers/CMSIS
CMSIS_DEV_DIR  = Drivers/CMSIS/Device/ST/STM32H7xx
HAL_DIR        = Drivers/STM32H7xx_HAL_Driver
USBMW_DIR      = Middlewares/ST/STM32_USB_Device_Library

C_SOURCES = \
  $(wildcard $(CORE_DIR)/Src/*.c) \
  $(CMSIS_DEV_DIR)/Source/Templates/system_stm32h7xx.c \
  $(wildcard $(HAL_DIR)/Src/*.c) \
  $(wildcard $(USBMW_DIR)/Core/Src/*.c) \
  $(wildcard $(USBMW_DIR)/Class/CDC/Src/*.c)

# HAL "_template.c" files are meant to be copied+customized, not compiled as-is.
# Most LL (*_ll_*.c) driver files are a separate, optional API we never call
# directly, and they pull in stm32_assert.h (a user-supplied header that was
# never vendored). stm32h7xx_ll_usb.c is the one exception: it's not a
# generic peripheral LL API, it's the internal OTG core driver that
# stm32h7xx_hal_pcd.c (which we DO use, for USB-CDC) calls into directly --
# confirmed it has no stm32_assert.h dependency, so it's excluded from the
# blanket LL filter and added back explicitly.
C_SOURCES := $(filter-out %_template.c,$(C_SOURCES))
C_SOURCES := $(filter-out $(HAL_DIR)/Src/stm32h7xx_ll_%.c,$(C_SOURCES))
C_SOURCES += $(HAL_DIR)/Src/stm32h7xx_ll_usb.c

ASM_SOURCES = $(CMSIS_DEV_DIR)/Source/Templates/gcc/startup_stm32h723xx.s

C_INCLUDES = \
  -I$(CORE_DIR)/Inc \
  -I$(CMSIS_DIR)/Include \
  -I$(CMSIS_DEV_DIR)/Include \
  -I$(HAL_DIR)/Inc \
  -I$(HAL_DIR)/Inc/Legacy \
  -I$(USBMW_DIR)/Core/Inc \
  -I$(USBMW_DIR)/Class/CDC/Inc

# App is flashed at 0x08000000 (see linker script) -- default vector table
# location, no relocation needed.
#
# USE_PWR_LDO_SUPPLY: without this, CMSIS's ExitRun0Mode() (called from the
# startup file before anything else, even before SystemInit) silently no-ops
# instead of configuring the chip's power supply path -- discovered while
# chasing a "zero USB activity, ever" symptom. LDO is the simple/common
# supply mode (matches this board having no external SMPS inductor
# populated); matches the PWR_LDO_SUPPLY choice already used in our own
# SystemClock_Config() in main.c.
C_DEFS = -DSTM32H723xx -DUSE_HAL_DRIVER -DUSE_PWR_LDO_SUPPLY

CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) -Og -g3 -Wall -fdata-sections -ffunction-sections -std=gnu11
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"

LDSCRIPT = $(CMSIS_DEV_DIR)/Source/Templates/gcc/STM32H723ZETX_FLASH.ld
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) -lc -lm -lnosys \
          -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections

OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).bin $(BUILD_DIR)/$(TARGET).hex

$(BUILD_DIR)/%.o: %.c Makefile | $(BUILD_DIR)
	$(CC) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/%.o: %.s Makefile | $(BUILD_DIR)
	$(AS) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS) Makefile
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@

$(BUILD_DIR)/%.hex: $(BUILD_DIR)/%.elf
	$(CP) -O ihex $< $@

$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf
	$(CP) -O binary -S $< $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

# Board must be in DFU mode first: hold BOOT0, tap RESET, release BOOT0.
# Run this from a terminal on the CB1 (the CB1<->MCU link is internal USB,
# so no separate host PC is involved) -- see docs/bringup.md. sudo is needed
# here (not for the build itself) because claiming a USB device via libusb
# needs more than the read access `lsusb` gets by default -- matches BTT's
# own documented dfu-util invocation, which also uses sudo.
flash: $(BUILD_DIR)/$(TARGET).bin
	sudo dfu-util -d ,0483:df11 -a 0 -s 0x08000000:leave -D $(BUILD_DIR)/$(TARGET).bin || \
	  echo "dfu-util's post-'leave' status-query error is expected -- the device resets \
itself to jump to the new firmware before it can ACK that query. Check for \
'File downloaded successfully' above; that's the real success signal."

.PHONY: all clean flash
