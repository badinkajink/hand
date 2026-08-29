Vendored third-party code (all BSD-3-Clause, Copyright (c) STMicroelectronics):

- `Drivers/CMSIS/Include` — https://github.com/STMicroelectronics/cmsis_core
- `Drivers/CMSIS/Device/ST/STM32H7xx` — https://github.com/STMicroelectronics/cmsis_device_h7
- `Drivers/STM32H7xx_HAL_Driver` — https://github.com/STMicroelectronics/stm32h7xx_hal_driver
- `Middlewares/ST/STM32_USB_Device_Library` — https://github.com/STMicroelectronics/stm32_mw_usb_device (Core + Class/CDC only)

Pulled as a shallow clone of each repo's default branch; not tracked as git submodules since this project isn't a git repo yet. Re-vendor by re-running the same shallow clone + copy if you need a newer HAL/CMSIS release.

Everything else in `firmware/` (Core/, Makefile, linker script, TMC2209 driver, stepper motion code, protocol) is original to this project.
