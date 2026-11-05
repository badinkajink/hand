# manta-hand firmware

Custom STM32H723 firmware for the BIGTREETECH Manta M8P V2.0 -- 8
independent open-loop stepper axes over TMC2209, controlled from a Python
host over USB-CDC. No Klipper. See `../docs/` for the wire protocol, pin
mapping, and (most important before first power-up) the bring-up checklist.

## Build

```sh
make          # -> build/manta_hand_fw.bin
make flash    # requires BOOT0+RESET into DFU mode first
make clean
```

Needs `arm-none-eabi-gcc` and `dfu-util` -- see `docs/bringup.md` for the
one-line apt install on the CB1's Armbian.

## Layout

- `Core/` -- application code (`main.c`, `stepper.c`, `tmc2209_uart.c`,
  `protocol.c`, the USB-CDC glue) plus HAL/board config headers.
- `Drivers/`, `Middlewares/` -- vendored ST code (CMSIS, HAL, USB Device
  library). See `VENDORED.md` for exactly what and where from. Don't hand-edit
  these; re-vendor instead.
