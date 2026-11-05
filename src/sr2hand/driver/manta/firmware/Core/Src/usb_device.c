#include "usb_device.h"
#include "usbd_core.h"
#include "usbd_cdc.h"
#include "usbd_cdc_if.h"
#include "usbd_desc.h"

USBD_HandleTypeDef hUsbDeviceFS;

void MX_USB_DEVICE_Init(void)
{
    USBD_Init(&hUsbDeviceFS, &FS_Desc_Impl, 0U); /* 3rd arg is a plain device-instance
                                                     id in this library version, not a
                                                     speed enum -- DEVICE_FS was a
                                                     leftover from an older USB library
                                                     generation, doesn't exist here. */
    USBD_RegisterClass(&hUsbDeviceFS, &USBD_CDC);
    USBD_CDC_RegisterInterface(&hUsbDeviceFS, &USBD_Interface_fops_FS);

    /* Enumeration has been intermittent -- but the real distinction, found
     * by directly comparing a warm RESET-button press against a genuine
     * power-cycle, is NOT "cold vs DFU-then-reset" as first assumed: a
     * plain warm reset (NRST pin, power rails already stable from before)
     * enumerates fine on its own, every time. Only a true power-on
     * (board and CB1 booting simultaneously from dead power) fails. Two
     * earlier attempts at a short, fixed delay here (100ms, then 800ms)
     * made no difference at all, which makes sense in hindsight: this was
     * never about the STM32's own VBUS/pull-up settling (a sub-second
     * concern) -- it's the CB1's Linux USB host stack not being ready yet.
     * On a simultaneous cold boot the STM32 can easily finish its own
     * (fast) startup and assert the pull-up while Armbian is still
     * multiple seconds into booting and hasn't loaded its USB host driver
     * yet; the host never sees the device appear, and nothing here would
     * make it retry. Fixed by only paying this delay -- several seconds,
     * scaled to Linux boot time rather than STM32 peripheral settling --
     * on an actual power-on reset (RCC_FLAG_PORRST), not a warm one, since
     * a warm reset already works today and doesn't need it. */
    if (__HAL_RCC_GET_FLAG(RCC_FLAG_PORRST)) {
        HAL_Delay(5000);
    }
    __HAL_RCC_CLEAR_RESET_FLAGS();

    USBD_Start(&hUsbDeviceFS);
}
