/* USB device descriptors for the CDC-ACM virtual serial port the CB1 sees
 * as /dev/ttyACM0. VID/PID reuse ST's default CDC demo IDs (0483:5740) --
 * fine for a lab-internal board-to-board USB link; change if this ever
 * needs to coexist with other devices claiming that VID:PID pattern. */

#include "usbd_core.h"
#include "usbd_desc.h"

#define USBD_VID     0x0483
#define USBD_PID     0x5740
#define USBD_LANGID_STRING     1033

static uint8_t USBD_FS_DeviceDesc[USB_LEN_DEV_DESC] = {
    0x12, USB_DESC_TYPE_DEVICE, 0x00, 0x02, 0x02, 0x02, 0x01, 0x40,
    LOBYTE(USBD_VID), HIBYTE(USBD_VID), LOBYTE(USBD_PID), HIBYTE(USBD_PID),
    0x00, 0x02, 1, 2, 3, 1
};

static uint8_t USBD_LangIDDesc[USB_LEN_LANGID_STR_DESC] = {
    USB_LEN_LANGID_STR_DESC, USB_DESC_TYPE_STRING,
    LOBYTE(USBD_LANGID_STRING), HIBYTE(USBD_LANGID_STRING)
};

static uint8_t str_buf[256];

static uint8_t *USBD_FS_DeviceDescriptor(USBD_SpeedTypeDef speed, uint16_t *length)
{ *length = sizeof(USBD_FS_DeviceDesc); return USBD_FS_DeviceDesc; }

static uint8_t *USBD_FS_LangIDStrDescriptor(USBD_SpeedTypeDef speed, uint16_t *length)
{ *length = sizeof(USBD_LangIDDesc); return USBD_LangIDDesc; }

static uint8_t *USBD_FS_ManufacturerStrDescriptor(USBD_SpeedTypeDef speed, uint16_t *length)
{ USBD_GetString((uint8_t *)"CU Robotics Lab", str_buf, length); return str_buf; }

static uint8_t *USBD_FS_ProductStrDescriptor(USBD_SpeedTypeDef speed, uint16_t *length)
{ USBD_GetString((uint8_t *)"Manta Hand Controller", str_buf, length); return str_buf; }

static uint8_t *USBD_FS_SerialStrDescriptor(USBD_SpeedTypeDef speed, uint16_t *length)
{ USBD_GetString((uint8_t *)"MHC-0001", str_buf, length); return str_buf; }

USBD_DescriptorsTypeDef FS_Desc_Impl = {
    USBD_FS_DeviceDescriptor,
    USBD_FS_LangIDStrDescriptor,
    USBD_FS_ManufacturerStrDescriptor,
    USBD_FS_ProductStrDescriptor,
    USBD_FS_SerialStrDescriptor,
    NULL,
    NULL,
};
