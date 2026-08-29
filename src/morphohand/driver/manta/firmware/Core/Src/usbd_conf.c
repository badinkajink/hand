/* PCD (USB peripheral controller driver) HAL glue + the USBD_LL_* functions
 * the USB device middleware calls into. Standard CubeMX-generated pattern
 * for "USB_OTG_HS running as Full-Speed with the internal embedded PHY". */

#include "usbd_core.h"
#include "stm32h7xx_hal.h"

PCD_HandleTypeDef hpcd_USB_OTG_HS;

void HAL_PCD_MspInit(PCD_HandleTypeDef *hpcd)
{
    if (hpcd->Instance != USB_OTG_HS) return;

    /* PA11=DM, PA12=DP -- confirmed against the Manta M8P V2.0's own schematic
     * (BIGTREETECH MANTA M8P V2.0-SCH.pdf, MCU sheet: pin 103=PA11/OTG_FS_DM
     * net "USB_N", pin 104=PA12/OTG_FS_DP net "USB_P"), NOT PB14/PB15 -- an
     * earlier version of this code used PB14/PB15 by analogy with H7 Nucleo
     * boards without checking this board's actual wiring, which cost a very
     * long debugging session (everything inside the USB_OTG_HS peripheral
     * checked out correct via SWD/register inspection -- GAHBCFG.GINT,
     * GCCFG.PWRDWN, DCTL.SDIS, even the old PB14/15 AF10 config itself --
     * because none of that depends on which GPIO pins are actually wired to
     * the physical connector; only a real schematic check catches this). */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    GPIO_InitTypeDef gi = {0};
    gi.Pin = GPIO_PIN_11 | GPIO_PIN_12;
    gi.Mode = GPIO_MODE_AF_PP;
    gi.Pull = GPIO_NOPULL;
    gi.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gi.Alternate = GPIO_AF10_OTG1_FS; /* same AF10 (0x0A) as before -- only the pins were wrong */
    HAL_GPIO_Init(GPIOA, &gi);

    /* USB1_OTG_HS is this chip's only USB controller (used here in FS mode
     * via its internal PHY) -- NOT USB2_OTG_FS, which is a different
     * peripheral instance/RCC bit that doesn't exist on this board. Mixing
     * these up would silently clock the wrong peripheral and USB would
     * never come up. */
    __HAL_RCC_USB1_OTG_HS_CLK_ENABLE();

    HAL_NVIC_SetPriority(OTG_HS_IRQn, 3, 0);
    HAL_NVIC_EnableIRQ(OTG_HS_IRQn);
}

void HAL_PCD_MspDeInit(PCD_HandleTypeDef *hpcd)
{
    if (hpcd->Instance != USB_OTG_HS) return;
    __HAL_RCC_USB1_OTG_HS_CLK_DISABLE();
    HAL_NVIC_DisableIRQ(OTG_HS_IRQn);
}

void HAL_PCD_SetupStageCallback(PCD_HandleTypeDef *hpcd) { USBD_LL_SetupStage((USBD_HandleTypeDef *)hpcd->pData, (uint8_t *)hpcd->Setup); }
void HAL_PCD_DataOutStageCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum) { USBD_LL_DataOutStage((USBD_HandleTypeDef *)hpcd->pData, epnum, hpcd->OUT_ep[epnum].xfer_buff); }
void HAL_PCD_DataInStageCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum) { USBD_LL_DataInStage((USBD_HandleTypeDef *)hpcd->pData, epnum, hpcd->IN_ep[epnum].xfer_buff); }
void HAL_PCD_SOFCallback(PCD_HandleTypeDef *hpcd) { USBD_LL_SOF((USBD_HandleTypeDef *)hpcd->pData); }
void HAL_PCD_ResetCallback(PCD_HandleTypeDef *hpcd)
{
    USBD_SpeedTypeDef speed = USBD_SPEED_FULL;
    USBD_LL_SetSpeed((USBD_HandleTypeDef *)hpcd->pData, speed);
    USBD_LL_Reset((USBD_HandleTypeDef *)hpcd->pData);
}
void HAL_PCD_SuspendCallback(PCD_HandleTypeDef *hpcd) { USBD_LL_Suspend((USBD_HandleTypeDef *)hpcd->pData); }
void HAL_PCD_ResumeCallback(PCD_HandleTypeDef *hpcd) { USBD_LL_Resume((USBD_HandleTypeDef *)hpcd->pData); }
void HAL_PCD_ISOOUTIncompleteCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum) { USBD_LL_IsoOUTIncomplete((USBD_HandleTypeDef *)hpcd->pData, epnum); }
void HAL_PCD_ISOINIncompleteCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum) { USBD_LL_IsoINIncomplete((USBD_HandleTypeDef *)hpcd->pData, epnum); }
void HAL_PCD_ConnectCallback(PCD_HandleTypeDef *hpcd) { USBD_LL_DevConnected((USBD_HandleTypeDef *)hpcd->pData); }
void HAL_PCD_DisconnectCallback(PCD_HandleTypeDef *hpcd) { USBD_LL_DevDisconnected((USBD_HandleTypeDef *)hpcd->pData); }

USBD_StatusTypeDef USBD_LL_Init(USBD_HandleTypeDef *pdev)
{
    hpcd_USB_OTG_HS.Instance = USB_OTG_HS;
    hpcd_USB_OTG_HS.Init.dev_endpoints = 6;
    hpcd_USB_OTG_HS.Init.speed = PCD_SPEED_FULL;
    hpcd_USB_OTG_HS.Init.dma_enable = DISABLE;
    hpcd_USB_OTG_HS.Init.phy_itface = USB_OTG_EMBEDDED_PHY;
    hpcd_USB_OTG_HS.Init.Sof_enable = DISABLE;
    hpcd_USB_OTG_HS.Init.low_power_enable = DISABLE;
    hpcd_USB_OTG_HS.Init.lpm_enable = DISABLE;
    hpcd_USB_OTG_HS.Init.vbus_sensing_enable = DISABLE;
    hpcd_USB_OTG_HS.Init.use_dedicated_ep1 = DISABLE;
    hpcd_USB_OTG_HS.pData = pdev;
    pdev->pData = &hpcd_USB_OTG_HS;

    if (HAL_PCD_Init(&hpcd_USB_OTG_HS) != HAL_OK) return USBD_FAIL;

    HAL_PCDEx_SetRxFiFo(&hpcd_USB_OTG_HS, 0x80);
    HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_HS, 0, 0x40);
    HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_HS, 1, 0x40);
    return USBD_OK;
}

USBD_StatusTypeDef USBD_LL_DeInit(USBD_HandleTypeDef *pdev) { HAL_PCD_DeInit(pdev->pData); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_Start(USBD_HandleTypeDef *pdev) { HAL_PCD_Start(pdev->pData); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_Stop(USBD_HandleTypeDef *pdev) { HAL_PCD_Stop(pdev->pData); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_OpenEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr, uint8_t ep_type, uint16_t ep_mps) { HAL_PCD_EP_Open(pdev->pData, ep_addr, ep_mps, ep_type); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_CloseEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) { HAL_PCD_EP_Close(pdev->pData, ep_addr); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_FlushEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) { HAL_PCD_EP_Flush(pdev->pData, ep_addr); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_StallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) { HAL_PCD_EP_SetStall(pdev->pData, ep_addr); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_ClearStallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) { HAL_PCD_EP_ClrStall(pdev->pData, ep_addr); return USBD_OK; }
uint8_t USBD_LL_IsStallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr)
{
    PCD_HandleTypeDef *hpcd = pdev->pData;
    return (ep_addr & 0x80) ? hpcd->IN_ep[ep_addr & 0x7F].is_stall : hpcd->OUT_ep[ep_addr & 0x7F].is_stall;
}
USBD_StatusTypeDef USBD_LL_SetUSBAddress(USBD_HandleTypeDef *pdev, uint8_t dev_addr) { HAL_PCD_SetAddress(pdev->pData, dev_addr); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_Transmit(USBD_HandleTypeDef *pdev, uint8_t ep_addr, uint8_t *pbuf, uint32_t size) { HAL_PCD_EP_Transmit(pdev->pData, ep_addr, pbuf, size); return USBD_OK; }
USBD_StatusTypeDef USBD_LL_PrepareReceive(USBD_HandleTypeDef *pdev, uint8_t ep_addr, uint8_t *pbuf, uint32_t size) { HAL_PCD_EP_Receive(pdev->pData, ep_addr, pbuf, size); return USBD_OK; }
uint32_t USBD_LL_GetRxDataSize(USBD_HandleTypeDef *pdev, uint8_t ep_addr) { return HAL_PCD_EP_GetRxCount(pdev->pData, ep_addr); }
void USBD_LL_Delay(uint32_t Delay) { HAL_Delay(Delay); }

void *USBD_static_malloc(uint32_t size) { static uint32_t mem[64]; (void)size; return mem; }
void USBD_static_free(void *p) { (void)p; }
