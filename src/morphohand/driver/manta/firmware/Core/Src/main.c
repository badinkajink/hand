#include "main.h"
#include "usb_device.h"
#include "stepper.h"
#include "tmc2209_uart.h"
#include "tmc5160_spi.h"
#include "protocol.h"

static void SystemClock_Config(void);

/*
 * SystemClock_Config: HSE 25MHz -> PLL1 -> 550MHz SYSCLK, VOS0.
 *
 * PLL1 math (see docs/bringup.md for how to verify this on real hardware
 * before trusting anything downstream of it -- this is the single highest-
 * consequence guess in this firmware if wrong):
 *   DIVM1 = 5   => PLL1 input = 25MHz / 5 = 5MHz  (falls in the 4-8MHz
 *                  "RGE2" input range for the wide VCO option)
 *   DIVN1 = 110 => VCO = 5MHz * 110 = 550MHz  (within wide-VCO 192-960MHz)
 *   DIVP1 = 1   => SYSCLK = VCO / 1 = 550MHz
 * No fractional PLL needed (FRACN=0) since 5 * 110 divides evenly.
 *
 * AHB/APBx prescalers all /2 from the 550MHz CPU clock, matching the
 * standard VOS0 H723 pattern (AHB max 275MHz, APBx max ~137.5MHz):
 *   HPRE=/2 (AHB=275MHz), D1PPRE=/2, D2PPRE1=/2, D2PPRE2=/2, D3PPRE=/2
 *
 * This also means every timer on APB1 (TIM2-7,12-14) and APB2 (TIM1,8,15-17)
 * gets a 2x-multiplied 275MHz kernel clock (STM32H7 doubles a timer's kernel
 * clock relative to its APB bus whenever that bus's prescaler != 1) -- see
 * TIMER_KERNEL_CLOCK_HZ in stepper.c, which depends on this.
 *
 * Flash latency: FLASH_LATENCY_4, per RM0468's VOS0/AXI-clock wait-state
 * table for a clock in the 275MHz AXI range.
 */
static void SystemClock_Config(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) { }

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    osc.HSEState = RCC_HSE_ON; /* crystal, not an active oscillator module -- flip to
                                   RCC_HSE_BYPASS if the clock never locks on real hw */
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM = 5;
    osc.PLL.PLLN = 110;
    osc.PLL.PLLP = 1;
    osc.PLL.PLLQ = 4;
    osc.PLL.PLLR = 2;
    osc.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;   /* 4-8MHz input range */
    osc.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;   /* wide VCO: 192-960MHz output */
    osc.PLL.PLLFRACN = 0;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) Error_Handler();

    clk.ClockType = RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_D1PCLK1 |
                    RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.SYSCLKDivider = RCC_SYSCLK_DIV1;
    clk.AHBCLKDivider = RCC_HCLK_DIV2;
    clk.APB3CLKDivider = RCC_APB3_DIV2;
    clk.APB1CLKDivider = RCC_APB1_DIV2;
    clk.APB2CLKDivider = RCC_APB2_DIV2;
    clk.APB4CLKDivider = RCC_APB4_DIV2;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_4) != HAL_OK) Error_Handler();

    /* USB kernel clock: dedicated HSI48 RC oscillator + CRS trimming against
     * USB start-of-frame, rather than deriving 48MHz from PLL1Q/PLL3 --
     * simpler and avoids adding more constraints to the PLL1 math above. */
    __HAL_RCC_HSI48_ENABLE();
    while (__HAL_RCC_GET_FLAG(RCC_FLAG_HSI48RDY) == RESET) { }
    RCC_PeriphCLKInitTypeDef pclk = {0};
    pclk.PeriphClockSelection = RCC_PERIPHCLK_USB;
    pclk.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;
    if (HAL_RCCEx_PeriphCLKConfig(&pclk) != HAL_OK) Error_Handler();

    RCC_CRSInitTypeDef crs = {0};
    crs.Prescaler = RCC_CRS_SYNC_DIV1;
    crs.Source = RCC_CRS_SYNC_SOURCE_USB2;
    crs.Polarity = RCC_CRS_SYNC_POLARITY_RISING;
    crs.ReloadValue = __HAL_RCC_CRS_RELOADVALUE_CALCULATE(48000000, 1000);
    crs.ErrorLimitValue = 34;
    crs.HSI48CalibrationValue = 32;
    __HAL_RCC_CRS_CLK_ENABLE();
    HAL_RCCEx_CRSConfig(&crs);
}

int main(void)
{
    HAL_Init();
    SystemClock_Config();

    Stepper_Init();
    TMC_Init();
    TMC5160_Init();
    Protocol_Init();
    MX_USB_DEVICE_Init();

    while (1) {
        Protocol_Poll();
        Stepper_HomingPollTick();
        TMC5160_KickPollTick();
    }
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) {
        /* No confirmed status LED pin for V2.0 yet (see main.h) -- if you
         * add one during bring-up, blink it here so a clock/USB init
         * failure is visible without a debug probe attached. */
    }
}
