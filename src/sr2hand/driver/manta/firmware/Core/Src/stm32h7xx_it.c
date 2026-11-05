#include "stm32h7xx_it.h"
#include "main.h"
#include "stepper.h"

extern PCD_HandleTypeDef hpcd_USB_OTG_HS;

void NMI_Handler(void) { while (1) { } }
void HardFault_Handler(void) { while (1) { } }
void MemManage_Handler(void) { while (1) { } }
void BusFault_Handler(void) { while (1) { } }
void UsageFault_Handler(void) { while (1) { } }
void SVC_Handler(void) { }
void DebugMon_Handler(void) { }
void PendSV_Handler(void) { }

void SysTick_Handler(void)
{
    HAL_IncTick();
}

void OTG_HS_IRQHandler(void) { HAL_PCD_IRQHandler(&hpcd_USB_OTG_HS); }

/* --- Supervisory 1kHz tick (TIM6) --- */
void TIM6_DAC_IRQHandler(void)
{
    if (TIM6->SR & TIM_SR_UIF) {
        TIM6->SR &= ~TIM_SR_UIF;
        Stepper_SupervisorTick();
    }
}

/* --- Per-axis step-pulse timers: clear the update flag directly on the
 * peripheral rather than going through a HAL_TIM_IRQHandler + registered
 * callback for each -- fewer moving parts for 8 nearly-identical ISRs. --- */
static inline void axis_timer_irq(TIM_TypeDef *tim, uint8_t axis)
{
    if (tim->SR & TIM_SR_UIF) {
        tim->SR &= ~TIM_SR_UIF;
        Stepper_TimerIRQ(axis);
    }
}

void TIM2_IRQHandler(void)  { axis_timer_irq(TIM2, 0); }
void TIM3_IRQHandler(void)  { axis_timer_irq(TIM3, 1); }
void TIM4_IRQHandler(void)  { axis_timer_irq(TIM4, 2); }
void TIM5_IRQHandler(void)  { axis_timer_irq(TIM5, 3); }
void TIM7_IRQHandler(void)  { axis_timer_irq(TIM7, 4); }
void TIM16_IRQHandler(void) { axis_timer_irq(TIM16, 5); }
void TIM17_IRQHandler(void) { axis_timer_irq(TIM17, 6); }
void TIM15_IRQHandler(void) { axis_timer_irq(TIM15, 7); }

/* --- Sensorless-homing DIAG pins (TMC2209 StallGuard4) -- J0/PF4, J1/PF3,
 * J2/PF2, J3/PF1, see stepper.c's axis_diag[] table. Same direct-register
 * style as axis_timer_irq above: check/clear this pin's own EXTI pending
 * bit (not shared with anything else -- EXTI1-4 each have a standalone
 * vector on this chip) and hand off to Stepper_DiagIRQ, which no-ops if
 * that axis isn't actually homing right now. */
void EXTI1_IRQHandler(void)
{
    if (EXTI->PR1 & EXTI_PR1_PR1) {
        EXTI->PR1 = EXTI_PR1_PR1; /* write-1-to-clear */
        Stepper_DiagIRQ(3);
    }
}

void EXTI2_IRQHandler(void)
{
    if (EXTI->PR1 & EXTI_PR1_PR2) {
        EXTI->PR1 = EXTI_PR1_PR2; /* write-1-to-clear */
        Stepper_DiagIRQ(2);
    }
}

void EXTI3_IRQHandler(void)
{
    if (EXTI->PR1 & EXTI_PR1_PR3) {
        EXTI->PR1 = EXTI_PR1_PR3; /* write-1-to-clear */
        Stepper_DiagIRQ(1);
    }
}

void EXTI4_IRQHandler(void)
{
    if (EXTI->PR1 & EXTI_PR1_PR4) {
        EXTI->PR1 = EXTI_PR1_PR4; /* write-1-to-clear */
        Stepper_DiagIRQ(0);
    }
}

/* TMC2209 UARTs are driven purely by blocking HAL_UART_Transmit/Receive
 * calls from protocol.c command handlers (current/microstep config only
 * happens occasionally, not in a hot path), so these vectors just need to
 * exist for HAL_UART_IRQHandler's polling-mode-adjacent bookkeeping. If a
 * future revision moves to interrupt-driven UART, wire these to
 * HAL_UART_IRQHandler(&tmc_huart[n]) with a per-instance handle lookup. */
void USART1_IRQHandler(void) { }
void USART2_IRQHandler(void) { }
void USART3_IRQHandler(void) { }
void USART6_IRQHandler(void) { }
void UART4_IRQHandler(void)  { }
void UART5_IRQHandler(void)  { }
void UART7_IRQHandler(void)  { }
void UART8_IRQHandler(void)  { }
