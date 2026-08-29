#ifndef STM32H7xx_IT_H
#define STM32H7xx_IT_H

#ifdef __cplusplus
extern "C" {
#endif

void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);

void OTG_HS_IRQHandler(void);

/* One supervisory timer (velocity-profile recompute, 1kHz) */
void TIM6_DAC_IRQHandler(void);

/* Eight per-axis step-pulse timers, one per joint (see stepper.c for the
 * axis <-> timer instance table). */
void TIM2_IRQHandler(void);
void TIM3_IRQHandler(void);
void TIM4_IRQHandler(void);
void TIM5_IRQHandler(void);
void TIM7_IRQHandler(void);
void TIM16_IRQHandler(void);
void TIM17_IRQHandler(void);
void TIM15_IRQHandler(void);

/* Eight TMC2209 single-wire half-duplex UARTs (see tmc2209_uart.c). */
void USART1_IRQHandler(void);
void USART2_IRQHandler(void);
void USART3_IRQHandler(void);
void USART6_IRQHandler(void);
void UART4_IRQHandler(void);
void UART5_IRQHandler(void);
void UART7_IRQHandler(void);
void UART8_IRQHandler(void);

#ifdef __cplusplus
}
#endif

#endif /* STM32H7xx_IT_H */
