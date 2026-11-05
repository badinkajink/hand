// Timing of one sweep of three ReSkin 5X boards behind a PCA9548 mux, emulated on one board: per board, an ACKed
// address-only write to 0x0C in place of the mux's channel write (one byte shorter, 22.5 us at 400 kHz) and five burst
// reads; then a 242-byte record over USB. Every 200 sweeps prints a STAT line, with 20 NACKed writes to 0x70 timed.
#include <Wire.h>
#include <MLX90393.h>
#define BUS Wire1
MLX90393 mlx[5];
MLX90393::txyz buf[15];
const uint8_t ADDR[5] = {0x0C, 0x11, 0x12, 0x13, 0x10};
uint8_t filler[242];
uint32_t n = 0, t_mux = 0, t_read = 0, t_usb = 0, t_max = 0, nack = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }
  BUS.begin(SDA1, SCL1);
  BUS.setClock(400000);
  for (uint8_t i = 0; i < 5; ++i) {
    mlx[i].begin(ADDR[i], -1, BUS);
    mlx[i].setGainSel(1);
    mlx[i].setResolution(1, 1, 1);
    mlx[i].setOverSampling(0);
    mlx[i].setDigitalFiltering(5);
    mlx[i].startBurst(0xF);
  }
  memset(filler, '#', sizeof(filler));
  filler[240] = '\r'; filler[241] = '\n';
}

void loop() {
  uint32_t t0 = micros(), tm = 0, tr = 0;
  for (uint8_t b = 0; b < 3; ++b) {
    uint32_t a = micros();
    BUS.beginTransmission(0x0C);
    nack += BUS.endTransmission() != 0;
    uint32_t c = micros();
    tm += c - a;
    for (uint8_t i = 0; i < 5; ++i) mlx[i].readBurstData(buf[5 * b + i]);
    tr += micros() - c;
  }
  uint32_t t1 = micros();
  Serial.write(filler, sizeof(filler));
  uint32_t t2 = micros();
  t_mux += tm; t_read += tr; t_usb += t2 - t1;
  if (t2 - t0 > t_max) t_max = t2 - t0;
  if (++n == 200) {
    uint32_t p0 = micros();
    for (int k = 0; k < 20; ++k) { BUS.beginTransmission(0x70); BUS.write(1); BUS.endTransmission(); }
    float probe = (micros() - p0) / 20.0f;
    Serial.printf("STAT sweeps=%u mux_us=%.1f reads_us=%.1f per_read_us=%.1f usb_us=%.1f sweep_us=%.1f max_sweep_us=%u nack=%u nacked_write_us=%.1f x0=%.1f\r\n",
                  n, t_mux / (3.0f * n), t_read / (float)n, t_read / (15.0f * n), t_usb / (float)n,
                  (t_mux + t_read + t_usb) / (float)n, t_max, nack, probe, buf[0].x);
    n = t_mux = t_read = t_usb = t_max = nack = 0;
  }
}
