// Two measurements on one ReSkin 5X board at the firmware's settings (GAIN_SEL 1, RES 1, OSR 0, DIG_FILT 5, burst T+XYZ):
// (1) the conversion period of the chip at 0x0C, from the times its x, y and z values change while it is read back to
//     back for 10 s; intervals between successive changes of one axis that lie in 4-10 ms are one conversion apart;
// (2) the time of a write to the absent mux address 0x70 (NACKed) at Wire's default 50 ms timeout and at 1 ms.
#include <Wire.h>
#include <MLX90393.h>
#define BUS Wire1
MLX90393 mlx;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }
  BUS.begin(SDA1, SCL1);
  BUS.setClock(400000);
  mlx.begin(0x0C, -1, BUS);
  mlx.setGainSel(1);
  mlx.setResolution(1, 1, 1);
  mlx.setOverSampling(0);
  mlx.setDigitalFiltering(5);
  mlx.startBurst(0xF);
}

void loop() {
  MLX90393::txyz d, prev = {0, 0, 0, 0};
  uint32_t last[3] = {0, 0, 0}, cnt[3] = {0, 0, 0}, changes[3] = {0, 0, 0}, reads = 0;
  double sum[3] = {0, 0, 0};
  uint32_t t0 = millis();
  while (millis() - t0 < 10000) {
    mlx.readBurstData(d);
    uint32_t t = micros();
    ++reads;
    float v[3] = {d.x, d.y, d.z}, p[3] = {prev.x, prev.y, prev.z};
    for (int k = 0; k < 3; ++k) {
      if (v[k] != p[k]) {
        ++changes[k];
        if (last[k]) {
          uint32_t dt = t - last[k];
          if (dt > 4000 && dt < 10000) { sum[k] += dt; ++cnt[k]; }
        }
        last[k] = t;
      }
    }
    prev = d;
  }
  Serial.printf("CONV reads=%u read_us=%.1f changes=%u,%u,%u period_us=%.1f,%.1f,%.1f n=%u,%u,%u\r\n", reads,
                10e6 / reads, changes[0], changes[1], changes[2], sum[0] / cnt[0], sum[1] / cnt[1], sum[2] / cnt[2],
                cnt[0], cnt[1], cnt[2]);
  for (int pass = 0; pass < 2; ++pass) {
    BUS.setTimeOut(pass == 0 ? 50 : 1);
    uint32_t a = micros();
    uint8_t err = 0;
    for (int k = 0; k < 10; ++k) { BUS.beginTransmission(0x70); BUS.write(1); err = BUS.endTransmission(); }
    float us = (micros() - a) / 10.0f;
    MLX90393::txyz q;
    uint8_t st = mlx.readBurstData(q);
    Serial.printf("NACK timeout_ms=%u write_us=%.1f err=%u next_read_status=%u x=%.1f\r\n", BUS.getTimeOut(), us, err, st, q.x);
  }
  BUS.setTimeOut(50);
}
