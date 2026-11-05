/*
  Scan an I2C mux's eight channels at several bus clocks, every two seconds, and print the addresses that answer on
  each (lab, 2026-11-04: on the SR2 hand the C6's start-up scan found five chips on channel 3 and none on channels 0
  and 1, whose fingertips have longer cables). One line a clock: "<clock> Hz  ch0: <addresses>  ch1: ...". The sketch
  only addresses chips (an empty write each); it reads nothing and writes no register. Put reskin5x_mux_stream back
  afterwards.
*/

#include <Wire.h>

#if defined(ARDUINO_ADAFRUIT_QTPY_ESP32S3_N4R2) || defined(ARDUINO_ADAFRUIT_QTPY_ESP32S3_NOPSRAM)
#define SENSOR_BUS Wire1
#define SENSOR_SDA SDA1
#define SENSOR_SCL SCL1
#else
#define SENSOR_BUS Wire
#define SENSOR_SDA SDA
#define SENSOR_SCL SCL
#endif

static const uint32_t CLOCKS[] = {400000, 100000, 50000, 20000};
uint8_t muxAddr = 0;

bool ack(uint8_t addr) {
  SENSOR_BUS.beginTransmission(addr);
  return SENSOR_BUS.endTransmission() == 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }
  SENSOR_BUS.begin(SENSOR_SDA, SENSOR_SCL);
  SENSOR_BUS.setTimeOut(5);
}

void loop() {
  for (uint32_t clk : CLOCKS) {
    SENSOR_BUS.setClock(clk);
    delay(2);
    muxAddr = 0;
    for (uint8_t a = 0x70; a <= 0x77 && !muxAddr; ++a) if (ack(a)) muxAddr = a;
    Serial.print(clk); Serial.print(F(" Hz mux 0x")); Serial.print(muxAddr, HEX);
    if (!muxAddr) { Serial.println(F(" none")); continue; }
    for (uint8_t ch = 0; ch < 8; ++ch) {
      SENSOR_BUS.beginTransmission(muxAddr);
      SENSOR_BUS.write((uint8_t)(1u << ch));
      bool sel = SENSOR_BUS.endTransmission() == 0;
      delay(5);
      Serial.print(F("  ch")); Serial.print(ch); Serial.print(sel ? ':' : '!');
      uint8_t n = 0;
      for (uint8_t a = 0x08; a < 0x78; ++a) {
        if (a == muxAddr || a == 0x36) continue;   // the mux itself; the C6's fuel gauge on the controller side
        if (ack(a)) { Serial.print(' '); Serial.print(a, HEX); ++n; }
      }
      if (!n) Serial.print(F(" -"));
    }
    SENSOR_BUS.beginTransmission(muxAddr);
    SENSOR_BUS.write((uint8_t)0);
    SENSOR_BUS.endTransmission();
    Serial.println();
  }
  Serial.println();
  delay(2000);
}
