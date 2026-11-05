/*
  Scan an I2C mux's eight channels at several bus clocks, every two seconds, and print the addresses that answer on
  each (lab, 2026-11-04: on the SR2 hand the C6's start-up scan found five chips on channel 3 and none on channels 0
  and 1, whose fingertips have longer cables). One line a clock: "<clock> Hz mux 0x71  ch0: <addresses>  ch1: ...".
  A second mux on a channel shows as "ch1: mux 70 [ch0: C 10 11 12 13]", the addresses on each of its channels that
  answer (lab, 2026-11-05: the index's board on channel 0 of a mux at 0x70, cabled to channel 1 of the mux at 0x71, to
  lengthen its cable). A second mux at the first mux's own address answers with it and takes its writes; on a channel
  where nothing else answers, the writes ch, s, ch leave such a mux on its channel s, and the chips found there show as
  "twin [ch0: ...]". The sketch writes only the muxes' channel registers; it addresses chips with an empty write each
  and reads nothing. Put reskin5x_mux_stream back afterwards.
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
static const uint8_t CHIPS[] = {0x0C, 0x10, 0x11, 0x12, 0x13};

bool ack(uint8_t addr) {
  SENSOR_BUS.beginTransmission(addr);
  return SENSOR_BUS.endTransmission() == 0;
}

bool setMux(uint8_t mux, uint8_t mask) {
  SENSOR_BUS.beginTransmission(mux);
  SENSOR_BUS.write(mask);
  return SENSOR_BUS.endTransmission() == 0;
}

// The addresses that answer, other than the muxes in skip and the C6's fuel gauge (0x36, on the controller side).
uint8_t scan(uint8_t* out, const uint8_t* skip, uint8_t nSkip) {
  uint8_t n = 0;
  for (uint8_t a = 0x08; a < 0x78; ++a) {
    bool s = a == 0x36;
    for (uint8_t k = 0; k < nSkip; ++k) s |= a == skip[k];
    if (!s && ack(a)) out[n++] = a;
  }
  return n;
}

void printList(const uint8_t* a, uint8_t n) {
  for (uint8_t i = 0; i < n; ++i) { Serial.print(' '); Serial.print(a[i], HEX); }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }
  SENSOR_BUS.begin(SENSOR_SDA, SENSOR_SCL);
  SENSOR_BUS.setTimeOut(5);
}

void loop() {
  uint8_t found[112];
  for (uint32_t clk : CLOCKS) {
    SENSOR_BUS.setClock(clk);
    delay(2);
    // a mux behind another's channel answers only while that channel is open: close every mux that answers, twice,
    // and the first mux is the lowest address that still answers
    for (uint8_t pass = 0; pass < 2; ++pass)
      for (uint8_t a = 0x70; a <= 0x77; ++a) if (ack(a)) setMux(a, 0);
    uint8_t mux = 0;
    for (uint8_t a = 0x70; a <= 0x77 && !mux; ++a) if (ack(a)) mux = a;
    Serial.print(clk); Serial.print(F(" Hz mux 0x")); Serial.print(mux, HEX);
    if (!mux) { Serial.println(F(" none")); continue; }
    for (uint8_t ch = 0; ch < 8; ++ch) {
      bool sel = setMux(mux, 1u << ch);
      delay(5);
      Serial.print(F("  ch")); Serial.print(ch); Serial.print(sel ? ':' : '!');
      uint8_t skip[9] = {mux}, nSkip = 1;
      for (uint8_t a = 0x70; a <= 0x77; ++a) if (a != mux && ack(a)) { setMux(a, 0); skip[nSkip++] = a; }
      uint8_t n = scan(found, skip, nSkip);
      printList(found, n);
      for (uint8_t k = 1; k < nSkip; ++k) {
        Serial.print(F(" mux ")); Serial.print(skip[k], HEX); Serial.print(F(" ["));
        bool any = false;
        for (uint8_t s = 0; s < 8; ++s) {
          setMux(skip[k], 1u << s);
          delay(5);
          uint8_t m = scan(found, skip, nSkip);
          if (!m) continue;
          Serial.print(any ? F(" ch") : F("ch")); Serial.print(s); Serial.print(':'); printList(found, m);
          any = true;
        }
        setMux(skip[k], 0);
        Serial.print(any ? F("]") : F("-]"));
      }
      if (n || nSkip > 1) continue;
      bool any = false;
      for (uint8_t s = 0; s < 8; ++s) {
        setMux(mux, 1u << ch); setMux(mux, 1u << s); setMux(mux, 1u << ch);
        delay(5);
        uint8_t m = 0;
        for (uint8_t a : CHIPS) if (ack(a)) found[m++] = a;
        if (!m) continue;
        Serial.print(any ? F(" ch") : F(" twin [ch")); Serial.print(s); Serial.print(':'); printList(found, m);
        any = true;
      }
      Serial.print(any ? F("]") : F(" -"));
    }
    setMux(mux, 0);
    Serial.println();
  }
  Serial.println();
  delay(2000);
}
