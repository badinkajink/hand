/*
  Several ReSkin 5X boards behind an I2C mux (PCA9548 or TCA9548A, 0x70 to 0x77) on the STEMMA QT port of an Adafruit
  QT Py ESP32-S3 or the Qwiic port of a SparkFun Thing Plus ESP32-C6, streamed as one record per sweep (owner,
  2026-10-25: three fingers on mux channels 0, 1 and 3 of the QT Py; one finger on channel 0 of a mux at 0x71 on the C6).

  Every board of this design answers at the same five addresses (U1 at 0x0C, U2 to U5 at 0x10 to 0x13), so each board
  has a mux channel of its own. At start-up the sketch finds the mux, then the chips on each of its eight channels, and
  the start-up text names the channels in streaming order ("Boards on mux channels: 0 1 3"). A sweep writes each
  board's channel and reads its chips in the order of firmware/reskin5x_stream (0x0C, 0x11, 0x12, 0x13, 0x10), with
  that sketch's settings: GAIN_SEL 1, RES 1, OSR 0, DIG_FILT 5, burst mode T X Y Z, so each chip converts every 7.0 ms.
  The bus runs at 400 kHz, the MLX90393's limit. Only the chips found at start-up are read: a transfer to an absent
  address costs about 19 ms in this core.

  Record: for each board in channel order and each of its five chips, t (degC) and x, y, z (uT) as little-endian
  float32, then "\r\n": 80 n + 2 bytes for n boards (242 for three, which anyskin reads with num_mags=15). A chip that
  was not found and a failed transfer give zeros; a read without a new conversion (0xFFFF) keeps the last values.
  The byte 'i' from the host prints the mux address, the channels and the record length again between two records,
  since the start-up text goes to whichever program opens the port first. host/reskin5x_mux.py sends it, then splits the
  stream into one 82-byte stream a board for host/wrench_viewer.py.
*/

#include <Wire.h>
#include <MLX90393.h>

#if defined(ARDUINO_ADAFRUIT_QTPY_ESP32S3_N4R2) || defined(ARDUINO_ADAFRUIT_QTPY_ESP32S3_NOPSRAM)
#define SENSOR_BUS Wire1  // QT Py ESP32-S3: STEMMA QT connector on SDA1 = GPIO41, SCL1 = GPIO40
#define SENSOR_SDA SDA1
#define SENSOR_SCL SCL1
#else
#define SENSOR_BUS Wire   // SparkFun Thing Plus ESP32-C6 (2026-10-25): Qwiic connector on SDA = GPIO6, SCL = GPIO7, a bus
#define SENSOR_SDA SDA    // shared with the board's LiPo fuel gauge at 0x36; the variant's SDA and SCL on other boards
#define SENSOR_SCL SCL
#endif

static const uint8_t GAIN_SEL = 1;
static const uint8_t RES_XYZ  = 1;
static const uint8_t DIG_FILT = 5;
static const uint8_t OSR      = 0;

static const uint8_t CHIPS = 5;
static const uint8_t ADDR[CHIPS] = {0x0C, 0x11, 0x12, 0x13, 0x10};
static const uint8_t MAX_BOARDS = 8;

uint8_t muxAddr = 0;
uint8_t channel[MAX_BOARDS];
uint8_t nBoards = 0;
MLX90393 mlx[MAX_BOARDS][CHIPS];
bool live[MAX_BOARDS][CHIPS];
MLX90393::txyz dataBuf[MAX_BOARDS][CHIPS];

void printInfo();

bool ack(uint8_t addr) {
  SENSOR_BUS.beginTransmission(addr);
  return SENSOR_BUS.endTransmission() == 0;
}

bool selectChannel(uint8_t ch) {
  SENSOR_BUS.beginTransmission(muxAddr);
  SENSOR_BUS.write((uint8_t)(1u << ch));
  return SENSOR_BUS.endTransmission() == 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }
  SENSOR_BUS.begin(SENSOR_SDA, SENSOR_SCL);
  SENSOR_BUS.setClock(400000);
  delay(10);

  for (uint8_t a = 0x70; a <= 0x77 && !muxAddr; ++a) if (ack(a)) muxAddr = a;
  if (!muxAddr) {
    Serial.println(F("[ERROR] no I2C mux at 0x70 to 0x77 on the STEMMA QT port"));
    while (true) { delay(1000); }
  }
  Serial.print(F("Mux at 0x")); Serial.println(muxAddr, HEX);

  for (uint8_t ch = 0; ch < 8 && nBoards < MAX_BOARDS; ++ch) {
    if (!selectChannel(ch)) continue;
    delay(2);
    uint8_t found = 0;
    bool here[CHIPS];
    for (uint8_t i = 0; i < CHIPS; ++i) found += (here[i] = ack(ADDR[i]));
    Serial.print(F("Channel ")); Serial.print(ch); Serial.print(F(": ")); Serial.print(found); Serial.println(F(" chips"));
    if (!found) continue;
    uint8_t b = nBoards++;
    channel[b] = ch;
    for (uint8_t i = 0; i < CHIPS; ++i) {
      live[b][i] = here[i];
      dataBuf[b][i] = {0, 0, 0, 0};
      if (!here[i]) continue;
      byte status = mlx[b][i].begin(ADDR[i], -1, SENSOR_BUS);
      mlx[b][i].setGainSel(GAIN_SEL);
      mlx[b][i].setResolution(RES_XYZ, RES_XYZ, RES_XYZ);
      mlx[b][i].setOverSampling(OSR);
      mlx[b][i].setDigitalFiltering(DIG_FILT);
      uint8_t g = 0, rx = 0, ry = 0, rz = 0, df = 0, osr = 0;
      mlx[b][i].getGainSel(g);
      mlx[b][i].getResolution(rx, ry, rz);
      mlx[b][i].getDigitalFiltering(df);
      mlx[b][i].getOverSampling(osr);
      Serial.print(F("  MLX @0x")); Serial.print(ADDR[i], HEX);
      Serial.print(F(" status=")); Serial.print(status, HEX);
      Serial.print(F(" gain_sel=")); Serial.print(g);
      Serial.print(F(" res=")); Serial.print(rx); Serial.print(ry); Serial.print(rz);
      Serial.print(F(" dig_filt=")); Serial.print(df);
      Serial.print(F(" osr=")); Serial.println(osr);
      mlx[b][i].startBurst(0xF);
    }
  }
  printInfo();
  if (!nBoards) { while (true) { delay(1000); } }
}

void printInfo() {
  Serial.print(F("Mux at 0x")); Serial.println(muxAddr, HEX);
  Serial.print(F("Boards on mux channels: "));
  for (uint8_t b = 0; b < nBoards; ++b) { Serial.print(channel[b]); Serial.print(' '); }
  Serial.println();
  Serial.print(F("Record bytes: ")); Serial.println(80 * nBoards + 2);
}

void loop() {
  while (Serial.available()) if (Serial.read() == 'i') printInfo();
  for (uint8_t b = 0; b < nBoards; ++b) {
    if (!selectChannel(channel[b])) {
      for (uint8_t i = 0; i < CHIPS; ++i) dataBuf[b][i] = {0, 0, 0, 0};
      continue;
    }
    for (uint8_t i = 0; i < CHIPS; ++i) {
      if (!live[b][i]) continue;
      MLX90393::txyzRaw raw;
      if (mlx[b][i].readMeasurement(0xF, raw) == MLX90393::STATUS_ERROR) dataBuf[b][i] = {0, 0, 0, 0};
      else if (raw.t != 0xFFFF) dataBuf[b][i] = mlx[b][i].convertRaw(raw);
    }
  }
  Serial.write((uint8_t*)dataBuf, sizeof(MLX90393::txyz) * CHIPS * nBoards);
  Serial.write((const uint8_t*)"\r\n", 2);
}
