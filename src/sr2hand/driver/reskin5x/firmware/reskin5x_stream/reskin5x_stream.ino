/*
  ReSkin 5X magnetometer board streaming from an Adafruit QT Py ESP32-S3 (4 MB flash, 2 MB PSRAM; the 8 MB no-PSRAM
  QT Py has the same pins)

  Port of eFlesh's arduino/5X_eflesh_stream/5X_eflesh_stream.ino (external/eFlesh; V. Pattabiraman, after ReSkin and
  AnySkin), on the MLX90393 library of T. Yapo and T. Hellebrekers (external/eFlesh/arduino/arduino-MLX90393). Changes:
    - The STEMMA QT connector is the board's second I2C bus, Wire1 on SDA1 = GPIO41 and SCL1 = GPIO40. Wire drives the
      castellated SDA/SCL pads (GPIO7, GPIO6), so the original sketch finds no chips through the Qwiic cable.
    - Serial is the TinyUSB CDC port (USB CDC On Boot, the board's default); SERIAL_PORT_USBVIRTUAL is a SAMD name.
    - Gain and resolution are set for the concept-b pad. Its largest field component at a chip is 27.5 mT
      (docs/experiments/20261019-v2_tip_fe_checks/sensing.json, case b_v2); GAIN_SEL 1 with RES 1 gives a 38.4 mT range,
      1.4 times that. DIG_FILT 5 with OSR 0 converts T, X, Y and Z in 7.0 ms (143 Hz), the setting of the noise model in
      scripts/mlx90393_model.py.
    - The start-up text reports each chip's register values as read back.
    - Only the chips found at start-up are read. The library leaves an object without begin() with a null bus, which
      the original dereferences when fewer than five chips answer; here their records are zeros, as is the record of a
      chip whose transfer fails. A transfer to an absent address costs about 19 ms in this core, at any Wire timeout. The
      chip's status byte is not a validity flag: in burst mode its error bit was set on 93 % of reads of valid data.
    - Until its first conversion, 5 to 222 ms after start-up, and on one read in about 18,700 after that, a chip
      returns 0xFFFF in every field; such a read keeps the chip's last values, which are zeros at start-up.

  Record: for each chip in streaming order, t (degC) and x, y, z (uT) as little-endian float32, then "\r\n": 82 bytes.
  The anyskin package (AnySkinProcess(num_mags=5)) and external/eFlesh/visualizer/viz_eflesh.py parse it unchanged.
*/

#include <Wire.h>
#include <MLX90393.h>

#define SENSOR_BUS Wire1  // STEMMA QT connector

// MLX90393 settings (HALLCONF 0xC from the library's begin())
static const uint8_t GAIN_SEL = 1;
static const uint8_t RES_XYZ  = 1;
static const uint8_t DIG_FILT = 5;
static const uint8_t OSR      = 0;

// MLX90393 objects and data buffers
static const uint8_t NUM_SENSORS = 5;
MLX90393 mlx[NUM_SENSORS];
MLX90393::txyz dataBuf[NUM_SENSORS];
uint8_t activeCount = 0;  // chips initialized in setup(); the others have no bus and stream zeros

// Known sequences
const uint8_t TARGETS_ALL_CONSEC[NUM_SENSORS] = {0x0C, 0x0D, 0x0E, 0x0F, 0x10};
const uint8_t TARGETS_WHITE_SET[NUM_SENSORS]   = {0x0C, 0x10, 0x11, 0x12, 0x13};

// Forward decls
void scanI2C(uint8_t* found, uint8_t& count);
void chooseOrderedAddresses(const uint8_t* found, uint8_t count, uint8_t* ordered, uint8_t& orderedCount);
bool hasExactSet(const uint8_t* found, uint8_t count, const uint8_t* pattern);
void sortAscending(uint8_t* arr, uint8_t n);

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(5); }

  SENSOR_BUS.begin(SDA1, SCL1);
  SENSOR_BUS.setClock(400000);
  delay(10);

  uint8_t found[16] = {0};
  uint8_t foundCount = 0;
  scanI2C(found, foundCount);

  Serial.println(F("I2C scan complete."));
  Serial.print(F("Found MLX candidates: "));
  for (uint8_t i = 0; i < foundCount; ++i) {
    Serial.print("0x"); Serial.print(found[i], HEX); Serial.print(" ");
  }
  Serial.println();

  uint8_t ordered[NUM_SENSORS] = {0};
  uint8_t orderedCount = 0;
  chooseOrderedAddresses(found, foundCount, ordered, orderedCount);

  if (orderedCount != NUM_SENSORS) {
    Serial.println(F("[WARN] Did not find exactly 5 target MLX addresses. Proceeding with what was found."));
  }

  // Initialize sensors in the chosen order
  for (uint8_t i = 0; i < orderedCount; ++i) {
    byte status = mlx[i].begin(ordered[i], -1, SENSOR_BUS);
    mlx[i].setGainSel(GAIN_SEL);
    mlx[i].setResolution(RES_XYZ, RES_XYZ, RES_XYZ);
    mlx[i].setOverSampling(OSR);
    mlx[i].setDigitalFiltering(DIG_FILT);

    uint8_t g = 0, rx = 0, ry = 0, rz = 0, df = 0, osr = 0;
    mlx[i].getGainSel(g);
    mlx[i].getResolution(rx, ry, rz);
    mlx[i].getDigitalFiltering(df);
    mlx[i].getOverSampling(osr);
    Serial.print(F("Init MLX[")); Serial.print(i); Serial.print(F("] @0x"));
    Serial.print(ordered[i], HEX);
    Serial.print(F(" status=")); Serial.print(status, HEX);
    Serial.print(F(" gain_sel=")); Serial.print(g);
    Serial.print(F(" res=")); Serial.print(rx); Serial.print(ry); Serial.print(rz);
    Serial.print(F(" dig_filt=")); Serial.print(df);
    Serial.print(F(" osr=")); Serial.println(osr);

    // Start burst mode (Temp + X + Y + Z) = 0xF
    mlx[i].startBurst(0xF);
  }

  // If fewer than 5 were found, zero-fill the remaining buffers so the binary record stays a fixed length
  for (uint8_t i = orderedCount; i < NUM_SENSORS; ++i) {
    dataBuf[i] = {0, 0, 0, 0};
  }
  activeCount = orderedCount;

  Serial.print(F("Streaming order: "));
  for (uint8_t i = 0; i < orderedCount; ++i) {
    Serial.print("0x"); Serial.print(ordered[i], HEX); Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Read in the same order the sensors were initialized
  for (uint8_t i = 0; i < activeCount; ++i) {
    // readBurstData would convert an unfilled buffer after a failed transfer, and the status byte's error bit, which
    // it reports as a failure, is set on most burst-mode reads of valid data; only a failed transfer zeroes the record.
    // A read that returns 0xFFFF as temperature (451.8 degC) has no conversion in it and keeps the last values.
    MLX90393::txyzRaw raw;
    if (mlx[i].readMeasurement(0xF, raw) == MLX90393::STATUS_ERROR) dataBuf[i] = {0, 0, 0, 0};
    else if (raw.t != 0xFFFF) dataBuf[i] = mlx[i].convertRaw(raw);
  }

  // Write binary records in-order
  for (uint8_t i = 0; i < NUM_SENSORS; ++i) {
    Serial.write((uint8_t*)&dataBuf[i], sizeof(dataBuf[i]));
  }
  Serial.println();

  // adjust for desired sample rate
  delayMicroseconds(500);
}

// Scan all I2C addresses to find MLX90393 sensors
void scanI2C(uint8_t* found, uint8_t& count) {
  count = 0;
  for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
    SENSOR_BUS.beginTransmission(addr);
    uint8_t err = SENSOR_BUS.endTransmission();
    if (err == 0) {
      found[count++] = addr;
      if (count >= 16) break;
    }
  }
}

void chooseOrderedAddresses(const uint8_t* found, uint8_t count, uint8_t* ordered, uint8_t& orderedCount) {
  orderedCount = 0;

  // Copy to a temp for set checks (exact set of 5)
  if (count >= NUM_SENSORS) {
    // Check for the two exact sets
    if (hasExactSet(found, count, TARGETS_ALL_CONSEC)) {
      for (uint8_t i = 0; i < NUM_SENSORS; ++i) ordered[i] = TARGETS_ALL_CONSEC[i];
      orderedCount = NUM_SENSORS;
      return;
    }
    if (hasExactSet(found, count, TARGETS_WHITE_SET)) {
      // custom order: C,11,12,13,10
      ordered[0] = 0x0C; ordered[1] = 0x11; ordered[2] = 0x12; ordered[3] = 0x13; ordered[4] = 0x10;
      orderedCount = NUM_SENSORS;
      return;
    }
  }

  uint8_t tmp[16];
  uint8_t n = count;
  if (n > 16) n = 16;
  for (uint8_t i = 0; i < n; ++i) tmp[i] = found[i];
  sortAscending(tmp, n);

  orderedCount = (n >= NUM_SENSORS) ? NUM_SENSORS : n;
  for (uint8_t i = 0; i < orderedCount; ++i) ordered[i] = tmp[i];
}

bool hasExactSet(const uint8_t* found, uint8_t count, const uint8_t* pattern) {
  uint8_t hits = 0;
  for (uint8_t p = 0; p < NUM_SENSORS; ++p) {
    bool present = false;
    for (uint8_t i = 0; i < count; ++i) {
      if (found[i] == pattern[p]) { present = true; break; }
    }
    if (present) ++hits;
  }
  return (hits == NUM_SENSORS);
}

void sortAscending(uint8_t* arr, uint8_t n) {
  for (uint8_t i = 0; i < n; ++i) {
    for (uint8_t j = i + 1; j < n; ++j) {
      if (arr[j] < arr[i]) {
        uint8_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
    }
  }
}
