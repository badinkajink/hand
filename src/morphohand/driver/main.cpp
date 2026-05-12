#include <Arduino.h>
#include <Stepper.h>

const int STEPS = 200;

// Note the interleaved pin order required by the Stepper library for L298N
Stepper motor1(STEPS, 8, 10, 9, 11);   // L298N #1
Stepper motor2(STEPS, 4, 6,  5, 7);    // L298N #2

void setup() {
  motor1.setSpeed(30);   // RPM — keep low for NEMA 8
  motor2.setSpeed(30);
  Serial.begin(9600);
}

void loop() {
  Serial.println("Both motors CW");
  motor1.step(200);
  motor2.step(200);
  delay(500);

  Serial.println("Both motors CCW");
  motor1.step(-200);
  motor2.step(-200);
  delay(500);
}