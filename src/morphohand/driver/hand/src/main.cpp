#include <Arduino.h>
#include <AccelStepper.h>

// FULL4WIRE mode, pins in IN1 IN3 IN2 IN4 order
AccelStepper motor1(AccelStepper::FULL4WIRE, 8, 10, 9, 11);
AccelStepper motor2(AccelStepper::FULL4WIRE, 4, 6,  5, 7);

void setup() {
  motor1.setMaxSpeed(100);
  motor1.setAcceleration(10);
  motor1.moveTo(400);   // 2 revolutions

  motor2.setMaxSpeed(100);
  motor2.setAcceleration(10);
  motor2.moveTo(-400);  // 2 revolutions opposite direction
}

void loop() {
  motor1.run();  // non-blocking — call every loop iteration
  motor2.run();

  if (motor1.distanceToGo() == 0) motor1.moveTo(-motor1.currentPosition());
  if (motor2.distanceToGo() == 0) motor2.moveTo(-motor2.currentPosition());
}