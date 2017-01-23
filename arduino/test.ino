#include <Servo.h>

// Front motors
Servo motor1;
Servo motor2;
// Back motors
Servo motor3;
Servo motor4;

const int motor1Pin = 2;
const int motor2Pin = 3;
const int motor3Pin = 4;
const int motor4Pin = 5;

// 0 - 93 Forward, > 93 Reverse
const int motor1StopPos = 93;
const int motor4StopPos = 93;
// 0 - 93 Reverse, > 93 Forward
const int motor2StopPos = 93;
const int motor3StopPos = 93;

bool stopFlag = false;

void setup() {
  // put your setup code here, to run once:
  initializePins();
  stop();
}

void initializePins() {
  motor1.attach(motor1Pin);
  motor2.attach(motor2Pin);
  motor3.attach(motor3Pin);
  motor4.attach(motor4Pin);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (!stopFlag) {
    drive();
    delay(1000);
    stop();
    stopFlag = true;
  }
}

void drive() {
  motor1.write(83);
  motor2.write(103);
  motor3.write(103);
  motor4.write(83);  
}

void stop() {
  motor1.write(motor1StopPos);
  motor2.write(motor2StopPos);
  motor3.write(motor3StopPos);
  motor4.write(motor4StopPos);
}

