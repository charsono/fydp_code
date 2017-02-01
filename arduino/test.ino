#include <Servo.h>

// Front motors
Servo motor1;
Servo motor2;
// Back motors
Servo motor3;
Servo motor4;

const int motor1Pin = 3;
const int motor2Pin = 4;
const int motor3Pin = 5;
const int motor4Pin = 6;

// 0 - 93 Reverse, > 93 Forward
const int motor1StopPos = 93;
const int motor3StopPos = 93;
// 0 - 93 Forward, > 93 Reverse
const int motor2StopPos = 93;
const int motor4StopPos = 93;


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
    turnLeft(5);
    delay(1500);
    stopAndDelay();
    turnRight(5);
    delay(1500);
    stopAndDelay();
    stopFlag = true;
  }
}

void demo() {
  drive(-5);
  delay(1000);
  stopAndDelay();
  drive(5);
  delay(1000);
  stopAndDelay();
  
  turnSharpLeft(10);
  delay(3000);
  stopAndDelay();
  turnSharpRight(10);
  delay(3000);
  stopAndDelay();
}

void stopAndDelay() {
  stop();
  delay(1000);
}

void drive(int speed) {
  setSpeed(speed);
}

void setSpeed(int speed) {
  setLeftWingSpeed(speed);
  setRightWingSpeed(speed);
}

void setLeftWingSpeed(int speed) {
  motor1.write(motor1StopPos + speed);
  motor3.write(motor3StopPos + speed);
}

void setRightWingSpeed(int speed) {
  motor2.write(motor2StopPos - speed);
  motor4.write(motor4StopPos - speed);
}

void stop() {
  setSpeed(93);
}

void turnSharpLeft(int speed) {
  setLeftWingSpeed(-speed);
  setRightWingSpeed(speed);
}

void turnSharpRight(int speed) {
  setLeftWingSpeed(speed);
  setRightWingSpeed(-speed);
}

void turnLeft(int speed) {
  int bufferSpeed = speed/2;
  setLeftWingSpeed(bufferSpeed);
  setRightWingSpeed(speed);
}

void turnRight(int speed) {
  int bufferSpeed = speed/2;
  setLeftWingSpeed(speed);
  setRightWingSpeed(bufferSpeed);
}

