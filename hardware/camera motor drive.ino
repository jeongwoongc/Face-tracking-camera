#include <VarSpeedServo.h>

VarSpeedServo panServo;
VarSpeedServo tiltServo;

String inputString = "";
unsigned int cont = 0;

const int PAN_MIN_ANGLE = 0;
const int PAN_MAX_ANGLE = 180;
const int TILT_MIN_ANGLE = 0;
const int TILT_MAX_ANGLE = 180;
const int PAN_IDLE_ANGLE = 60;   
const int TILT_IDLE_ANGLE = 150; 

void setup() {
  panServo.attach(9);
  tiltServo.attach(10);

  panServo.write(60);   // Set the initial position of panServo to 0 degrees
  tiltServo.write(150); // Set the initial position of tiltServo to 160 degrees

  Serial.begin(250000);
  Serial.println("Ready");
}

void loop() {
  signed int vel;
  unsigned int pos;

  if (Serial.available()) {
    inputString = Serial.readStringUntil('!');
    vel = inputString.toInt();

    if (vel >= -4 && vel <= 4) {
      // If velocity is within the range, stop the pan servo
      panServo.stop();
      tiltServo.stop();
    } 
    else {
      if (inputString.endsWith("x")) {
        int targetPan = constrain(panServo.read() + vel, PAN_MIN_ANGLE, PAN_MAX_ANGLE);
        panServo.write(targetPan, 4, true); // Move to the targetPan with a speed of 20
      }
      else if(inputString.endsWith("y")) {
        vel = -vel;
        int targetTilt = constrain(tiltServo.read() + vel, TILT_MIN_ANGLE, TILT_MAX_ANGLE);
        tiltServo.write(targetTilt, 3, true);
      }
    }
    inputString = "";
  }

}