#include <Servo.h>

Servo myservoV;  
Servo myservoH;

int x; 
int y;

int prevX;
int prevY;


void setup() {
  myservoV.attach(9); 
  myservoH.attach(10);
  Serial.begin(9600);
  myservoV.write(90);
  myservoH.write(90);
}

void Pos() {
  if (prevX != x || prevY != y)
  {
    int servoX = map(x, 450, 0, 55, 140);
    int servoY = map(y, 450, 0, 135, 40);

    servoX = min(servoX, 140);
    servoX = max(servoX, 55);
    servoY = min(servoY, 135);
    servoY = max(servoY, 40);

    myservoV.write(servoX);
    myservoH.write(servoY);
    // delay(20);
  }
}

void loop() {

  if (Serial.available() > 0) 
  {
    // if (Serial.read() == 'X'){
    //   x = Serial.parseInt();
    //   Pos();
    // }
    if (Serial.read() == 'X')
    {
      x = Serial.parseInt();
      if (Serial.read() == 'Y')
      {
        y = Serial.parseInt();
        Pos();
      }
    }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  }

}