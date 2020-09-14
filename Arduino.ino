#include<Servo.h>
Servo x;
int m1a = 7;
int m1b = 8;
int m2a = 11;
int m2b = 12;
int ENA = 5;
int lb=13;
int lr=2;
int lg=3;
int ENB = 6;
int servopin=9;
char val;
void setup()
{
  pinMode(m1a, OUTPUT);
  pinMode(m1b, OUTPUT);
  pinMode(m2a, OUTPUT);
  pinMode(m2b, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(lb,OUTPUT);
  pinMode(lr,OUTPUT);
  pinMode(lg,OUTPUT);
  int angle=100;
  x.attach(servopin); 
  analogWrite(ENA, 74);
  analogWrite(ENB, 74);
  Serial.begin(9600);
}
void loop()
{
  while (Serial.available() > 0)
  {
    val = Serial.read();
    Serial.println(val);
  }
  if (val == 'f') //Forward
  {
    digitalWrite(m1a, HIGH);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, HIGH);
  }
  else if (val == 'b') //Backward
  {
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, HIGH);
    digitalWrite(m2a, HIGH);
    digitalWrite(m2b, LOW);
  }
  else if (val == 'r') //Right
  {
    digitalWrite(m1a, HIGH);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, HIGH);
    digitalWrite(m2b, LOW);
  }
  else if (val == 'l') //Right
  {
    
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, HIGH);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, HIGH);
    
  }
  else if (val == 'z') //Right
  {
    analogWrite(ENA, 34);
  analogWrite(ENB, 34);
    digitalWrite(m1a, HIGH);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, HIGH);
    digitalWrite(m2b, LOW);
    analogWrite(ENA, 74);
  analogWrite(ENB, 74);
  }
  else if (val == 'a') //Right
  {
    analogWrite(ENA, 34);
  analogWrite(ENB, 34);
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, HIGH);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, HIGH);
    analogWrite(ENA, 74);
  analogWrite(ENB, 74);
    
  }
  else if (val == 'd') //servodown
  { 
    
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, LOW);
    for(int angle = 100; angle < 180; angle++)  
  {                                  
    x.write(angle);               
    delay(15);                   
  } 
  
  }
  /*{
  x.write(180);
  delay(1000);
 
  }*/
  else if (val == 'j') //servodown1
  { 
    
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, LOW);
    for(int angle = 100; angle < 180; angle++)  
  {                                  
    x.write(angle);               
    delay(15);                   
  } 
  digitalWrite(lg,HIGH);
    delay(1000);
    digitalWrite(lg,LOW);
  
  }
  else if (val == 'u') //servoup
  {
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, LOW);
    for(int angle = 180; angle > 100; angle--)    
  {                                
    x.write(angle);           
    delay(15);       
  } 
  digitalWrite(lb,HIGH);
    delay(1000);
    digitalWrite(lb,LOW);
  
  }
  else if (val == 'k') //servoup1
  {
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, LOW);
    for(int angle = 180; angle > 100; angle--)    
  {                                
    x.write(angle);           
    delay(15);       
  } 
  digitalWrite(lr,HIGH);
    delay(1000);
    digitalWrite(lr,LOW);
  
  }
  /*{
 
  x.write(100);
  delay(1000);
  }*/
  else if (val == 'e') //stop
  {
 
    digitalWrite(lg,HIGH);
    delay(3000);
    digitalWrite(lg,LOW);
  }
  
  else if (val == 's') //stop
  {
 
    digitalWrite(m1a, LOW);
    digitalWrite(m1b, LOW);
    digitalWrite(m2a, LOW);
    digitalWrite(m2b, LOW);
    
  }
  

}
