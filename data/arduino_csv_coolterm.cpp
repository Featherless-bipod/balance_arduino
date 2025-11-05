int voltage_pin = 
int sensor1 = 
int sensor2 = 
int sensor3 = 
int sensor4 = 
int sensor5 = 

void setup(){
    Serial.begin(9600);

}

void loop(){
    s1 = analogRead(sensor1)
    s2 = analogRead(sensor2)

    Serial.print(millis());
    Serial.print(",");
    Serial.print(s1);
    Serial.print(",");
    Serial.println(s2);

    delay(10)
}