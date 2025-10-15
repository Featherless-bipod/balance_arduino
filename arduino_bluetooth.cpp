#include 'BluetoothSerial.h'
BluetoothSerial serialBT



void setup(){
    serialBT.begin("Esp32-BT");
    int inputPin = 2;
}

void loop(){
    int input = digitalRead(inputPin);
    



    if(serialBT.availiable(){
        serialBT.write(); 
    })
}
