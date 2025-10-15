#include <BluetoothSerial.h>
#include <BasicLinearAlgebra.h>

BluetoothSerial serialBT



void setup(){
    serialBT.begin("Esp32-BT");
    int inputPin1 = 2;
    int inputPin2 = 3;
    int inputPin3 = 4;
    int inputPin4 = 5;

    int weight_matrix = 
    int bias_matrix = 
 }

void loop(){
    int input1 = digitalRead(inputPin1);
    int input2 = digitalRead(inputPin2);
    int input3 = digitalRead(inputPin3);
    int input4 = digitalRead(inputPin4);

    


    



    if(serialBT.availiable(){
        serialBT.write(); 
    })

    delay(20);
}
