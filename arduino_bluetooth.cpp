#include <BluetoothSerial.h>
#include <ArduinoBLE.h>
#include <BasicLinearAlgebra.h>

BLEService streamService("12345678-1234-5678-1234-56789abcdef0");  // random UUID
BLECharacteristic streamChar("12345678-1234-5678-1234-56789abcdef1",
                             BLERead | BLENotify,  // readable + notify
                             244);                 // max len you *want* (cap at 244 if supported)


void setup(){
    
    Serial.begin(115200);
    while (!Serial) {}

    int inputPin1 = 2;
    int inputPin2 = 3;
    int inputPin3 = 4;
    int inputPin4 = 5;

    if (!BLE.begin()) {
        Serial.println("BLE init failed!");
        while (1);
    }

    serialBT.begin("Esp32-BT");

    BLE.setLocalName("BalanceBuds");
    BLE.setAdvertisedService(streamService);
    streamService.addCharacteristic(streamChar);
    BLE.addService(streamService);
    streamChar.writeValue((const unsigned char *)"init", 4); // initial value

    BLE.advertise();
    Serial.println("Advertising...");

    int weight_matrix = 
    int bias_matrix = 
 }

unsigned long lastSend = 0;
uint32_t seq = 0;

void loop(){
    int input1 = digitalRead(inputPin1);
    int input2 = digitalRead(inputPin2);
    int input3 = digitalRead(inputPin3);
    int input4 = digitalRead(inputPin4);

    BLEDevice central = BLE.central();
    if (central) {
        Serial.print("Connected to central: ");
        Serial.println(central.address());

        while (central.connected()) {
        // prepare payload - example: 8-byte packet (seq + timestamp)
        uint8_t buf[12];
        memcpy(buf + 0, &seq, 4);
        uint32_t now = (uint32_t)millis();
        memcpy(buf + 4, &now, 4);
        float sample = analogRead(A0) * (3.3 / 1023.0);
        memcpy(buf + 8, &sample, 4); // careful with endianness

        // Attempt to write the value. This may block or fail if radio busy.
        // Adjust payload length to negotiated MTU in advanced implementations.
        bool ok = streamChar.writeValue(buf, sizeof(buf));
        if (!ok) {
            // optional: back off slightly or count failures
        }
        seq++;
        // delay(1); micro-delay if needed
        }
        Serial.println("Central disconnected");
  }
}
