#include <ArduinoBLE.h>
//#include <BasicLinearAlgebra.h>

// Define pins
const int inputPin1 = 2;
const int inputPin2 = 3;
const int inputPin3 = 4;
const int inputPin4 = 5;

// Define BLE Service and Characteristic
BLEService streamService("12345678-1234-5678-1234-56789abcdef0");
BLECharacteristic streamChar("12345678-1234-5678-1234-56789abcdef1",
                             BLERead | BLENotify,  // readable + notify
                             20);                  // payload length (max ~244)

unsigned long lastSend = 0;
uint32_t seq = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Initialize pins
  pinMode(inputPin1, INPUT_PULLUP);
  pinMode(inputPin2, INPUT_PULLUP);
  pinMode(inputPin3, INPUT_PULLUP);
  pinMode(inputPin4, INPUT_PULLUP);

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("BalanceBuds");
  BLE.setAdvertisedService(streamService);
  streamService.addCharacteristic(streamChar);
  BLE.addService(streamService);

  // Initial value
  streamChar.writeValue("init");

  BLE.advertise();
  Serial.println("BLE Advertising as BalanceBuds...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    while (central.connected()) {
      unsigned long now = millis();
      if (now - lastSend >= 100) {  // send every 100 ms
        lastSend = now;

        int in1 = digitalRead(inputPin1);
        int in2 = digitalRead(inputPin2);
        int in3 = digitalRead(inputPin3);
        int in4 = digitalRead(inputPin4);

        float analogVal = analogRead(A0) * (3.3 / 1023.0);

        struct __attribute__((packed)) {
          uint32_t seq;
          uint32_t timestamp;
          float analogVal;
          uint8_t digital[4];
        } payload;

        payload.seq = seq++;
        payload.timestamp = millis();
        payload.analogVal = analogVal;
        payload.digital[0] = in1;
        payload.digital[1] = in2;
        payload.digital[2] = in3;
        payload.digital[3] = in4;

        bool ok = streamChar.writeValue((uint8_t *)&payload, sizeof(payload));

        if (!ok) {
          Serial.println("⚠️ BLE send failed (radio busy)");
        }
      }
    }

    Serial.println("Central disconnected");
  }
}
