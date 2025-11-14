#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>

// -----------------------
// Pins
// -----------------------
const int inputPin1 = 14;
const int inputPin2 = 27;
const int inputPin3 = 26;
const int inputPin4 = 25;

#define ANALOG_PIN 34   // ADC input

// -----------------------
// WiFi Credentials
// -----------------------
const char* ssid     = "DukeVisitor";
const char* password = "";

// -----------------------
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81);

uint32_t seq = 0;

// -----------------------
// Webpage served to browser
// -----------------------
const char webpage[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<title>BalanceBuds Live</title>
<meta charset="utf-8">
<style>
body { font-family: Arial, sans-serif; padding: 20px; }
#box { 
  border: 1px solid #aaa; 
  width: 300px; 
  padding: 20px; 
  border-radius: 10px; 
  margin-bottom: 20px;
}
button {
  padding: 10px 20px;
  border: none;
  background: #007bff;
  color: white;
  border-radius: 6px;
  cursor: pointer;
}
button:hover {
  background: #0056cc;
}
</style>
</head>
<body>

<h2>BalanceBuds – Real-Time Sensor Feed</h2>

<div id="box">
  <p><strong>Seq:</strong> <span id="seq">0</span></p>
  <p><strong>Timestamp:</strong> <span id="ts">0</span></p>
  <p><strong>Analog:</strong> <span id="an">0</span></p>
  <p><strong>Digital:</strong> <span id="dg">0,0,0,0</span></p>
</div>

<button onclick="downloadCSV()">Download CSV</button>

<script>
let ws = new WebSocket("ws://" + location.hostname + ":81/");
let log = [];  // CSV data storage

ws.onmessage = (event) => {
  let obj = JSON.parse(event.data);

  // Update UI
  document.getElementById("seq").textContent = obj.seq;
  document.getElementById("ts").textContent = obj.timestamp;
  document.getElementById("an").textContent = obj.analog;
  document.getElementById("dg").textContent = obj.digital.join(",");

  // Store a row for CSV
  log.push([
    obj.seq,
    obj.timestamp,
    obj.analog,
    ...obj.digital
  ]);
};

// Create and download CSV
function downloadCSV() {
  let csv = "seq,timestamp,analog,";

  // Add dynamic number of digital columns
  let numD = log[0][3] !== undefined ? log[0].length - 3 : 0;
  for (let i = 0; i < numD; i++) {
    csv += "d" + i + ",";
  }
  csv = csv.slice(0, -1) + "\n";  // trim last comma

  // Add rows
  log.forEach(row => {
    csv += row.join(",") + "\n";
  });

  // Download
  let blob = new Blob([csv], { type: "text/csv" });
  let url = URL.createObjectURL(blob);
  let a = document.createElement("a");
  a.href = url;
  a.download = "balancebuds_data.csv";
  a.click();
  URL.revokeObjectURL(url);
}
</script>

</body>
</html>
)rawliteral";


// -----------------------
// WebSocket event handler
// -----------------------
void onWsEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t len) {
  if (type == WStype_CONNECTED) {
    Serial.printf("Client %u connected\n", num);
  } 
  else if (type == WStype_DISCONNECTED) {
    Serial.printf("Client %u disconnected\n", num);
  }
}

// -----------------------
// Setup
// -----------------------
void setup() {
  Serial.begin(115200);

  pinMode(inputPin1, INPUT_PULLUP);
  pinMode(inputPin2, INPUT_PULLUP);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // HTTP server
  server.on("/", []() { server.send_P(200, "text/html", webpage); });
  server.begin();
  Serial.println("HTTP server started");

  // WebSocket server
  webSocket.begin();
  webSocket.onEvent(onWsEvent);
  Serial.println("WebSocket server started");
}

// -----------------------
// Loop
// -----------------------
void loop() {
  server.handleClient();
  webSocket.loop();

  // Read sensors
  int in1 = digitalRead(inputPin1);
  int in2 = digitalRead(inputPin2);

  float analogVal = analogRead(ANALOG_PIN) * (3.3 / 4095.0);
  uint32_t ts = millis();

  // Build JSON
  String json = "{";
  json += "\"seq\":" + String(seq++) + ",";
  json += "\"timestamp\":" + String(ts) + ",";
  json += "\"analog\":" + String(analogVal, 4) + ",";
  json += "\"digital\":[" + String(in1) + "," + String(in2) + "]";
  json += "}";

  webSocket.broadcastTXT(json);

  delay(10); // 100 Hz update rate; change to 1 for 1000 Hz
}
