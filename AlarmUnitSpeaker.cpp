#include <SD.h>                           //include SD module library
#include <TMRpcm.h>                       //include speaker control library

// Radar pins
#define RADAR_RX 2
#define RADAR_TX 3
SoftwareSerial radarSerial(RADAR_RX, RADAR_TX);
// Alarm or LED output
#define OUTPUT_PIN 4
#define SD_ChipSelectPin 10
TMRpcm tmrpcm;
String radarBuffer = "";
void setup() {
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, LOW);
  Serial.begin(19200);           // For debugging
  radarSerial.begin(19200);      // OPS243 radar
  delay(500);
  radarSerial.write("\r\nOU\r\n");  // Optional radar config
  Serial.println("System ready.");
  tmrpcm.speakerPin = 9;                  //define speaker pin 
  tmrpcm.setVolume(5);                    //0 to 7. Set volume level
  }
void loop() {
  // Only process radar data
  readRadarData();
}
void readRadarData() {
  while (radarSerial.available()) {
    char c = radarSerial.read();
    if (c != '\n' && c != '\r') {
      radarBuffer += c;
    } else {
      processRadarLine(radarBuffer);
      radarBuffer = "";
    }
  }
}
void processRadarLine(String line) {
  line.trim();
  if (line.indexOf("ALERT") != -1) {
    // ALERT detected
    Serial.println("🚨 ALERT detected!");
    digitalWrite(OUTPUT_PIN, HIGH);
    tmrpcm.play("abcd.wav");              // Play sound when alert triggers 
    delay(3000);                          // Keep alarm on for 3 full seconds
    tmrpcm.stopPlayback();                // Stop the sound after 3 seconds
    digitalWrite(OUTPUT_PIN, LOW);
  }
}