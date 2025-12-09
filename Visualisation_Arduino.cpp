
// ============================================================
// 16×16 Velostat Pressure Matrix Reader for Arduino Uno
// ------------------------------------------------------------
// - Reads 256 sensors via dual 16-channel multiplexers
// - Streams raw bytes to serial at 115200 baud
// - Compatible with Python real-time visualization
// ============================================================

// ----- MUX control pins -----
const byte s0 = 7; 
const byte s1 = 10; 
const byte s2 = 11; 
const byte s3 = 12;

const byte w0 = 6;  
const byte w1 = 5; 
const byte w2 = 4; 
const byte w3 = 3;

// ----- I/O pins -----
const byte SIG_pin = A0;   // analog input from sensor mux
const byte OUT_pin = 2;    // output to write mux (column selector)
const byte COL_pin = 9;    // status toggle (optional)
const byte STATUS_pin = 8; // onboard LED indicator


// ----- Multiplexer address map -----
const boolean muxChannel[16][4] = {
  {0,0,0,0}, {1,0,0,0}, {0,1,0,0}, {1,1,0,0},
  {0,0,1,0}, {1,0,1,0}, {0,1,1,0}, {1,1,1,0},
  {0,0,0,1}, {1,0,0,1}, {0,1,0,1}, {1,1,0,1},
  {0,0,1,1}, {1,0,1,1}, {0,1,1,1}, {1,1,1,1}
};

void setup() {
  // --- Pin setup ---
  pinMode(s0, OUTPUT); pinMode(s1, OUTPUT); 
  pinMode(s2, OUTPUT); pinMode(s3, OUTPUT);
  pinMode(w0, OUTPUT); pinMode(w1, OUTPUT);
  pinMode(w2, OUTPUT); pinMode(w3, OUTPUT);
  pinMode(OUT_pin, OUTPUT);
  pinMode(COL_pin, OUTPUT);
  pinMode(STATUS_pin, OUTPUT);

  digitalWrite(OUT_pin, HIGH);
  digitalWrite(COL_pin, HIGH);
  digitalWrite(STATUS_pin, HIGH);

  // --- Serial setup ---
  Serial.begin(115200);
  delay(1000);
  Serial.println("Velostat 16x16 Streaming Ready");
}

void loop() {
  // --- Wait for Python to request frame ---
  if (Serial.available() && Serial.read() == 'A') {
    digitalWrite(STATUS_pin, HIGH);

    // Read each column (via write MUX)
    for (byte j = 0; j < 16; j++) {
      writeMux(j);

      // Read each row (via read MUX)
      for (byte i = 0; i < 16; i++) {
        int value = readMux(i);

        // Clamp & scale (optional normalization)
        value = constrain(value, 0, 1023);
        byte mappedVal = map(value, 0, 1023, 1, 255);

        // Send as a single byte
        Serial.write(mappedVal);
      }
    }
    digitalWrite(STATUS_pin, LOW);
  }
}

// ----- Read from the row multiplexer -----
int readMux(byte channel) {
  byte controlPins[] = {s0, s1, s2, s3};
  for (int i = 0; i < 4; i++) {
    digitalWrite(controlPins[i], muxChannel[channel][i]);
  }
  return analogRead(SIG_pin);
}

// ----- Write to the column multiplexer -----
void writeMux(byte channel) {
  byte controlPins[] = {w0, w1, w2, w3};
  for (int i = 0; i < 4; i++) {
    digitalWrite(controlPins[i], muxChannel[channel][i]);
  }
}
