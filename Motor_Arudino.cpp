
// ============================================================
// INTEGRATED: 16×16 Pressure Matrix + 30-Motor Haptic Control
// ============================================================
// This code handles BOTH:
// 1. Reading 16x16 pressure sensor matrix
// 2. Controlling 30 vibration motors via 2× PCA9685 boards
//
// Communication Protocol:
// Arduino → Python: Send sensor data when 'A' received
// Python → Arduino: Send motor commands starting with 'M'
// ============================================================

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ============================================================
// SENSOR READING CONFIGURATION
// ============================================================

// ----- MUX control pins (Updated to free A4/A5 for I2C) -----
const byte s0 = 7;   // Row select bit 0
const byte s1 = 10;  // Row select bit 1
const byte s2 = 11;  // Row select bit 2
const byte s3 = 12;  // Row select bit 3
const byte w0 = 6;   // Column select bit 0
const byte w1 = 5;   // Column select bit 1
const byte w2 = 4;   // Column select bit 2
const byte w3 = 3;   // Column select bit 3

// ----- I/O pins -----
const byte SIG_pin = A0;   // Analog input from sensor mux
const byte OUT_pin = 2;    // Output to write mux (column selector)
const byte COL_pin = 9;    // Status toggle (optional)
const byte STATUS_pin = 8; // Onboard LED indicator

// NOTE: A4 and A5 are FREE for I2C (PCA9685)!
// A4 = SDA (I2C data line)
// A5 = SCL (I2C clock line)

// ----- Multiplexer address map -----
const boolean muxChannel[16][4] = {
  {0,0,0,0}, {1,0,0,0}, {0,1,0,0}, {1,1,0,0},
  {0,0,1,0}, {1,0,1,0}, {0,1,1,0}, {1,1,1,0},
  {0,0,0,1}, {1,0,0,1}, {0,1,0,1}, {1,1,0,1},
  {0,0,1,1}, {1,0,1,1}, {0,1,1,1}, {1,1,1,1}
};

// ============================================================
// MOTOR CONTROL CONFIGURATION
// ============================================================

// ----- PCA9685 Setup -----
Adafruit_PWMServoDriver pca1 = Adafruit_PWMServoDriver(0x40); // Board 1
Adafruit_PWMServoDriver pca2 = Adafruit_PWMServoDriver(0x41); // Board 2

// ----- Motor Configuration -----
#define TOTAL_MOTORS 30
#define PWM_FREQUENCY 1000  // 1 kHz for motors

// PWM values (0-4095 range for 12-bit PWM)
#define MOTOR_OFF_PWM 0
#define CENTROID_MOTOR_PWM 2048  // 50% duty cycle (~350mA per motor)
#define BRAILLE_MOTOR_PWM 4095   // 100% duty cycle (~700mA per motor)

// Debug mode
#define DEBUG_MODE false  // Set to true for verbose serial output

// ============================================================
// SETUP
// ============================================================

void setup() {
  // --- Sensor pin setup ---
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
  
  Serial.println(F("========================================"));
  Serial.println(F("Integrated Sensor + Motor Control"));
  Serial.println(F("========================================"));
  
  // --- Initialize PCA9685 boards ---
  if (DEBUG_MODE) Serial.println(F("Initializing PCA9685 boards..."));
  
  pca1.begin();
  pca1.setPWMFreq(PWM_FREQUENCY);
  pca1.setOutputMode(true); // Totem pole mode for better motor control
  
  pca2.begin();
  pca2.setPWMFreq(PWM_FREQUENCY);
  pca2.setOutputMode(true);
  
  // Small delay for boards to stabilize
  delay(100);
  
  // CRITICAL: Immediately turn off all channels to prevent motors from running
  if (DEBUG_MODE) Serial.println(F("Setting all PCA9685 channels to OFF..."));
  for (uint8_t ch = 0; ch < 16; ch++) {
    pca1.setPWM(ch, 0, 0);  // Board 1, all channels OFF
    pca2.setPWM(ch, 0, 0);  // Board 2, all channels OFF
  }
  
  delay(100);
  
  // Initialize all motors to OFF state
  if (DEBUG_MODE) Serial.println(F("Turning off all motors..."));
  turnOffAllMotors();
  
  Serial.println(F("========================================"));
  Serial.println(F("System Ready!"));
  Serial.println(F("Commands:"));
  Serial.println(F("  'A' - Send sensor data (Python)"));
  Serial.println(F("  'M' - Motor command (Python)"));
  Serial.println(F("========================================\n"));
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  // Check for incoming commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch (command) {
      case 'A':
        // Python requesting sensor frame
        readAndSendSensorData();
        break;
        
      case 'M':
        // Python sending motor command
        receiveAndExecuteMotorCommand();
        break;
        
      default:
        // Ignore unknown commands
        break;
    }
  }
}

// ============================================================
// SENSOR READING FUNCTIONS
// ============================================================

void readAndSendSensorData() {
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

// ============================================================
// MOTOR CONTROL FUNCTIONS
// ============================================================

void receiveAndExecuteMotorCommand() {
  // Protocol:
  // 'M' (already read)
  // 1 byte: centroid motor count
  // N bytes: centroid motor indices
  // 1 byte: braille motor count
  // M bytes: braille motor indices
  
  // Wait for centroid count
  unsigned long start_time = millis();
  while (Serial.available() == 0 && millis() - start_time < 100);
  
  if (Serial.available() == 0) {
    if (DEBUG_MODE) Serial.println(F("Timeout waiting for motor command"));
    return;
  }
  
  // Read centroid motor count
  uint8_t centroid_count = Serial.read();
  
  if (centroid_count > TOTAL_MOTORS) {
    if (DEBUG_MODE) Serial.println(F("ERROR: Invalid centroid count"));
    return;
  }
  
  // Read centroid motor indices
  uint8_t centroid_motors[TOTAL_MOTORS];
  for (uint8_t i = 0; i < centroid_count; i++) {
    start_time = millis();
    while (Serial.available() == 0 && millis() - start_time < 100);
    
    if (Serial.available() == 0) {
      if (DEBUG_MODE) Serial.println(F("Timeout reading centroid motors"));
      return;
    }
    
    centroid_motors[i] = Serial.read();
    
    if (centroid_motors[i] >= TOTAL_MOTORS) {
      if (DEBUG_MODE) {
        Serial.print(F("ERROR: Invalid centroid motor index: "));
        Serial.println(centroid_motors[i]);
      }
      return;
    }
  }
  
  // Read braille motor count
  start_time = millis();
  while (Serial.available() == 0 && millis() - start_time < 100);
  
  if (Serial.available() == 0) {
    if (DEBUG_MODE) Serial.println(F("Timeout waiting for braille count"));
    return;
  }
  
  uint8_t braille_count = Serial.read();
  
  if (braille_count > TOTAL_MOTORS) {
    if (DEBUG_MODE) Serial.println(F("ERROR: Invalid braille count"));
    return;
  }
  
  // Read braille motor indices
  uint8_t braille_motors[TOTAL_MOTORS];
  for (uint8_t i = 0; i < braille_count; i++) {
    start_time = millis();
    while (Serial.available() == 0 && millis() - start_time < 100);
    
    if (Serial.available() == 0) {
      if (DEBUG_MODE) Serial.println(F("Timeout reading braille motors"));
      return;
    }
    
    braille_motors[i] = Serial.read();
    
    if (braille_motors[i] >= TOTAL_MOTORS) {
      if (DEBUG_MODE) {
        Serial.print(F("ERROR: Invalid braille motor index: "));
        Serial.println(braille_motors[i]);
      }
      return;
    }
  }
  
  // --- Execute motor command ---
  
  // First, turn off all motors
  turnOffAllMotors();
  
  // Activate centroid motors at 50% power
  for (uint8_t i = 0; i < centroid_count; i++) {
    setMotorPWM(centroid_motors[i], CENTROID_MOTOR_PWM);
  }
  
  // Activate braille motors at 100% power
  for (uint8_t i = 0; i < braille_count; i++) {
    setMotorPWM(braille_motors[i], BRAILLE_MOTOR_PWM);
  }
  
  if (DEBUG_MODE) {
    Serial.print(F("Motors activated: "));
    Serial.print(centroid_count);
    Serial.print(F(" centroid, "));
    Serial.print(braille_count);
    Serial.println(F(" braille"));
  }
}

// ----- Set motor PWM value -----
void setMotorPWM(uint8_t motor_index, uint16_t pwm_value) {
  if (motor_index >= TOTAL_MOTORS) {
    return; // Invalid motor index
  }
  
  // Determine which board and channel to use
  uint8_t board;
  uint8_t channel = motorIndexToChannel(motor_index, &board);
  
  // Set PWM on appropriate board
  if (board == 1) {
    pca1.setPWM(channel, 0, pwm_value);
  } else {
    pca2.setPWM(channel, 0, pwm_value);
  }
}

// ----- Convert motor index to PCA9685 board and channel -----
// Handles non-sequential mapping due to gaps at Channel 14
uint8_t motorIndexToChannel(uint8_t motor_index, uint8_t* board) {
  // Motors 0-14 → Board 1 (0x40)
  // Motors 15-29 → Board 2 (0x41)
  
  if (motor_index <= 14) {
    *board = 1;
    
    // Motors 0-13 use channels 0-13 (sequential)
    if (motor_index <= 13) {
      return motor_index;
    }
    // Motor 14 uses channel 15 (skips 14)
    else {
      return 15;
    }
  } else {
    *board = 2;
    
    // Motors 15-28 use channels 0-13 (sequential, offset by 15)
    if (motor_index <= 28) {
      return motor_index - 15;
    }
    // Motor 29 uses channel 15 (skips 14)
    else {
      return 15;
    }
  }
}

// ----- Turn off all motors -----
void turnOffAllMotors() {
  for (uint8_t i = 0; i < TOTAL_MOTORS; i++) {
    setMotorPWM(i, MOTOR_OFF_PWM);
  }
}

