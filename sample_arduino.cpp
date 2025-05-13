// AgroBot_Arduino_Controller.ino
// Controls motors and sprayer based on serial commands from Raspberry Pi

#include <Servo.h>  // For servo motor control if needed

// Pin definitions
const int MOTOR_LEFT_EN = 5;    // PWM pin for left motor enable
const int MOTOR_LEFT_IN1 = 7;   // Direction control 1 for left motor
const int MOTOR_LEFT_IN2 = 8;   // Direction control 2 for left motor
const int MOTOR_RIGHT_EN = 6;   // PWM pin for right motor enable
const int MOTOR_RIGHT_IN1 = 9;  // Direction control 1 for right motor
const int MOTOR_RIGHT_IN2 = 10; // Direction control 2 for right motor
const int SPRAYER_PIN = 4;      // Relay control for sprayer

// Variables
String inputString = "";        // String to hold incoming command
boolean stringComplete = false; // Whether a complete command has been received
int leftSpeed = 0;              // Current left motor speed (0-255)
int rightSpeed = 0;             // Current right motor speed (0-255)
bool sprayerActive = false;     // Current sprayer state
unsigned long sprayerStartTime = 0; // When sprayer was activated
unsigned long sprayerDuration = 0;  // How long to spray in milliseconds

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  inputString.reserve(64);
  
  // Initialize motor pins
  pinMode(MOTOR_LEFT_EN, OUTPUT);
  pinMode(MOTOR_LEFT_IN1, OUTPUT);
  pinMode(MOTOR_LEFT_IN2, OUTPUT);
  pinMode(MOTOR_RIGHT_EN, OUTPUT);
  pinMode(MOTOR_RIGHT_IN1, OUTPUT);
  pinMode(MOTOR_RIGHT_IN2, OUTPUT);
  
  // Initialize sprayer pin
  pinMode(SPRAYER_PIN, OUTPUT);
  digitalWrite(SPRAYER_PIN, LOW); // Ensure sprayer is off
  
  // Send ready message
  Serial.println("ARDUINO:READY");
}

void loop() {
  // Process any complete commands
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // Auto-stop sprayer after duration
  if (sprayerActive && sprayerDuration > 0) {
    if (millis() - sprayerStartTime >= sprayerDuration) {
      setSprayerState(false);
      Serial.println("SPRAY:COMPLETED");
    }
  }
}

void processCommand(String command) {
  // Print received command for debugging
  Serial.print("Received: ");
  Serial.println(command);
  
  // Parse command
  if (command.startsWith("MOVE:")) {
    // Movement commands
    if (command.startsWith("MOVE:FWD")) {
      // Forward movement with optional speed
      int speed = 75; // Default speed 75%
      int commaIndex = command.indexOf(',');
      if (commaIndex > 0) {
        speed = command.substring(commaIndex + 1).toInt();
      }
      moveForward(speed);
      Serial.println("OK:MOVING_FORWARD");
    }
    else if (command.startsWith("MOVE:STOP")) {
      stopMotors();
      Serial.println("OK:STOPPED");
    }
  }
  else if (command.startsWith("SPRAY:")) {
    // Sprayer commands
    if (command.startsWith("SPRAY:ON")) {
      // Activate sprayer with optional duration
      float duration = 1.5; // Default 1.5 seconds
      int commaIndex = command.indexOf(',');
      if (commaIndex > 0) {
        duration = command.substring(commaIndex + 1).toFloat();
      }
      
      // Convert to milliseconds and activate
      sprayerDuration = (unsigned long)(duration * 1000);
      setSprayerState(true);
      Serial.println("OK:SPRAYING");
    }
    else if (command.startsWith("SPRAY:OFF")) {
      setSprayerState(false);
      Serial.println("OK:SPRAY_STOPPED");
    }
  }
}

void moveForward(int speed) {
  // Map 0-100% to 0-255 PWM
  int pwmSpeed = map(speed, 0, 100, 0, 255);
  
  // Set direction
  digitalWrite(MOTOR_LEFT_IN1, HIGH);
  digitalWrite(MOTOR_LEFT_IN2, LOW);
  digitalWrite(MOTOR_RIGHT_IN1, HIGH);
  digitalWrite(MOTOR_RIGHT_IN2, LOW);
  
  // Set speed
  analogWrite(MOTOR_LEFT_EN, pwmSpeed);
  analogWrite(MOTOR_RIGHT_EN, pwmSpeed);
  
  // Update current speed
  leftSpeed = pwmSpeed;
  rightSpeed = pwmSpeed;
}

void stopMotors() {
  // Set both enable pins to 0
  analogWrite(MOTOR_LEFT_EN, 0);
  analogWrite(MOTOR_RIGHT_EN, 0);
  
  // Update current speed
  leftSpeed = 0;
  rightSpeed = 0;
}

void setSprayerState(bool state) {
  // Set sprayer state
  digitalWrite(SPRAYER_PIN, state ? HIGH : LOW);
  sprayerActive = state;
  
  // If activating, record start time
  if (state) {
    sprayerStartTime = millis();
  } else {
    sprayerDuration = 0; // Clear duration when manually deactivating
  }
}

// Serial event is called when new data arrives
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    // Add character to input string
    if (inChar != '\n' && inChar != '\r') {
      inputString += inChar;
    }
    
    // If newline, set flag that a complete command has arrived
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}