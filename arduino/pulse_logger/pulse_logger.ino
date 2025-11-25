const int PULSE_SENSOR_PIN = A0;
const unsigned long SAMPLE_INTERVAL_MS = 10;  // 100Hz
const long SERIAL_BAUD_RATE = 115200;

unsigned long last_sample_ms = 0;

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  pinMode(PULSE_SENSOR_PIN, INPUT);
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_ms >= SAMPLE_INTERVAL_MS) {
    last_sample_ms = now;
    int value = analogRead(PULSE_SENSOR_PIN);
    Serial.print(now);
    Serial.print(',');
    Serial.println(value);
  }
}
