import RPi.GPIO as GPIO
import time

# Set GPIO numbering mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
IN1 = 17  # GPIO 17 (LED 1)
IN2 = 27  # GPIO 27 (LED 2)

# Set GPIO pins as outputs
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

try:
    while True:
        # LED 1 on (Motor supposed to be forward)
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        time.sleep(5)

        # LED 2 on (Motor supposed to be backward)
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        time.sleep(5)
        

except KeyboardInterrupt:
    pass

# Cleanup
GPIO.cleanup()
