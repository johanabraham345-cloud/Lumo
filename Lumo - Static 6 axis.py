import pigpio
from time import sleep

# List of servo GPIO pins
servo_pins = [5, 6, 12, 13, 16, 19]

# Start pigpio
pi = pigpio.pi()
if not pi.connected:
    print("ERROR: pigpio not running")
    exit()

# Convert angle (0-180) to pulse width
def angle_to_pulse(angle):
    return int(500 + (angle / 180.0) * 2000)

# Set angle for each servo here
angles = {
    5:  90,
    6:  40,
    12: 160,
    13: 20,
    16: 90,
    19: 90

}

# Apply angles
for pin, angle in angles.items():
    pw = angle_to_pulse(angle)
    pi.set_servo_pulsewidth(pin, pw)
    print("GPIO", pin, "Angle", angle, "Pulse", pw)

print("Holding positions... Press CTRL+C to stop.")

try:
    while True:
        sleep(1)

except KeyboardInterrupt:
    print("Stopping")

finally:
    for pin in servo_pins:
        pi.set_servo_pulsewidth(pin, 0)
    pi.stop()
   
