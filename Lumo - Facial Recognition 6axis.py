# FACE TRACKING FOR ROBOT ARM (ASCII SAFE VERSION)

from picamera2 import Picamera2
import cv2
import numpy as np
import pigpio
import time

# ================= SERVO START POSITIONS =================

servo_positions = {
    5:  90,
    6:  45,
    12: 170,
    13: 0,
    16: 90,
    19: 90

}

# Servo min/max angle limits
min_angle = {
    5: 0, 6: 0, 12: 0, 13: 0, 16: 0, 19: 0
}

max_angle = {
    5: 180, 6: 180, 12: 180, 13: 180, 16: 180, 19: 180
}

# Camera holder special limits
MAX_UP = 0
MAX_DOWN = 180

# ================= SETUP PIGPIO =================

pi = pigpio.pi()

def set_servo(pin, angle):
    if angle < min_angle[pin]:
        angle = min_angle[pin]
    if angle > max_angle[pin]:
        angle = max_angle[pin]

    pulse = 500 + (angle * 2000 // 180)
    pi.set_servo_pulsewidth(pin, pulse)
    servo_positions[pin] = angle

# Set initial positions
for pin, ang in servo_positions.items():
    set_servo(pin, ang)

# ================= SETUP CAMERA =================

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480)}
    )
)
picam2.start()
time.sleep(1)

# ================= LOAD FACE CASCADE =================

face_path = "/home/sarun/my_opencv_project/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_path)

if face_cascade.empty():
    print("Error loading Haar cascade.")
    exit()

# ================= TRACKING PARAMETERS =================

move_speed = 4

# ================= MAIN LOOP =================

while True:
    # Capture frame
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]

        face_x = x + w // 2
        face_y = y + h // 2

        cx = frame.shape[1] // 2
        cy = frame.shape[0] // 2

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ---------------- HORIZONTAL MOVEMENT (base servo) ----------------
        if face_x < cx - 40:
            set_servo(5, servo_positions[5] + move_speed)
        elif face_x > cx + 40:
            set_servo(5, servo_positions[5] - move_speed)

        # ---------------- VERTICAL MOVEMENT (elbow) ----------------
        if face_y < cy - 40:
            set_servo(12, servo_positions[12] - move_speed)
        elif face_y > cy + 40:
            set_servo(12, servo_positions[12] + move_speed)

        # ---------------- CAMERA HOLDER SERVO (pin 13) ----------------
        if face_y < cy - 40:
            new_ang = servo_positions[13] + move_speed
            if new_ang <= MAX_DOWN:
                set_servo(13, new_ang)

        elif face_y > cy + 40:
            new_ang = servo_positions[13] - move_speed
            if new_ang >= MAX_UP:
                set_servo(13, new_ang)

    # Show feed
    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
for pin in servo_positions:
    pi.set_servo_pulsewidth(pin, 0)
pi.stop()

