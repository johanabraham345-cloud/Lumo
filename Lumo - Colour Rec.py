from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialize PiCamera2
picam2 = Picamera2()
picam2.start()
time.sleep(1)  # Give the camera time to warm up

# Define HSV color ranges (as list of tuples)
color_ranges = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255]))
    ],
    "green": [
        (np.array([40, 100, 100]), np.array([80, 255, 255]))
    ],
    "blue": [
        (np.array([100, 150, 0]), np.array([140, 255, 255]))
        
        
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([30, 255, 255]))
    ],
}

# BGR colors used for rectangle and text drawing
label_colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}

# Load Haar cascade for face detection
face_cascade_path = '/home/sarun/my_opencv_project/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Error: Haar cascade not loaded.")
    exit()

while True:
    # Capture and convert frame
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through colors
    for color, ranges in color_ranges.items():
        # Combine all masks for this color
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, mask)

        # Find contours for this color
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # filter out noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), label_colors[color], 2)
                cv2.putText(frame, color.upper(), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_colors[color], 2)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show result
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
