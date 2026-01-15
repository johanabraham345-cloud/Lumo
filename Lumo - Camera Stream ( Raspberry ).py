from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()

time.sleep(1)
print("Camera streaming started. Press 'q' to exit.")

while True:
    frame = picam2.capture_array()
    # Convert color from RGB to BGR for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Raspberry Pi Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

