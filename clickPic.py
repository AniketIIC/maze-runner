import cv2
from datetime import datetime

# Open a connection to the camera (0 represents the default camera, you can change it if you have multiple cameras)
cap = cv2.VideoCapture(0)

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_name = f"captured_image_{current_date}.jpg"

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read a frame from the camera
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Could not read frame.")
    exit()

# Display the captured frame
cv2.imshow(image_name, frame)

# Save the captured image
cv2.imwrite(image_name, frame)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()



