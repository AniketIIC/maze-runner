import cv2
import screeninfo
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

image_name = f"captured_image_{current_date}.jpg"

def main():
    # Open the default camera (usually camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Hardcoded width and aspect ratio
    target_width = 3840
    aspect_ratio = (32, 9)

    # Calculate the corresponding height based on the specified width and aspect ratio
    target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])

    # Create a window with the specified width and height
    window_name = 'Camera Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, target_width, target_height)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        frame = cv2.resize(frame, (target_width, target_height))

        # Break the loop if reading the frame fails
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow(window_name, frame)

        # Resize the window to maintain the aspect ratio when the user scales it
        current_width, current_height = cv2.getWindowImageRect(window_name)[2:4]
        new_height = int(current_width / aspect_ratio[0] * aspect_ratio[1])
        cv2.resizeWindow(window_name, current_width, new_height)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imwrite(image_name, frame)

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
