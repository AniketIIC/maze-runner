import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform perspective transformation on a frame
def transform_frame(frame, input_pts, output_pts, max_width, max_height):
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (max_width, max_height), flags=cv2.INTER_LINEAR)
    return out

# Load the video
video_path = 'img11.jpg'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each frame in the video
while True:
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert to RGB for displaying with matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    plt.imshow(frame_rgb)
   # plt.title("Click on four points for perspective transformation")
   # plt.axis('on')

    # Define a list to store the selected points
    points = []

    # Define a callback function for mouse clicks
    def onclick(event):
        if event.button == 1:  # Left mouse button click
            points.append((event.xdata, event.ydata))
            plt.scatter(event.xdata, event.ydata, c='red')  # Mark the selected point
            plt.text(event.xdata, event.ydata, f"{len(points)}", color='red', fontsize=12, ha='right', va='bottom')
            plt.draw()
            if len(points) == 4:
                plt.close()

    # Connect the callback function to the figure
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    # Show the plot
    plt.show()

    # Once the plot is closed, the selected points will be stored in the 'points' list
    # print("Selected points:", points)

    # Calculate dimensions and perform perspective transformation
    width_AD = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    width_BC = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    height_CD = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Check if the calculated dimensions are valid
    if maxWidth > 0 and maxHeight > 0:
        length = min(maxHeight , maxWidth)
        input_pts = np.float32([points[0], points[1], points[2], points[3]])
        output_pts = np.float32([[0, 0],
                                 [0, length],
                                 [length, length],
                                 [length, 0]])
        print("input coordinate: ", input_pts)
        print("output coordinate: ", output_pts)

        # Apply perspective transformation to the frame
        transformed_frame = transform_frame(frame, input_pts, output_pts, length, length)

        # Display the output frame
        plt.figure()
        plt.imshow(transformed_frame)
        plt.title("Perspective Transformed Frame")
        plt.show()

        # Write the transformed frame to the output video
        out.write(transformed_frame)
    else:
        print("Invalid dimensions. Please make sure the selected points form a valid quadrilateral.")

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
