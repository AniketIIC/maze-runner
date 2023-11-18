import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('img11.jpg')

# Create a copy of the image
img_copy = np.copy(img)

# Convert to RGB for displaying with matplotlib
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_copy)
plt.title("Click on four points for perspective transformation")
plt.axis('on')

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
print("Selected points:", points)

# Now you can use the selected points to calculate the perspective transformation matrix
# transform_mat = cv2.getPerspectiveTransform(src, dst)


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
                             [0, length ],
                             [length , length ],
                             [length , 0]])

    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, M, (length, length), flags=cv2.INTER_LINEAR)

    # Display the output image
    plt.figure()
    plt.imshow(out)
    #lip_out = cv2.flip(out,1)
    #plt.imshow(Flip_out)
    plt.title("Perspective Transformed Image")
    plt.show()
else:
    print("Invalid dimensions. Please make sure the selected points form a valid quadrilateral.")
