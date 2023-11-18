import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define a list to store the selected points
points = []


def tranformImg(img):
    # Load the image
    #print(img)
    #img = cv2.imread(img)

    # Create a copy of the image
    img_copy = np.copy(img)

    # Convert to RGB for displaying with matplotlib
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_copy)
    plt.title("Click on four points for perspective transformation")
    plt.axis('on')



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
    if len(points)==0:
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

    
    
   

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--consecutive-frames', default=4, type=int,
                    dest='consecutive_frames', help='ref.png')
args = vars(parser.parse_args())

# capture frames from the camera
cap = cv2.VideoCapture(0)

# get the video frame height and width
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = "outputs/output_video.mp4"

# define codec and create VideoWriter object
out = cv2.VideoWriter(
    save_name,
    cv2.VideoWriter_fourcc(*'mp4v'), 10,
    (frame_width, frame_height)
)

# get the initial background frame from the camera
background = None
for _ in range(args['consecutive_frames']):
    _, background = cap.read()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

frame_count = 0
consecutive_frame = args['consecutive_frames']
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []

        frame_diff = cv2.absdiff(gray, background)
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        frame_diff_list.append(dilate_frame)

        if len(frame_diff_list) == consecutive_frame:
            sum_frames = sum(frame_diff_list)
            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                center_point = (center_x, center_y)

                cv2.circle(orig_frame, center_point, 5, (255, 0, 0), -1)

            cv2.imshow('Detected Objects', orig_frame)
            tranformImg(orig_frame)
            out.write(orig_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
