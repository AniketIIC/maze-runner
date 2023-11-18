import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_name = f"captured_image_{current_date}.jpg"
points = []
tranformed_img_name = f"transformed_image_{current_date}.jpg"

def center(coords):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    return center_x, center_y

def onclick(event):
    if event.button == 1:  # Left mouse button click
        points.append((event.xdata, event.ydata))
        plt.scatter(event.xdata, event.ydata, c='red')  # Mark the selected point
        plt.text(event.xdata, event.ydata, f"{len(points)}", color='red', fontsize=12, ha='right', va='bottom')
        plt.draw()
        if len(points) == 4:
            plt.close()

def clickReferencePic():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        exit()

    cv2.imshow(image_name, frame)

    cv2.imwrite(image_name, frame)

    cap.release()
    cv2.destroyAllWindows()

def tranformTheReferenceImage():
    img = cv2.imread(image_name)
    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.imshow(img_copy)
    #plt.title("Click on four points for perspective transformation")
    plt.axis('on')
    
    
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    width_AD = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    width_BC = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    height_CD = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
 
    if maxWidth > 0 and maxHeight > 0:
        squareSideLength = min(maxHeight , maxWidth)
        input_pts = np.float32([points[0], points[1], points[2], points[3]])
        output_pts = np.float32([[0, 0],
                                [0, squareSideLength ],
                                [squareSideLength , squareSideLength ],
                                [squareSideLength , 0]])

        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        original_point = center(points)
        print('Center is: ', original_point)
        transformed_point = cv2.perspectiveTransform(np.array([[original_point]], dtype=np.float32), M)[0][0]
        print('Centre on perpective image is: ', transformed_point)
        out = cv2.warpPerspective(img, M, (squareSideLength, squareSideLength), flags=cv2.INTER_LINEAR)

        # Display the output image
        plt.figure()
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.imshow(out)
        #lip_out = cv2.flip(out,1)
        #plt.imshow(Flip_out)
        plt.title("Perspective Transformed Image")
        plt.show()
        cv2.imwrite(tranformed_img_name, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    else:
        print("Invalid dimensions. Please make sure the selected points form a valid quadrilateral.")


clickReferencePic()

tranformTheReferenceImage()