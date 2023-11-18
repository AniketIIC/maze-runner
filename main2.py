import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time


current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_name = f"captured_image_{current_date}.jpg"
points = []
tranformed_img_name = f"transformed_image_{current_date}.jpg"
THRESHOLD_VALUE = 30



# Function to perform perspective transformation on a frame
def transform_frame(frame, input_pts, output_pts, max_width, max_height):
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(frame, M, (max_width, max_height), flags=cv2.INTER_LINEAR)
    return out


# click on pic 

def onclick(event):
    if event.button == 1:  # Left mouse button click
        points.append((event.xdata, event.ydata))
        plt.scatter(event.xdata, event.ydata, c='red')  # Mark the selected point
        plt.text(event.xdata, event.ydata, f"{len(points)}", color='red', fontsize=12, ha='right', va='bottom')
        plt.draw()
        if len(points) == 4:
            plt.close()


# open camera and click pic and save to root folder

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

# use the above pic to generate the transformed image and save the selected point for live tracking later

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
        input_pts = np.float32([points[0], points[1], points[2], points[3]])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight ],
                                [maxWidth , maxHeight ],
                                [maxWidth , 0]])

        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        out = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

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


def tranformImageLiveTrack(img,original_point):
    img_copy = np.copy(img)

    # Convert to RGB for displaying with matplotlib
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    # Display the image
    #plt.imshow(img_copy)
    #plt.title("Click on four points for perspective transformation")
    #plt.axis('on')

    width_AD = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    width_BC = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    height_CD = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))


    if maxWidth > 0 and maxHeight > 0:
        input_pts = np.float32([points[0], points[1], points[2], points[3]])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight ],
                                [maxWidth , maxHeight ],
                                [maxWidth , 0]])
        print("Dimenstions: ",output_pts)
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        print("Original Coordinates: ", original_point)
        transformed_point = cv2.perspectiveTransform(np.array([[original_point]], dtype=np.float32), M)[0][0]
       # if(transformed_point[0]>=0 and transformed_point[1]>=0 and transformed_point[0]<=maxWidth and transformed_point[1]<=maxHeight ):
        print("Perspective Coordinates: ", transformed_point)
        out = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
        #print("inputs are: ", input_pts)
        #print("outputs are: ", output_pts)
        # Display the output image
        plt.figure()
        plt.imshow(out)
        #lip_out = cv2.flip(out,1)
        #plt.imshow(Flip_out)
        plt.title("Perspective Transformed Image")
        plt.show()
        return out
        
    else:
        print("Invalid dimensions. Please make sure the selected points form a valid quadrilateral.")


# start the live tracking

def liveTrack():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--consecutive-frames', default=2, type=int,
                        dest='consecutive_frames', help=tranformed_img_name)
    args = vars(parser.parse_args())

    # capture frames from the camera
    cap = cv2.VideoCapture(0)

    # get the initial background frame from the camera
    background = None
    for _ in range(args['consecutive_frames']):
        _, background = cap.read()
    #background = cv2.imread(tranformed_img_name)
    if background is None:
        print(f"Error: Unable to read the image at {background}")
        exit()
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    consecutive_frame = args['consecutive_frames']
    while cap.isOpened():
        #print(cap.isOpened)
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            orig_frame = frame.copy()
            #orig_frame = tranformImageLiveTrack(orig_frame,center_point)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % consecutive_frame == 0 or frame_count == 1:
                frame_diff_list = []
            #print(gray.shape, background.shape)
            frame_diff = cv2.absdiff(gray, background)
            ret, thres = cv2.threshold(frame_diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            dilate_frame = cv2.dilate(thres, None, iterations=2)
            frame_diff_list.append(dilate_frame)

            if len(frame_diff_list) == consecutive_frame:
                sum_frames = sum(frame_diff_list)
                contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print("Total contours are: ", len(contours))
                center_point=(0,0)
                #print("contiurs: ", contours)
                for i, cnt in enumerate(contours):
                    cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

                for contour in contours:
                    print("Current countour area is: ",cv2.contourArea(contour) )
                    if cv2.contourArea(contour)<=300:
                        continue
                    #if 200 >= cv2.contourArea(contour) >= 500:
                        #continue
                     
                    #print("Countour is: ", contour)
                    (x, y, w, h) = cv2.boundingRect(contour)
                    #print(x,y,w,h)
                    cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center_point = (center_x, center_y)
                    #print("Center point is: ", center_point)

                    cv2.circle(orig_frame, center_point, 5, (255, 0, 0), -1)
                    
                tranformImageLiveTrack(orig_frame,center_point)    
                    #print("Orig Frame Shape",orig_frame.shape)
                #cv2.imshow('Detected Objects', orig_frame)
                #out.write(orig_frame)
                
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        else:
            break
        time.sleep(0.5)
    cap.release()
    #out.release()
    cv2.destroyAllWindows()






clickReferencePic()

tranformTheReferenceImage()

liveTrack()