import json
import socketio
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse
import screeninfo

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_name = f"captured_image_{current_date}.jpg"
points = []
tranformed_img_name = f"transformed_image_{current_date}.jpg"
THRESHOLD_VALUE = 30


# Create a Socket.IO client instance
sio = socketio.Client()

# Define event handlers
@sio.event
def connect():
    print('Connected to server')
    

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.event
def message(data):
    print('Message from server:', data)

# Connect to the server
sio.connect('http://localhost:3000')

# Send a message to the server
sio.emit('message_from_client', {'key': 'value'})

# Wait for events
#sio.wait()

def emitSocketEvent(eventName , data):
    print(data)
    sio.emit(eventName , data)


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

    target_width = 3840
    aspect_ratio = (32, 9)

    # Calculate the corresponding height based on the specified width and aspect ratio
    target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])

    # Create a window with the specified width and height
    window_name = 'Camera Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, target_width, target_height)


    ret, frame = cap.read()
    frame = cv2.resize(frame, (target_width, target_height))

    if not ret:
        print("Error: Could not read frame.")
        exit()

    cv2.imshow(image_name, frame)
    current_width, current_height = cv2.getWindowImageRect(window_name)[2:4]
    new_height = int(current_width / aspect_ratio[0] * aspect_ratio[1])
    cv2.resizeWindow(window_name, current_width, new_height)
    

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
        #squareSideLength = min(maxHeight , maxWidth)
        input_pts = np.float32([points[0], points[1], points[2], points[3]])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight ],
                                [maxWidth , maxHeight ],
                                [maxWidth , 0]])
    
        serializable_list = output_pts.tolist()
        json_data = json.dumps(serializable_list)
        emitSocketEvent("config", json_data)

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
    #cv2.imshow("Image Copy" ,img_copy)
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

        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        print("Original Coordinates: ", original_point)
        transformed_point = cv2.perspectiveTransform(np.array([[original_point]], dtype=np.float32), M)[0][0]
        #############################################################################################################################
        serializable_list = transformed_point.tolist()
        json_data = json.dumps(serializable_list)
        emitSocketEvent("live-coordinates" , json_data )
        #############################################################################################################################
        print("Perspective Coordinates: ", transformed_point)
        out = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
        #print("inputs are: ", input_pts)
        #print("outputs are: ", output_pts)
        # Display the output image
        #################################################
        #plt.figure()
        #mng = plt.get_current_fig_manager()
        #mng.window.showMaximized()
        #plt.imshow(out)
        
        #lip_out = cv2.flip(out,1)
        #plt.imshow(Flip_out)
        #plt.title("Perspective Transformed Image")
        #plt.show()
        #################################################
    else:
        print("Invalid dimensions. Please make sure the selected points form a valid quadrilateral.")


def liveTrack():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--consecutive-frames', default=4, type=int,
                        dest='consecutive_frames', help=tranformed_img_name)
    args = vars(parser.parse_args())

    # capture frames from the camera
    cap = cv2.VideoCapture(0)
    #target_width = 3840
    #aspect_ratio = (32, 9)
    #target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])

    # get the initial background frame from the camera
    background = None
    for _ in range(args['consecutive_frames']):
        _, background = cap.read()
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    consecutive_frame = args['consecutive_frames']
    while cap.isOpened():
        # Calculate the corresponding height based on the specified width and aspect ratio
        #window_name = 'Camera Feed'
        #target_height = int(target_width / aspect_ratio[0] * aspect_ratio[1])
        #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(window_name, target_width, target_height)
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            orig_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % consecutive_frame == 0 or frame_count == 1:
                frame_diff_list = []

            frame_diff = cv2.absdiff(gray, background)
            ret, thres = cv2.threshold(frame_diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            dilate_frame = cv2.dilate(thres, None, iterations=2)
            frame_diff_list.append(dilate_frame)

            if len(frame_diff_list) == consecutive_frame:
                sum_frames = sum(frame_diff_list)
                contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                center_point=(0,0)
                #print("contiurs: ", contours)
                for i, cnt in enumerate(contours):
                    cv2.drawContours(frame, contours, i, (0, 0, 255), 3)

                for contour in contours:
                    if not (200 <= cv2.contourArea(contour) <= 800):
                     continue
                    #print("Countour is: ", contour)
                    (x, y, w, h) = cv2.boundingRect(contour)
                    print(x,y,w,h)
                    cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center_point = (center_x, center_y)
                    #print("Center point is: ", center_point)

                    cv2.circle(orig_frame, center_point, 5, (255, 0, 0), -1)

                #cv2.imshow('Detected Objects', orig_frame)
                #window_name = 'Camera Feed'
                #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                #cv2.resizeWindow(window_name, target_width, target_height)
                #cv2.imshow(window_name, orig_frame)
                tranformImageLiveTrack(orig_frame,center_point)
                #out.write(orig_frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        else:
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()


clickReferencePic()

tranformTheReferenceImage()

liveTrack()


sio.wait()