import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import time

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# For the refrence image
image_name = f"captured_image_{current_date}.jpg"
tranformed_img_name = f"transformed_image_{current_date}.jpg"
points = []

THRESHOLD_VALUE = 30

def onclick(event):
    if event.button == 1:  # Left mouse button click
        points.append((event.xdata, event.ydata))
        plt.scatter(event.xdata, event.ydata, c='red')  # Mark the selected point
        plt.text(event.xdata, event.ydata, f"{len(points)}", color='red', fontsize=12, ha='right', va='bottom')
        plt.draw()
        if len(points) >= 4:
            plt.close()



def tranformImage(imagePath , imageFile , imageName):
    img = None
    if imageFile is None:
        img = cv2.imread(imagePath)
    
    if imageFile is not None:
        img = imageFile

    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()

    if(len(points) == 0):
        plt.imshow(img)
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
        ##plt.figure()
        #mng = plt.get_current_fig_manager()
        #mng.window.showMaximized()
        ##plt.imshow(out)
        #lip_out = cv2.flip(out,1)
        #plt.imshow(Flip_out)
        ##plt.title("Perspective Transformed Image")
        ##plt.show()

        if(imageName):
           cv2.imwrite(imageName,out)
        
        return out



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


def liveTrack():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
      ret, frame = cap.read()
      if ret:
        orig_frame = frame.copy()
        tranformImage('',orig_frame , 'temp.jpg')
        #refrenceImage = cv2.imread(tranformed_img_name , cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(tranformed_img_name)
        image2 = cv2.imread('temp.jpg')
        
        gray1 = cv2.cvtColor(image1 , cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2 , cv2.COLOR_BGR2GRAY)

        #grayFrame = cv2.imread('temp.jpg' , cv2.IMREAD_GRAYSCALE)
        diff = cv2.absdiff(gray2 , gray1)
        #diff = cv2.GaussianBlur(diff, (5, 5), 0)
        #edges = cv2.Canny(diff, 25, 200 )
        _, thresholded = cv2.threshold(diff , 50 , 255 , cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("TOtal Countours: ", len(contours))
        tempImage2 = image2
        final_x =-1
        final_y =-1
        final_w =-1
        final_h =-1
        max_area =-1
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w*h
            if(area>=max_area):
                final_x =x
                final_y =y
                final_w =w
                final_h =h
                max_area = area

        center_x = final_x+final_w //2
        center_y = final_y+final_h //2
        cv2.rectangle(tempImage2, (final_x, final_y), (final_x +final_w, final_y + final_h), (0, 0, 255), 2)
        print((center_x , center_y))
        cv2.circle(tempImage2, (center_x, center_y), 5, (0, 255, 0), -1)
       

        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     point1 = (x,y)  # Top-Left
        #     point2 = (x+w , y)  # Top-right
        #     point3 = (x , y+h)  # Bottom-left 
        #     point4 = (x+w , y+h) # Bottom-right
        #     center_x = x+w //2
        #     center_y = y+h //2
        
        #     if center_x>=x and center_x <=x+w and center_y>=y and center_y<=y+h:
        #         cv2.rectangle(tempImage2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #         print((center_x , center_y))
        #         cv2.circle(tempImage2, (center_x, center_y), 5, (0, 255, 0), -1)
        #         break
        
           
            
        plt.imshow(tempImage2)
        plt.title("Contours")
        plt.show()
        time.sleep(0.5)
        #cv2.drawContours(image1, contours, -1, (0, 255, 0), 2)
        #plt.imshow(image1)
        #plt.title("Contours")
        #plt.show()




clickReferencePic()

tranformImage(image_name , None , tranformed_img_name)

liveTrack()