"""
This script detects oving objects inn the video through the webcam
and creates a bokeh plot for it
author: Rahul
date: 28th Dec 2017
"""
import cv2, time
import datetime
import pandas as pd

video = cv2.VideoCapture(0)
##cascade classifier
#face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
##
##first frame
first_frame = None
status_list = []
time_list = []
time_sheet = pd.DataFrame()
while True:    
    status = 0
    frame, image = video.read(0)
    
    #print(frame)
    #print(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (21,21), 0)
    ##capture the first frame which is compared with all others
    if first_frame is None:
        first_frame = image_gray
        continue
    
    #print(first_frame)
    ##creating the delta frame
    delta_frame = cv2.absdiff(first_frame,image_gray)
    ##threshold frame
    thresh_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    #print(type(thresh_frame))
    #print(thresh_frame)

    #print(delta)
    (__,cnts,__) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contours in cnts:
        if cv2.contourArea(contours) < 10000:
            continue
        ##calculating the coordinates of rectangle
        x,y,w,h = cv2.boundingRect(contours)
        ##drawing the rectangle
        cv2.rectangle(image, (x,y), (x+w,y+h), color = (0,0,255), thickness = 4)
        status = 1
    status_list.append(status)
    if len(status_list) >= 2:
        if status_list[-1] == 1 and status_list[-2] == 0:
            time_list.append(datetime.datetime.now())
        if status_list[-1] == 0 and status_list[-2] == 1:
            time_list.append(datetime.datetime.now())
    
    cv2.imshow("thresh", thresh_frame)
    cv2.imshow("delta", delta_frame)
    cv2.imshow("image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            time_list.append(datetime.datetime.now())
        break

# time_diff = []
# if len(time_list) >= 2:
#     diff = (time_list[-1] - time_list[-2])
#     time_diff.append(diff)


#print(time_diff)
#print(status_list)

print(time_list)
print(len(time_list))

for i in range(0,len(time_list),2):
    time_sheet = time_sheet.append({"Entry_time":str(time_list[i]),
    "Exit_time":str(time_list[i+1])}, ignore_index=True)

time_sheet.to_csv("time_sheet.csv")


video.release()
cv2.destroyAllWindows()

# type(video)
# print(video)