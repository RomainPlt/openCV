# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:28:35 2020

@author: Romain
"""

import cv2
import numpy as np

def empty(a):
    pass

 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


################################## Face Detection


faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
img = cv2.imread("nini.JPG")
imgResize = cv2.resize(img,(400,300))
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(imgResize,(x,y),(x+w, y+h), (255,0,0), 2)
    

    
cv2.imshow("face",  imgResize)
cv2.waitKey(0)

"""
################################## Color Detection

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue min", "TrackBars",0,179,empty)
cv2.createTrackbar("Hue max", "TrackBars",179,179,empty)
cv2.createTrackbar("Sat min", "TrackBars",0,255,empty)
cv2.createTrackbar("Sat max", "TrackBars",255,255,empty)
cv2.createTrackbar("Val min", "TrackBars",0,255,empty)
cv2.createTrackbar("Val max", "TrackBars",255,255,empty)

while True:
    img = cv2.imread("nini.JPG")

    imgResize = cv2.resize(img,(400,300)) #OpenCV : Width first, then high
    imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val max", "TrackBars")

    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)    
    imgResult = cv2.bitwise_and(imgResize,imgResize,mask=mask)
    
    

    # cv2.imshow("Original", imgResize)
    # cv2.imshow("HSV", imgHSV)
    # cv2.imshow("mask", mask)
    # cv2.imshow("Results", imgResult)
    
    imgStack = stackImages(0.2, ([img,imgHSV],[mask,imgResult]))
    cv2.imshow("stacked Images", imgStack)
    
    cv2.waitKey(1)

"""
    
    
"""
################################## Redressage d'image
img = cv2.imread("Redresse.jpg")
imgResize = cv2.resize(img,(400,700)) #OpenCV : Width first, then high
width, height = 400, 700
pts1 = np.float32([[179,619], [763,561],[253,1401],[1030,1260]])
pts2 = np.float32([[0,0], [width, 0], [0,height], [width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutPut = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Image", imgResize)
cv2.imshow("Feuille", imgOutPut)


cv2.waitKey(0)
"""

"""
################################## Shapes and texts

img = np.zeros((512,512,3),np.uint8)

#img[200:300,200:300] = 135,45,56
cv2.line(img,(0,0),(200,300),(0,0,255),4) #(img,startPt, EndPt, Color, Thickness)
cv2.rectangle(img,(0,0),(200,300),(0,255,0),6) # the same
cv2.circle(img,(300,150), (100),(255,0,0),8) # (img, centerPt, radius, color, thickness)
cv2.putText(img, "Coucou", (0,400), cv2.FONT_HERSHEY_COMPLEX, 4, (20,130,200),2) #(img, text, centerPt, Font, Scale, Color, thickness)

cv2.imshow("Image", img)


cv2.waitKey(0)
"""

"""
##################################    Image modifications
img = cv2.imread("nini.JPG")
kernel = np.ones((11,11),np.uint8)
print(img.shape)
imgResize = cv2.resize(img,(400,300)) #OpenCV : Width first, then high
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGrayBlur = cv2.GaussianBlur(imgGray, (7,7),50)
imgBlur = cv2.GaussianBlur(img, (51,51), 0)
imgCanny = cv2.Canny(img, 20, 20)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgCropped = img[0:150,50:250] #Not OpenCV so first high and then width
cv2.imshow("Gray Image", imgGray)
"""

"""
##################################   Camera video capture
cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
"""
