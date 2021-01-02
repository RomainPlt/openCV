#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:12:31 2020


utils 


@author: romain
"""

import numpy as np
import cv2

################################## Shapes and texts

def draw_shape(img,shape,dim1, dim2, color, thickness):

    #img = np.zeros((512,512,3),np.uint8)
    #img[200:300,200:300] = 135,45,56
    # example cv2.line(img,(0,0),(200,300),(0,0,255),4) #(img,startPt, EndPt, Color, Thickness)
    if shape == "line":
        return cv2.line(img,dim1,dim2,color,thickness) #(img,startPt, EndPt, Color, Thickness)
        
    if shape == "rectangle":
        return cv2.rectangle(img,dim1,dim2,color,thickness) # the same
    if shape == "circle":
        return cv2.circle(img,dim1,dim2,color,thickness) # (img, centerPt, radius, color, thickness)
        
def put_text(img, text, position, scale, color, thickness):
    return cv2.putText(img, text, position, cv2.FONT_HERSHEY_COMPLEX, scale, color,thickness) #(img, text, centerPt, Font, Scale, Color, thickness)
   


##################################    Image modifications
    
def img_blur(img, nb1, nb2):
    return cv2.GaussianBlur(img, nb1, nb2)

def img_canny(img, nb1, nb2):
    return cv2.canny(img, nb1, nb2)

def img_crop(img,crop11,crop12,crop21,crop22):
    return img[crop11:crop12,crop21:crop22]


img = cv2.imread("nini.JPG")
kernel = np.ones((11,11),np.uint8)
print(img.shape)
imgResize = cv2.resize(img,(400,300)) #OpenCV : Width first, then high
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGrayBlur = cv2.GaussianBlur(imgGray, (7,7),50)
imgCanny = cv2.Canny(img, 20, 20)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgCropped = img[0:150,50:250] #Not OpenCV so first high and then width
cv2.imshow("Gray Image", imgGray)

##################################   Camera video capture
cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
