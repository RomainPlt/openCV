#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:09:36 2020

Get colors

@author: romain
"""
import numpy as np
import cv2

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
    img = cv2.imread("random.jpeg")

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
    
    
    

