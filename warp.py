#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:11:15 2020

image crop + warp perspectives

@author: romain
"""


import cv2
import numpy as np


################################## Redressage d'image

def warp_perspective(img, pts1, pts2, width, height):
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutPut = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutPut


    
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


