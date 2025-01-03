#CV2 FILTER IMAGES
"""
Created on Sun Dec 29 22:59:47 2024

@author: TH3 PRO HER0, MAU BAUTISTA
PYTHON VERSION 3.12
OS W11
"""
import cv2
import numpy as np
bananos=cv2.imread(r'F:\OpenCV\resources\bananos.jpg') #your path

cv2.imshow('Original', bananos)

kernel_3x3=np.ones((3,3))/(3*3)

output3x3= cv2.filter2D(bananos, -1, kernel_3x3)

cv2.imshow('filtro3x3', output3x3)

#Comparation

kernel_11x11=np.ones((11,11))/(11*11)

output11x11= cv2.filter2D(bananos, -1, kernel_11x11)

cv2.imshow('filtro11x11', output11x11)

#comparation 2

kernel_31x31=np.ones((31,31))/(31*31)

output31x31= cv2.filter2D(bananos, -1, kernel_31x31)

cv2.imshow('filtro31x31', output31x31)
