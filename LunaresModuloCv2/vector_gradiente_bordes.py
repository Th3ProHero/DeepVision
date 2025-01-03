#CVector Gradiente
"""
Created on Sun Dec 29 22:59:47 2024

@author: TH3 PRO HER0, MAU BAUTISTA
PYTHON VERSION 3.12
OS W11
"""
import cv2
import numpy as np
import time
bananos=cv2.imread(r'F:\OpenCV\resources\bananos.jpg')

gray=cv2.cvtColor(bananos, cv2.COLOR_BGR2GRAY)

gx=cv2.Sobel(gray,cv2.CV_64F,1,0,5)
gy=cv2.Sobel(gray,cv2.CV_64F,0,1,5)

mag,ang=cv2.cartToPolar(gx, gy)

mag=np.uint8(255*mag/np.max(mag))

cv2.imshow('mag', mag)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Versión de OpenCV:", cv2.__version__)
sift = cv2.SIFT_create()
print("SIFT creado exitosamente.")
