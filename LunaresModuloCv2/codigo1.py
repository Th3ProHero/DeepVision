#
#CV2 CODES
#THIS IS THE FIRST ONE, OPEN IMAGES.

import cv2
import numpy as np
import matplotlib.pyplot as plt

bananos=cv2.imread(r'F:\OpenCV\resources\bananos.jpg')
b=bananos[:,:,0]
g=bananos[:,:,1]
r=bananos[:,:,2]
 
#cv2.imshow('',bananos)
#cv2.imshow('',b)
#cv2.imshow('',g)
#cv2.imshow('',r)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#WITHOUT RGB CHANNELS
img_gray=cv2.cvtColor(bananos,cv2.COLOR_BGR2GRAY)

#IMAGE ON BINARY SCALE
binaria = np.uint8(255*(img_gray<233))

gray_segmentada=np.uint8(img_gray*(binaria/255))


#cv2.imshow('',binaria)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

seg_color=bananos.copy()

seg_color[:,:,0]=np.uint8(b*(binaria/255))
seg_color[:,:,1]=np.uint8(g*(binaria/255))
seg_color[:,:,2]=np.uint8(r*(binaria/255))

#cv2.imshow('',seg_color)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.hist(img_gray.flatten(),bins=15)
#plt.show() 



#Otsu method
Th_otsu,_=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

binaria_otsu=np.uint8(255*(img_gray<Th_otsu))

#LUNAR
lunar=cv2.imread(r'F:\OpenCV\resources\lunar.jpg')

lunar_gray=cv2.cvtColor(lunar,cv2.COLOR_BGR2GRAY)

thlunar_otsu,_=cv2.threshold(lunar_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

lunar_otsu=np.uint8(255*(lunar_gray<thlunar_otsu))



