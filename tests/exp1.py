import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('t3.jpg') #logo.jpg
img2 = cv2.imread('image.jpg') #image.jpg

gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#descriptor = cv2.xfeatures2d.SIFT_create()
descriptor = cv2.ORB_create()

kp1, ds1 = descriptor.detectAndCompute(gray1,None)
kp2, ds2 = descriptor.detectAndCompute(gray2,None)

cv2.drawKeypoints(gray1,kp1, img1)
cv2.drawKeypoints(gray2,kp2, img2)


# Usar NORM_L2 no lugar de NORM_HAMMING quando se usa SIFT ou SURF
#bf = cv2.BFMatcher(cv2.NORM_L2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(ds1,ds2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

plt.imshow(img3)
plt.show()
