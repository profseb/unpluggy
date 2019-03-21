import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('gain2.jpg',0) #logo.jpg
img2 = cv2.imread('bee4.jpg',0) #image.jpg

sift = cv2.xfeatures2d.SIFT_create()

kp1, ds1 = sift.detectAndCompute(img1,None)
kp2, ds2 = sift.detectAndCompute(img2,None)

#img2 = cv2.drawKeypoints(img2, kp2, None)

FLANN_INDEX_KDTREE = 0
flannParams = dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
flann = cv2.FlannBasedMatcher(flannParams,{})

matches = flann.knnMatch(ds1, ds2, k = 2)

#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#matches = bf.match(ds1,ds2)
#matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)



#outImage = cv.drawKeypoints(img1, kp1, outImage[, color[, flags]])
#outImg	 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])
#outImg   = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])

plt.imshow(img3)
plt.show()