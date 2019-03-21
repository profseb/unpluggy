import os;
import cv2 as cv
import numpy as np

from pathlib import Path

class Unpluggy:

	DEFAULT_EXT = ".jpg"

	detector = False
	target = False
	target_features = False
	keypoints_descriptors = []
	blocks_list = []
	blocks_path = 'images/'
	keypoints_path = 'keypoints/'
	
	def __init__(self):
		
		self.detector = cv.xfeatures2d_SIFT.create()

	def extractFilenames(self, itens):

		l = []
		for item in itens:
			l.append(Path(item).stem)
		return l

	def checkSource(self):
		
		keypoints_list = self.extractFilenames(os.listdir(self.keypoints_path))		
		self.blocks_list = self.extractFilenames(os.listdir(self.blocks_path))		
		
		return (set(keypoints_list) == set(self.blocks_list))

	def buildKeypoints(self):
	
		for item in self.blocks_list:
			imfile = self.blocks_path+item+self.DEFAULT_EXT
			imcv = cv.imread(imfile, cv.IMREAD_GRAYSCALE)
			keypoints, descriptors = self.detector.detectAndCompute(imcv, None)					
			filename = self.keypoints_path+item+'.npy'
			np.save(filename, Utils.packKeypoints(keypoints, descriptors))

	def loadKeypointsAndDescriptors(self):

		for item in self.blocks_list:			
			filename = self.keypoints_path+item+".npy"
			self.keypoints_descriptors.append(np.load(filename))			
					
	def loadBlocks(self):
		
		if self.checkSource() == False:
			self.buildKeypoints()

		self.loadKeypointsAndDescriptors()

	def matchKeypoints(self, idx):

		matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
		kp1, d1 = Utils.unpackKeypoints(self.keypoints_descriptors[idx])
		kp2, d2 = Utils.unpackKeypoints(self.target_features)		
		knn_matches = matcher.knnMatch(d1, d2, 2)	

		ratio_thresh = 0.75
		good_matches = []
		for m,n in knn_matches:
		    if m.distance < ratio_thresh * n.distance:
		        good_matches.append(m)

		obj = np.empty((len(good_matches),2), dtype=np.float32)
		scene = np.empty((len(good_matches),2), dtype=np.float32)

		for i in range(len(good_matches)):		
			obj[i,0] = kp1[good_matches[i].queryIdx].pt[0]
			obj[i,1] = kp1[good_matches[i].queryIdx].pt[1]
			scene[i,0] = kp2[good_matches[i].trainIdx].pt[0]
			scene[i,1] = kp2[good_matches[i].trainIdx].pt[1]

		return obj, scene


	def fillCorners(self, block):

		w = block.shape[1]
		h = block.shape[0]

		corners = np.empty((4,1,2), dtype=np.float32)

		corners[0,0,0] = 0
		corners[0,0,1] = 0
		corners[1,0,0] = w
		corners[1,0,1] = 0
		corners[2,0,0] = w
		corners[2,0,1] = h
		corners[3,0,0] = 0
		corners[3,0,1] = h


		return corners

	def drawBlock(self, target_corners):

		cv.line(self.target, (int(target_corners[0,0,0]), int(target_corners[0,0,1])),\
		    (int(target_corners[1,0,0]), int(target_corners[1,0,1])), (0,255,0), 4)
		cv.line(self.target, (int(target_corners[1,0,0]), int(target_corners[1,0,1])),\
		    (int(target_corners[2,0,0]), int(target_corners[2,0,1])), (0,255,0), 4)
		cv.line(self.target, (int(target_corners[2,0,0]), int(target_corners[2,0,1])),\
		    (int(target_corners[3,0,0]), int(target_corners[3,0,1])), (0,255,0), 4)
		cv.line(self.target, (int(target_corners[3,0,0]), int(target_corners[3,0,1])),\
		    (int(target_corners[0,0,0]), int(target_corners[0,0,1])), (0,255,0), 4)
		

	def matchBlocks(self):

		for idx in range(len(self.blocks_list)):						
		
			obj, scene = self.matchKeypoints(idx)		
			H, _ =  cv.findHomography(obj, scene, cv.RANSAC)		
			block = cv.imread(self.blocks_path+self.blocks_list[idx]+self.DEFAULT_EXT,cv.IMREAD_GRAYSCALE)
			corners = self.fillCorners(block)					
			target_corners = cv.perspectiveTransform(corners, H)
			self.drawBlock(target_corners)
		
		cv.imshow('Good Matches & Object detection', self.target)
		cv.waitKey(25000)

		
	def loadTarget(self, imsource):

		self.target = cv.imread(imsource, cv.IMREAD_COLOR)
		keypoints, descriptors = self.detector.detectAndCompute(self.target, None)
		self.target_features = Utils.packKeypoints(keypoints, descriptors)

		
	def proccess(self, target):
		
		self.loadBlocks()
		self.loadTarget(target)
		self.matchBlocks()


class Utils:

	def packKeypoints(keypoints, descriptors):

		i = 0
		temp_array = []
		for point in keypoints:	   			
			temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i])     
			i += 1	        
			temp_array.append(temp)
		return temp_array

	def unpackKeypoints(array):

	    keypoints = []
	    descriptors = []
	    for point in array:
	    	temp_feature = cv.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
	    	temp_descriptor = point[6]
	    	keypoints.append(temp_feature)
	    	descriptors.append(temp_descriptor)
	    return keypoints, np.array(descriptors)