import cv2
import skimage.io as ski
import numpy as np
from sklearn import svm
import matplotlib.pylab as plt
img = cv2.imread('test.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)


ski.load_sift('test2.jpg')