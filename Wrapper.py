from email.policy import default
from select import kevent
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

from misc.homography import *
from misc.parameters import *

parser = argparse.ArgumentParser(description='Input the checkerboard images.')
parser.add_argument('--Path', default = "Calibration_Imgs" , help="path to image folder")
parser.add_argument('--Show' , default = "False" ,help = "Show Outout Images" )

args = parser.parse_args()
Path = args.Path + "/*.jpg"
show = args.Show

#Import Images
images = [cv2.imread(file) for file in glob.glob(Path)]
og = images.copy()

#Calculate Homography

homography , imgPoints , objPoints = Homography(images , show)

#Calculate Initial Intrinsic and Extrinsic parameters
k = intrinsicParameters(homography)
print("Initial intrinsic parameters: " ,k)

error = []
R_ = []
# print('Initial Intrinsic Parameter MAtrix', k)
for H , imgPoint  in zip(homography , imgPoints ):
    R = extrinsicParameters(k , H)
    R_.append(R)
    err = reprojectionError(k , R , imgPoint , objPoints)
    error.append(err)

error = np.mean(error)
print('Mean Projection Error: ', error)

k = np.array(k , dtype = np.float64)
imgPoints = np.array(imgPoints,dtype=np.float64)
homography = np.array(homography ,dtype=np.float64)
objPoints = np.array(objPoints , dtype=np.float64)

# print(k.dtype)
# print(imgPoints.dtype)
# print(objPoints.dtype)
# print(homography.dtype)
k_final  , k1 , k2 = optimization(k , imgPoints , objPoints , homography)

pts = []
error_= []
for imgPoint , i in zip(imgPoints , homography):
    R = extrinsicParameters(k_final , i)
    err , points = reprojectionErrorDistortion(k_final , R , imgPoint , objPoints , k1 , k2)
    pts.append(points)
    error_.append(err)

error_ = np.mean(error_)
print("Projection Error after Calibration : " , error_)
print("Final Intrinsic Parameter Matrix : " , k_final)
print("Final Distortion co-efficients : " , k1 ,k2)


if show == "True":
    displayPoints(imgPoints ,pts , og )

final_images = rectifyImages(imgPoints , pts , og)












    












