import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from scipy.optimize import curve_fit

#model points
def Homography(images ,show):
    #model points
    b   = []
    imgpoints = []
    #creating model points which are 21.5 apart in each direction
    for i in range(6):
        for j in range(9):
            a = [j,i]
            b.append(a)
    b = np.float64(b)
    b = b*21.5
    homography = []

    for i , image in enumerate(images):
        img=np.uint8(image)
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        retval , corners = cv2.findChessboardCorners(gray , (9,6),None)

        
        if retval == True:
            count = 1
            corners = corners.reshape(-1,2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER ,40 , 0.001)

            #accuracy of corner points
            corners = cv2.cornerSubPix(gray , corners , (15,15) , (-1,-1) , criteria)
            imgpoints.append(corners)
            
            # print(imgpoints)
            H , status = cv2.findHomography(b[:30] , corners[:30], method = cv2.RANSAC)
            homography.append(H)
            if show == "True":
                
                img = cv2.drawChessboardCorners(img , (9,6) , corners , retval)
                cv2.imwrite("/Users/sheriarty/Desktop/CMSC733/HW1/Calibration_Imgs/Outputs/corners{}.jpg".format(i+1) , image)
                # plt.imshow(img)
                # plt.show()
            #homography ,image poinyts , object points
    return homography , imgpoints , b 
        

