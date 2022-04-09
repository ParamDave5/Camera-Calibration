
import numpy as np
from scipy import optimize as opt
import cv2
import matplotlib.pyplot as plt
#vij = [hi1 hj1 , hi1 hj2 + hi2 hj1 , hi2 hj2 , hi3 hj1 + hi1 hj3 , hi3 h j2 + hi2 hj3 , hi3 hj3 ].T

def v(i,j,H):
    v = [[H[0,i]*H[0,j] ] ,
         [H[0,i]*H[1,j] + H[1,i]*H[0,j] ] ,
         [H[1,i]*H[1,j] ],
         [H[2,i]*H[0,j] + H[0,i]*H[2,j] ] ,
         [H[2,i]*H[1,j] + H[1,i]*H[2,j]] ,
         [H[2,i]*H[2,j]] ]
    return np.array(v)

def intrinsicParameters(H):
    V = []
    for i in H:
        #check for transpose
        V.append(v(0,1,i).T)
        V.append((v(0,0,i) - v(1,1,i)).T)
    #V.b = 0
    V = np.array(V).reshape(26,6)
    #calculate b by svd of v
    
    u , s  , Vt = np.linalg.svd(V)
    # print(bh)

    b = Vt[np.argmin(s)]
    # print('b is ',b)
    # print(b)
    B11 = b[0] 
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)/B11)
    alpha = np.sqrt(lamda /B11)
    beta = np.sqrt((lamda*B11) / (B11*B22 - B12**2))
    #gamma value  error 
    gamma = (-1*B12*(alpha**2)*beta)/lamda
    u0 = gamma*v0 / beta - B13*(alpha**2)/lamda

    K = np.float64([[alpha , gamma ,u0],
                  [0     , beta  ,v0] ,
                  [0     , 0     ,1]])
    return K

def extrinsicParameters(A , H):
    A_inv = np.linalg.inv(A)
    
    lamda = 1/np.linalg.norm(np.dot(A_inv,H[:,0]) , ord=2)
    r1 = lamda * np.dot(A_inv,H[:,0])

    r2 = lamda * np.dot(A_inv,H[:,1])

    r3 = np.cross(r1 ,r2)

    t = lamda * np.dot(A_inv , H[:,2])
    
    return np.stack((r1.T,r2.T,r3.T,t.T) , axis = 1)


def reprojectionError(K , R , imgPoints , objPoints):

    P = np.dot(K,R)
    error = []
    
    for imgPoint , objPoint in zip(imgPoints , objPoints):

        modelWorldCoordinate = np.array([ objPoint[0] , objPoint[1] ,0 , 1])
        imagePtsCoordinate = np.array([   imgPoint[0] , imgPoint[1] , 1])

        projectionPoint = np.dot(P, modelWorldCoordinate)
        #homogeneous coordinates
        projectionPoint = projectionPoint/projectionPoint[2]
        error.append(np.linalg.norm(imagePtsCoordinate - projectionPoint , ord = 2) )
    return np.mean(error)

def minimized(init , imgPoints , objPoints , H):
    imgPoints = np.array(imgPoints , dtype = np.float32)
    objPoints = np.array(objPoints , dtype = np.float32)
    H = np.array(H , dtype = np.float32)
    # print("Printing : .." , init)
    k = np.zeros(shape = (3,3),dtype = np.float64)
    k[0,0] , k[1,1] , k[0,2] , k[1,2] , k[0,1] , k[2,2] =init[0] , init[1] , init[2] ,init[3] , init[4] ,1
    k1 , k2 = init[5] , init[6]
    # k1 , k2 = 0,0
    u0 , v0 = init[2] , init[3]
    # print('printing k :  ' , k)

    error = []
    for imagePoint , i in zip(imgPoints , H):
        R = extrinsicParameters(k , i)
        for pt , objpt in zip(imagePoint , objPoints):
            model = np.array([[objpt[0]] ,[objpt[1]] , [0] , [1] ])

            proj_point = np.float64(np.dot(R,model))
            proj_point = proj_point/proj_point[2]

            x,y = proj_point[0] , proj_point[1] 
            t = x**2 + y**2
            U = np.float64(np.dot(k , proj_point))
            
            U = U/U[2]
            u,v = U[0] , U[1]

            u_hat = u + (u-u0)*( (k1*t) + k2*(t**2) )
            v_hat = v + (v-v0)*( (k1*t) + k2*(t**2) )
            

            error.append(pt[0] - u_hat)
            error.append(pt[1] - v_hat)
    return np.float32(error).flatten()

def optimization(k,imgPoints , objPoints , H):

    alpha = k[0,0]
    beta = k[1,1]
    u0 = k[0,2]
    v0 = k[1,2]
    gamma = k[0,1]

    initial = [alpha , beta , u0 , v0 , gamma, 0 , 0 ]
    optimized = opt.least_squares(fun = minimized , x0 = initial  , method = 'lm' , args = [imgPoints , objPoints , H])
    [alpha , beta , u0 , v0 , gamma , k1 , k2] = optimized.x
    K = np.array([[alpha , gamma , u0] , [0,beta , v0 ] , [0,0,1]])
    return K , k1 , k2 


def reprojectionErrorDistortion(k , R , imgPoints , objPoints , k1 , k2):
    error = []
    reprojection_points = []

    u0 , v0 = k[0,2] , k[1,2]

    for pt , objpt in zip(imgPoints , objPoints):
            model = np.array([[objpt[0]] ,[objpt[1]] , [0] , [1] ])

            proj_point = np.float64(np.dot(R,model))
            proj_point = proj_point/proj_point[2]

            x,y = proj_point[0] , proj_point[1] 
            t = x**2 + y**2
            U = np.float64(np.dot(k , proj_point))
            
            U = U/U[2]
            u,v = U[0] , U[1]

            u_hat = u + (u-u0)*[ (k1*t) + k2*(t**2) ]
            v_hat = v + (v-v0)*[ (k1*t) + k2*(t**2) ]
            reprojection_points.append([u_hat , v_hat])

            err = np.sqrt(((pt[0] - u_hat)**2 + (pt[1] - v_hat)**2))
            error.append(err)
            
    return error , reprojection_points

def displayPoints(imgPoints , objPoints , images):
    count = 1
    for imgPoint , objPoint , image in zip(imgPoints , objPoints , images):
        
        for imPt , ojPt in zip(imgPoint , objPoint):
            [x,y] = np.int64(imPt)
            [x1,y1] = np.int64(ojPt)
            x1 ,y1 = int(x1),int(y1)
            # print("img pts: " , [x,y] )
            # print("obj pts: " , [x1,y1])
            cv2.rectangle(image , (x-1 , y-1) , (x+1,y+1) , (255,0,0) , -1)
            cv2.rectangle(image , ( abs(x1-1) , abs(y1-1) ) , ( x1+5 , y1+5 ) , (0,0,255) , -1)
        # cv2.imwrite("/Users/sheriarty/Desktop/CMSC733/HW1/Outputs/reprojected{}.jpg".format(count) , image)
        count +=1

def rectifyImages(imgPoints , objPoints , images):
    
    # objPoints = np.array(objPoints ,  dtype = np.float64)
    count = 1
    for imgPoint , objPoint , image in zip(imgPoints , objPoints , images):
        
        # print("Image Point: ",imgPoint)
        pts = []
        for i in range(len(objPoint)):
            a = [ float(objPoint[i][0][0]) , float(objPoint[i][1][0]) ]
            pts.append(a)
        pts = np.array(pts)
        # print("Object Point: ",float(objPoint[0][1][0]))
        H , _ = cv2.findHomography(imgPoint , pts , method = cv2.RANSAC)
        a , b  = image.shape[1] , image.shape[0]
        warp = cv2.warpPerspective(image , H ,(a,b))
        # plt.imshow(warp)
        # plt.title("rectified images")
        # plt.show()
        # cv2.imwrite("/Users/sheriarty/Desktop/CMSC733/HW1/Outputs/rectified{}.jpg".format(count) , warp)
        count += 1


        