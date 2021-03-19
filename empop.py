import numpy as np
import cv2
import imutils



if __name__=='__main__':
    image = cv2.imread('I:\\Desk\\tf2.0study\\1.jpg',0)
    
   

    r_left, r_right = 150, 230
    r_min, r_max = 0, 255
    level_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if r_left <= image[i, j] <= r_right:
                level_img[i, j] = r_max
            else:
                level_img[i, j] = image[i, j]

    


    
    #拉普拉斯算法增强
    kernel = np.array([ [0, -1, 0],  
                    [-1,  5, -1],  
                    [0, -1, 0] ]) 
    image_lap = cv2.filter2D(image,cv2.CV_8UC3 , kernel)
    
    #对象算法增强
    image_log = np.uint8(np.log(np.array(image) +1))    
    cv2.normalize(image_log, image_log,0,255,cv2.NORM_MINMAX)
    #转换成8bit图像显示
    cv2.convertScaleAbs(image_log,image_log)

    #伽马变换
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image)/255.0),fgamma)*255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)

    ##自适应直方图均衡化
    cla = cv2.createCLAHE(2.0,(8,8))
    res = cla.apply(image)

    cv2.imshow('1',image_lap)
    cv2.imshow('2',image_log)
    cv2.imshow('3',image_gamma)
    cv2.imshow('4',res)
    cv2.imshow('5',level_img)
    
    cv2.waitKey()
  