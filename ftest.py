from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from numpy import ma
from numpy.lib.type_check import imag
from numpy.matrixlib.defmatrix import matrix
from tcn import TCN,tcn_full_summary
patht = "I:\\频谱数据\\2\\"
fileNames = os.listdir(patht)
dirs = ['2','3','z']
test_data = []
y_test = []
def getData(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        count = 0
        for o in reader:
            if count < 150000:
                data.append(np.float32(o[0]))
            count = count+1
    return data

def dataSparate(dataks):
    ytest = []
    datatest = []
    index1 = np.argmax(dataks)
    da1 = dataks[0:index1-5000]
    da2 = dataks[index1+5000:len(dataks)]
    max1 = np.max(da1)
    max2 = np.max(da2)
    if max1>max2:
        index2 = np.argmax(da1)
        test_data.append(da1[index2-5000:index2+5000])
        y_test.append(1)
    else:
        index2 = np.argmax(da2)
        test_data.append(dataks[index1+index2-5000:index1+index2+5000])
        y_test.append(1)
    test_data.append(dataks[index1-5000:index1+5000])
    y_test.append(0)





'''
savepath = "./Images/"
count = 0
for o in dirs:
    path = patht+o+"/"
    fileNames = os.listdir(path)
    if o!='z':
        for o1 in fileNames:
            res = getData(path+o1)
            indexmax = np.argmax(res)
            data =  res[indexmax-5000:indexmax+5000]
            #plt.plot(data)
            #plt.show()
            img = np.matrixlib.matrix(data).T*np.matrixlib.matrix(data)
            img = np.array(img,dtype=np.float32)
            img = cv2.resize(img,dsize=(1000,1000))
            cv2.imwrite(savepath+str(count)+".png",img)
            count=count+ 1    
'''



'''
res = getData(patht+fileNames[4])
depart = res[60000:70000]


img = np.matrixlib.matrix(depart).T*np.matrixlib.matrix(depart)
img = np.array(img,dtype=np.float32)
r = 255/np.max(img)
img = img*r




print(img.shape)


maxv = np.max(img)
r = 255/np.max(img)

img = cv2.resize(img,dsize=(512,512))
plt.imshow(img)
#plt.plot(depart)
plt.show()
'''
#cv2.imshow('img',img)
#cv2.waitKey()







