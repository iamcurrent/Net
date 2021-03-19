import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy.fft.helper import fftshift
from pandas.core.base import DataError
from scipy.io import savemat 
from collections import Counter
from sklearn.cluster import KMeans
import cv2
import pywt
import math
import os

def getMaxValue(data):
    return np.max(data)

def getMeanValue(data):
    return np.mean(data)

def getdata(file_path):
    data = []
    count = 0
    with open(file_path) as f:
        reader = csv.reader(f)
        for o in reader:
            if count==0:
                count+=1
                continue
            data_t = []
            for j in range(len(o)-1):
                data_t.append(float(o[j]))
            data.append(data_t)
    data_ss = np.array(data)
    return data_ss

def getselfdata(path):
    with open(path) as f:
        reader = csv.reader(f)
        x = []
        y = []
        for o in reader:
            x.append(np.float(o[3].strip()))
            y.append(np.float(o[4].strip()))

        return x,y
if __name__=="__main__":
    path = 'I:\\电源测试数据\\CSVFile\\'
    fileName = os.listdir(path)
    file1 = fileName[len(fileName)-1]
    res = []
    with open(path+file1) as f:
        reader = csv.reader(f)
        for o in reader:
            res.append(float(o[1]))
    mins = np.min(res)
    maxs = np.max(res)
    aver = np.average(res)
    fft = np.fft.fft(res)
    fftshift = np.fft.fftshift(fft)
    result = np.abs(fftshift)
    print(mins,maxs,aver)
    plt.plot(result[int(len(result)/2):len(result)])
    plt.show()
   
    