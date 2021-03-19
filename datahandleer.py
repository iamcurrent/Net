
import csv
import os
from os import path, read
import pandas as pd
import numpy as np    #进行具体的sum,count等计算时候要用到的
import matplotlib.pyplot as plt
def getData(path):
    data = []
    with open(path,'r') as f:
        reader = csv.reader(f)
        count = 0
        for o in reader:
            if count < 150000:
                data.append(np.float32(o[0]))
            count = count+1
    return data


def getDataSimple(path):
    data = []
    with open(path,'r') as f:
        reader = csv.reader(f)
        for o in reader:
            for i in range(0,len(o)-1):
                data.append(np.float32(o[i]))
    return data




def handleSimple(path,fileNames):
    resData = []
    for i in  range(len(fileNames)):
        res = getData(path+fileNames[i])
        indexmax = np.argmax(res)
        resData.append(res[indexmax-5000:indexmax+5000])
        
    return resData

def insertValue(data,ln):
    for i in range(ln-len(data)):
        data.append(0)
    return data


def handleMix(path,fileNames,y):
    ytest = []
    datatest = []
    for i in range(len(fileNames)):
        res = getData(path+fileNames[i])
        index1 = np.argmax(res)
        da1 = list(res[0:index1-5000])
        da2 = list(res[index1+5000:len(res)])
        max1 = np.max(da1)
        max2 = np.max(da2)
        if max1>max2:
            index2 = np.argmax(da1)
            sd = da1[index2-5000:index2+5000]
            if len(sd)<10000:
                sd = insertValue(sd,10000)
            datatest.append(sd)
            ytest.append(y[0])
        else:
            index2 = np.argmax(da2)
            sd = res[index1+index2-5000:index1+index2+5000]
            if len(sd)<10000:
                sd = insertValue(sd,10000)
            datatest.append(sd)
            
            ytest.append(y[0])
        sd = res[index1-5000:index1+5000]
        if len(sd)<10000:
            sd = insertValue(sd,10000)
        datatest.append(sd)
        
        ytest.append(y[1])
    return (datatest,ytest)

'''
path = "I:\\频谱数据\\"
fileNames1 = os.listdir(os.path.join(path,'2'))
fileNames2 = os.listdir(os.path.join(path,'3'))
fileNames3 = os.listdir(os.path.join(path,'z'))
fileNames4 = os.listdir(os.path.join(path,'ztest'))
fileNames3 = ["z/"+o for o in fileNames3]
fileNames4 = ["ztest/"+o for o in fileNames4]
yz = [2,3]
fileNames1 = ["2/"+o for o in fileNames1]
y1 = [2 for i in range(len(fileNames1))]
y2 = [3 for i in range(len(fileNames2))]
fileNames2 = ["3/"+o for o in fileNames2]
fileNames = np.concatenate((fileNames1,fileNames2),0)
ys = np.concatenate((y1,y2),0)
data = handleSimple(path,fileNames)
data1,yz = handleMix(path,fileNames3,yz)
save = "./data.csv"
with open(save,"a",newline="") as f:
    writer = csv.writer(f)
    for i in range(len(data)):
        d = data[i]
        d.append(ys[i])
        writer.writerow(d)
    for i in range(len(data1)):
        d = data1[i]
        d.append(yz[i])
        writer.writerow(d)
'''
'''
data = np.array(data)
data1 = np.array(data1)
print(data.shape,data1.shape)
'''
'''
paths = "./data.csv"
data = []
y = []
with open(paths) as f:
    reader = csv.reader(f)
    for o in reader:
        y.append(np.int(o[len(o)-1]))
        data.append(np.float32(o[:len(o)-2]))
print(data)
        
'''

