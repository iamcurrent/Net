import os
import csv
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from tensorflow import keras
from tensorflow.python.ops.gen_math_ops import sign
import tensorflow as tf
import cv2
sc = MinMaxScaler(feature_range=(0, 1))
path1 = "I:\\电源测试数据\\CSVTest\\"
path = "I:\\电源测试数据\\CSVFile\\"
fileNames = os.listdir(path)
fileNames1 = os.listdir(path1)
y_data  = []
mixs = []
timestep = 200
##数据集构造
def dataDetail(path):
    
    res = []
    with open(path) as f:
        reader = csv.reader(f)
        mix = []
        s1 = []
        s2 = []
        for o in reader:
            mix.append(float(o[1]))
            s1.append(float(o[2]))
            s2.append(float(o[3]))
        return mix,s1,s2

##数据集构造
def dataDetai2(path):
    res = []
    with open(path) as f:
        reader = csv.reader(f)
        mix = []
        s1 = []
        s2 = []
        for o in reader:
            mix.append(float(o[1]))
            s1.append(float(o[2]))
            s2.append(float(o[3]))
        mix  = mix - np.mean(mix)
        s1 = s1 - np.mean(s1)
        s2 = s2 - np.mean(s2)
        mix = filterBybutter(mix,50,200,4000)
        s1 = filterBybutter(s1,50,200,4000)
        s2 = filterBybutter(s2,50,200,4000)
        mix = np.round(mix,6)
        s1 = np.round(s1,6)
        s2 = np.round(s2,6)
        res.append(mix)
        res.append(s1)
        res.append(s2)
        
        mix = np.reshape(mix,(1,len(mix)))
        
        s1 = np.reshape(s1,(1,len(s1)))
  
        s2 = np.reshape(s2,(1,len(s2)))

        x = []
        y1 = []
        y2 = []
        for i in range(0,len(mix[0])-int(timestep/2),int(timestep/2)):
            x.append(np.reshape(mix[0,i:i+timestep],(1,timestep)))
            y1.append(np.reshape(s1[0,i:i+timestep],(1,timestep)))
            y2.append(np.reshape(s2[0,i:i+timestep],(1,timestep)))
        
        c = len(x)
        x_train = np.reshape(x,(c,timestep,1))
        y_train1 = np.reshape(y1,(c,timestep,1))
        y_train2 = np.reshape(y2,(c,timestep,1))
        return x_train,(y_train1,y_train2),res

def getData(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        for o in reader:
            data.append(float(o[1]))
    return data

##带通滤波
def filterBybutter(data,f1,f2,fs):
    f1 = 2*f1/fs
    f2 = 2*f2/fs
    b, a = signal.butter(2, [f1,f2], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
    x = signal.filtfilt(b, a, data)  #data为要过滤的信号
    return x
##使用卷积实现移动平均
def averFilter(data,n):
    return np.convolve(data, np.ones((n,))/n, mode='same')
'''
x = []
y1 = []
y2 = []
for i in range(1000):
    x.append(np.random.normal(size=(200,1)))
    if i%2 == 0:
        y1.append(1)
        y2.append(0)
    else:
        y1.append(0)
        y2.append(1)
x = np.reshape(x,(len(x),200,1))
y1 = np.reshape(y1,(len(y1),1,1))
y2 = np.reshape(y2,(len(y2),1,1))
y = (y1,y2)
'''
'''
x,y,_ = dataDetai2(path+fileNames[0])

input1 = keras.layers.Input(shape=(timestep,1))
o = keras.layers.Conv1D(32,3,activation='relu')(input1)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Flatten()(o)
out1 = keras.layers.Dense(timestep)(o)
out2 = keras.layers.Dense(timestep)(o)
model = keras.Model(input1,[out1,out2])
model.summary()
'''
#model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(0.001))
#model.fit(x,y,32,20)


'''
mix,s1,s2 = dataDetail(path+fileNames[0])
fs = 4000
f1 = 2*50/fs
f2 = 2*200/fs
b, a = signal.butter(2, [f1,f2], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
x = signal.filtfilt(b, a, mix)  #data为要过滤的信号
x1  = signal.filtfilt(b, a, s1)
x2  = signal.filtfilt(b, a, s2)
plt.subplot(3,1,1)
plt.plot(x)
plt.subplot(3,1,2)
plt.plot(x1)
plt.subplot(3,1,3)
plt.plot(x2)
plt.show()
'''
'''
mix,s1,s2 = dataDetail(path+fileNames[0])
res = np.convolve(mix, np.ones((100,))/100, mode='same')
plt.plot(res)
plt.show()
'''
#x,y,_ = dataDetai2(path+fileNames[0])
'''
input1 = keras.layers.Input(shape=(200,1))
o  = keras.layers.LSTM(80,return_sequences=True)(input1)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.LSTM(100)(o)
o = keras.layers.Dropout(0.2)(o)
o1 = keras.layers.Dense(200)(o)
model = keras.Model(input1,o1)
'''
'''
def dataConvert(y_predict):
    data = []
    for o in y_predict:
        data = np.concatenate((data,o),-1)
    return data


input1 = keras.layers.Input(shape=(timestep,1))
o = keras.layers.Conv1D(64,3,activation='relu')(input1)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.Flatten()(o)
o1 = keras.layers.Dense(512,activation='relu')(o)
o2 = keras.layers.Dense(512,activation='relu')(o)
out1 = keras.layers.Dense(timestep,name = 'out1')(o1)
out2 = keras.layers.Dense(timestep,name = 'out2')(o2)
model = keras.Model(input1,[out1,out2])
model.summary()

input1 = keras.layers.Input(shape=(timestep,1))
o = keras.layers.Conv1D(64,3,activation='relu')(input1)
o = keras.layers.AveragePooling1D(3,2)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.AveragePooling1D(3,2)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.Flatten()(o)
o1 = keras.layers.Dense(512,activation='relu')(o)
o2 = keras.layers.Dense(512,activation='relu')(o)
out1 = keras.layers.Dense(timestep,name = 'out1')(o1)
out2 = keras.layers.Dense(timestep,name = 'out2')(o2)
model = keras.Model(input1,[out1,out2])
model.summary()
'''
'''
x = []
y = []
for i in range(1000):
    d = np.random.normal(size=(1,200))
    x.append(d)
    y.append(d)
d = np.random.normal(size=(1,200))
x_test = [d]
x_test = np.reshape(x_test,(len(x_test),200,1))
y_test = [d]
y_test = np.reshape(y_test,(len(x_test),1,200))

x = np.reshape(x,(len(x),200,1))
y = np.reshape(y,(len(y),1,200))
model.compile(loss=tf.losses.mean_squared_error,optimizer=tf.optimizers.Adam(0.01))
model.fit(x,y,64,epochs = 10,validation_split=0.2)
pre = model.predict(x_test)
s = dataConvert(pre)
plt.subplot(2,1,1)
plt.plot(s)
plt.subplot(2,1,2)
plt.plot(y_test[0][0])
plt.show()
model.summary()
'''

mix,s1,s2 = dataDetail(path1+fileNames1[len(fileNames1)-2])
#s1 = getData(path1+fileNames1[len(fileNames1)-3])
filtedData = averFilter(s1,10)


plt.subplot(2,1,1)
plt.plot(s1)
plt.subplot(2,1,2)
plt.plot(filtedData)
plt.show()


'''
bins = np.linspace(np.min(mix),np.max(mix),200)
plt.subplot(2,1,1)
plt.hist(mix,bins)
plt.subplot(2,1,2)
plt.plot(mix)
plt.show()
'''
'''
res1 = filterBybutter(mix,45,65,4000)
res2 = filterBybutter(s1,45,65,4000)
res3 = filterBybutter(s2,45,65,4000)
plt.subplot(3,1,1)
plt.plot(res1)
plt.subplot(3,1,2)
plt.plot(res2)
plt.subplot(3,1,3)
plt.plot(res3)
plt.show()
'''

