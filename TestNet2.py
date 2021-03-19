import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import mean
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from numpy.core.function_base import linspace
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import csv
import os
timestep = 200
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
path = "I:\\traindata\\"
fileNames = os.listdir(path)
y_data  = []
mixs = []
def getData(start,stop,num):
    res = []
    t = np.linspace(start,stop,num)
    s1 = np.sin(2*np.pi*450*t)
    s2 = np.cos(2*np.pi*400*t)
    noise = np.random.normal(size=(1,len(s1)))
    mix = 0.2*s1+0.8*s2+noise[0]
    mix = filterBybutter(mix,400,450,1000)
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

def CreateData(path):
    res = []
    with open(path) as f:
        reader = csv.reader(f)
        mix = []
        s1 = []
        s2 = []
        for o in reader:
            mix.append(float(o[3])) ##混合信号
            s1.append(float(o[1])) ##信号1
            s2.append(float(o[2])) ##信号2
        mix = mix - np.mean(mix)
        s1 = s1 - np.mean(s1)
        s2 = s2 - np.mean(s2)
        res.append(mix)
        res.append(s1)
        res.append(s2)
        mix = np.reshape(mix,(1,len(mix)))
        s1 = np.reshape(s1,(1,len(s1)))
        s2 = np.reshape(s2,(1,len(s2)))

        x = []
        y1 = []
        y2 = []
        ##构造数据集
        for i in range(0,len(mix[0])-timestep,int(timestep/2)):
            ##混合信号
            sp = mix[0,i:i+timestep]
            A = np.reshape(sp,(len(sp),1))
            x.append(A)

            ##信号1
            ss1 = s1[0,i:i+timestep]
            A = np.reshape(ss1,(len(ss1),1))
            y1.append(A)

            ##信号2
            ss2 = s2[0,i:i+timestep]
            A = np.reshape(ss2,(len(ss2),1))
            y2.append(A)
        c = len(x)
   
        x_train = np.reshape(x,(c,200,1))
        y_train1 = np.reshape(y1,(c,200,1))
        y_train2 = np.reshape(y2,(c,200,1))
        return x_train,(y_train1,y_train2),res



##使用卷积实现移动平均
def averFilter(data,n):
    return np.convolve(data, np.ones((n,))/n, mode='same')

def filterBybutter(data,f1,f2,fs):
    f1 = 2*f1/fs
    f2 = 2*f2/fs
    b, a = signal.butter(2, [f1,f2], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
    x = signal.filtfilt(b, a, data)  #data为要过滤的信号
    return x

def dataConvert(y_predict):
    s1 = y_predict[0]
    s2 = y_predict[1]
    data1 = []
    data2 = []
    c = len(s1)
    for i in range(c):
        a = s1[i]
        a = np.reshape(a,(1,len(a)))
        #a = sc.inverse_transform(a)
        a = a[0]
        if c-1==i:
            data1 = np.concatenate((data1,a),-1)
        else:
            data1 = np.concatenate((data1,a[0:int(len(a)/2)]),-1)


        a = s2[i]
        a = np.reshape(a,(1,len(a)))
        #a = sc.inverse_transform(a)
        a = a[0]
        if c-1==i:
            data2 = np.concatenate((data2,a),-1)
        else:
            data2 = np.concatenate((data2,a[0:int(len(a)/2)]),-1)
    return data1,data2

x_train,y_train,_ = CreateData(path+fileNames[0])

x_test,y_test,y_data = CreateData(path+fileNames[0])


input1 = keras.layers.Input(shape=(timestep,1))
o = keras.layers.Conv1D(64,3,activation='relu')(input1)
o = keras.layers.BatchNormalization()(o)
o = keras.layers.AveragePooling1D(3,2)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.BatchNormalization()(o)
o = keras.layers.AveragePooling1D(3,2)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.Flatten()(o)
o1 = keras.layers.Dense(512,activation='relu')(o)
o2 = keras.layers.Dense(512,activation='relu')(o)
out1 = keras.layers.Dense(timestep,name = 'out1')(o1)
out2 = keras.layers.Dense(timestep,name = 'out2')(o2)
model = keras.Model(input1,[out1,out2])

model.compile(loss=tf.losses.mean_squared_error,optimizer=tf.optimizers.Adam(0.01))

checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

model.fit(x_train,y_train,epochs = 100,validation_split=0.2,callbacks=[cp_callback])



file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

y_predict = model.predict(x_test)


s1,s2 = dataConvert(y_predict)

mixs = y_data[0]
r1 = y_data[1]
r2 = y_data[2]


plt.subplot(5,1,1)
plt.plot(mixs)
plt.subplot(5,1,2)
plt.plot(r1)
plt.subplot(5,1,3)
plt.plot(s1)
plt.subplot(5,1,4)
plt.plot(r2)
plt.subplot(5,1,5)
plt.plot(s2)
plt.show()

