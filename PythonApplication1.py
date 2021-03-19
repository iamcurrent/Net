import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import mean
from numpy.lib.arraysetops import setxor1d
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from numpy.core.function_base import linspace
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dropout, Dense, LSTM
import tensorflow as tf
import csv
import os
from sklearn.preprocessing import MinMaxScaler
import pywt
from scipy import signal

sc = MinMaxScaler(feature_range=(0, 1))
timestep = 200
path = "I:\\电源测试数据\\CSVFile\\"
path1 = "I:\\电源测试数据\\CSVTest\\"
fileNames = os.listdir(path)
fileNames1 = os.listdir(path1)
y_data  = []
mixs = []

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
        mix  = mix - np.mean(mix)
        s1 = s1 - np.mean(s1)
        s2 = s2 - np.mean(s2)
        #mix = filterBybutter(mix,50,200,4000)
        #s1 = filterBybutter(s1,50,200,4000)
        #s2 = filterBybutter(s2,50,200,4000)
        mix = averFilter(mix,20)
        s1 = averFilter(s1,20)
        s2 = averFilter(s2,20)

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

def createTrainData(path,ratio):
    files = os.listdir(path)
    mixsignal = []
    signal1 = []
    signal2 = []
    for o in files:
        filePath = path+o
        with open(filePath) as f:
            reader = csv.reader(f)
            mix = []
            s1 = []
            s2 = []
            for d in reader:
                mix.append(float(d[1]))
                s1.append(float(d[2]))
                s2.append(float(d[3]))
            mix = mix[0:int(ratio*len(mix))]
            s1 = s1[0:int(ratio*len(s1))]
            s2 = s2[0:int(ratio*len(s2))]
            mix = averFilter(mix,20)
            s1 = averFilter(s1,20)
            s2 = averFilter(s2,20)
            mix = np.reshape(mix,(1,len(mix)))
            s1 = np.reshape(s1,(1,len(s1)))
            s2 = np.reshape(s2,(1,len(s2)))
            for i in range(0,len(mix[0])-int(timestep/2),int(timestep/2)):
                mixsignal.append(np.reshape(mix[0,i:i+timestep],(1,timestep)))
                signal1.append(np.reshape(s1[0,i:i+timestep],(1,timestep)))
                signal2.append(np.reshape(s2[0,i:i+timestep],(1,timestep)))
    c = len(mixsignal)
    mixsignal = np.reshape(mixsignal,(c,timestep,1))
    signal1 = np.reshape(signal1,(c,timestep,1))
    signal2 = np.reshape(signal2,(c,timestep,1))
    return mixsignal,(signal1,signal2)
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

##小波降噪
def denoise(data):
    coeffs = pywt.wavedec(data, 'sym8', level=8) 
    #coeffs=np.array(coeffs)# 将信号进行小波分解
    ca1=coeffs[0]
    cd1=coeffs[8]
    cd2=coeffs[7]
    cd3=coeffs[6]
    cd4=coeffs[5]
    cd5=coeffs[4]
    cd6=coeffs[3]
    cd7=coeffs[2]
    cd8=coeffs[1]

    # Td1=thselect(cd1,'heursure');  %%heursure阈值
    Td1=3.7188
    Td2=3.5317
    Td3=0.1206
    Td4=0.2542
    Td5=0.6816
    Td6=0.0099
    Td7=1.9913
    Td8=1.0525

    sa1=np.zeros(len(ca1))
    sd4=pywt.threshold(cd4,Td4,'soft')
    sd5=pywt.threshold(cd5,Td5,'soft')
    sd6=pywt.threshold(cd6,Td6,'soft')
    sd7=pywt.threshold(cd7,Td7,'soft')
    sd8=pywt.threshold(cd8,Td8,'soft')
    sd1=np.zeros(len(cd1));sd2=np.zeros(len(cd2));sd3=np.zeros(len(cd3))

    c2=[sa1,sd8,sd7,sd6,sd5,sd4,sd3,sd2,sd1]
    s0=pywt.waverec(c2, 'sym8')
    return s0

def dataConvert1(y_predict):
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
            data1 = np.concatenate((data1,np.round(a,6)),-1)
        else:
            data1 = np.concatenate((data1,np.round(a[0:int(len(a)/2)],6)),-1)

        a = s2[i]
        a = np.reshape(a,(1,len(a)))
        #a = sc.inverse_transform(a)
        a = a[0]
        if c-1==i:
            data2 = np.concatenate((data2,a),-1)
        else:
            data2 = np.concatenate((data2,a[0:int(len(a)/2)]),-1)
    data1 = np.reshape(data1,(1,len(data1)))
    data2 = np.reshape(data2,(1,len(data2)))
    #data1 = sc.inverse_transform(data1)
    #data2 = sc.inverse_transform(data2)
    
    return data1[0],data2[0]

'''
input_img = keras.layers.Input(shape=(timestep,))
encoded = keras.layers.Dense(1024, activation='relu')(input_img)
encoded = keras.layers.Dropout(0.2)(encoded)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoded = keras.layers.Dropout(0.2)(encoded)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
en1 = keras.layers.Dense(512,activation='relu')(encoded)
en2 = keras.layers.Dense(512,activation='relu')(encoded)
encoder_output = keras.layers.Dense(timestep)(en1)
encoder_output1 = keras.layers.Dense(timestep)(en2)
model = tf.keras.Model(input_img,[encoder_output,encoder_output1])
'''

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

'''
input1 = keras.layers.Input(shape=(timestep,1))
o  = keras.layers.LSTM(80,return_sequences=True)(input1)
o = keras.layers.Dropout(0.2)(o)
o = keras.layers.LSTM(100)(o)
o = keras.layers.Dropout(0.2)(o)
o1 = keras.layers.Dense(1024,activation='relu')(o)
o2 = keras.layers.Dense(1024,activation='relu')(o)
out1 = keras.layers.Dense(timestep)(o1)
out2 = keras.layers.Dense(timestep)(o2)
model = keras.Model(input1,[out1,out2])
'''
x_train,y_train = createTrainData(path,0.6)
x_test,y_test,y_data  = dataDetail(path+fileNames[0])



model.compile(loss={'out1':tf.keras.losses.mean_squared_error,'out2':tf.keras.losses.mean_squared_error},optimizer=tf.optimizers.Adam(0.001),loss_weights=[0.5,0.5])

checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

#model.fit(x_train,{'out1':y_train[0],'out2':y_train[1]},64,epochs = 10,validation_split=0.2,callbacks=[cp_callback])
'''
for i in range(len(model.layers)):
    for j in range(i):
        model.layers[j].trainable = False
    model.fit(x_train,y_train,epochs = 200,validation_split=0.2,callbacks=[cp_callback])
'''

model.summary()
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

rs = model.predict(x_test)
s1,s2 = dataConvert1(rs)


mixs = y_data[0]
r1 = y_data[1]
r2 = y_data[2]

sub1 = s1 - r1
sub2 = s2 - r2

loss1 = np.sqrt(np.dot(sub1,sub1)/len(sub1))
loss2 = np.sqrt(np.dot(sub2,sub2)/len(sub2))
print("信号1的loss",loss1,"信号2的loss",loss2)



plt.subplot(5,1,1)
plt.plot(mixs)
plt.subplot(5,1,2)
plt.plot(r1)
plt.subplot(5,1,3)
plt.plot(r2)
plt.subplot(5,1,4)
plt.plot(s1)
plt.subplot(5,1,5)
plt.plot(s2)
plt.show()








