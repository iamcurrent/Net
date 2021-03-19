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
import tensorflow as tf
import csv
import os
from numpy.fft import fft
from sklearn.preprocessing import MinMaxScaler

timestep = 64
sc = MinMaxScaler(feature_range=(0, 1))


def getMix():
    res = []
    t = np.linspace(0,10,640)
    s1 = np.sin(2*np.pi*450*t)
    s2 = np.cos(2*np.pi*400*t)
    mix = s1+s2
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
        sp = mix[0,i:i+timestep]
        spfft = np.fft.fft(sp)
        A = np.abs(spfft)
        ang = 180*np.angle(spfft)/np.pi
 
        spfft = np.reshape(spfft,(len(spfft),1))
        
        #A = sc.fit_transform(A)
        x.append(A)
        ss1 = s1[0,i:i+timestep]
        s1fft = np.fft.fft(ss1)
        A = np.abs(s1fft)
        ang = 180*np.angle(s1fft)/np.pi

        s1fft = np.reshape(s1fft,(len(s1fft),1))
        
        # A = sc.transform(A)

        y1.append(A)
        ss2 = s2[0,i:i+timestep]
        s2fft = np.fft.fft(ss2)
        A = np.abs(s2fft)
        ang = 180*np.angle(s2fft)/np.pi
        s2fft = np.reshape(s2fft,(len(s2fft),1))
        
        #A = sc.transform(A)
        y2.append(A)
    c = len(x)
    ss = sc.fit_transform(x[0])
    tx = []
    ty1 = []
    ty2 = []
    for i in range(c):
        tx.append(sc.transform(x[i]))
        ty1.append(sc.transform(y1[i]))
        ty2.append(sc.transform(y2[i]))
    x_train = np.reshape(tx,(c,timestep,1))
    y_train1 = np.reshape(ty1,(c,timestep,1))
    y_train2 = np.reshape(ty2,(c,timestep,1))
    return x_train,(y_train1,y_train2),res


def getMix1():
    res = []
    t = np.linspace(0,10,640)
    s1 = np.sin(2*np.pi*450*t)
    s2 = np.cos(2*np.pi*400*t)
    mix = s1+s2
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
        sp = mix[0,i:i+timestep]
        spfft = np.fft.fft(sp)
        A = np.abs(spfft)
        ang = 180*np.angle(spfft)/np.pi
        t = np.concatenate((A,ang),-1)
        t = np.reshape(t,(len(t),1))
        
        #A = sc.fit_transform(A)
        x.append(t)
        ss1 = s1[0,i:i+timestep]
        s1fft = np.fft.fft(ss1)
        A = np.abs(s1fft)
        ang = 180*np.angle(s1fft)/np.pi
        t = np.concatenate((A,ang),-1)
        t = np.reshape(t,(len(t),1))
        
        # A = sc.transform(A)

        y1.append(t)
        ss2 = s2[0,i:i+timestep]
        s2fft = np.fft.fft(ss2)
        A = np.abs(s2fft)
        ang = 180*np.angle(s2fft)/np.pi
        t = np.concatenate((A,ang),-1)
        t = np.reshape(t,(len(t),1))
        
        #A = sc.transform(A)
        y2.append(t)
    c = len(x)
    ss = sc.fit_transform(x[0])
    tx = []
    ty1 = []
    ty2 = []
    for i in range(c):
        tx.append(sc.transform(x[i]))
        ty1.append(sc.transform(y1[i]))
        ty2.append(sc.transform(y2[i]))
    x_train = np.reshape(tx,(c,timestep*2,1))
    y_train1 = np.reshape(ty1,(c,timestep*2,1))
    y_train2 = np.reshape(ty2,(c,timestep*2,1))
    return x_train,(y_train1,y_train2),res

##获取相位
def getAngle(mix):
    mix = np.reshape(mix,(1,len(mix)))
    ANG = []
    for i in range(0,len(mix[0])-timestep,int(timestep/2)):
        m = mix[0,i:i+timestep]
        fft = np.fft.fft(m)
        ang = 180*np.angle(fft)/np.pi
        ang = np.reshape(ang,(1,len(ang)))
        ANG.append(ang)
    return ANG


def dataConvert2(y_predict,mix):
    s1 = y_predict[0]
    s2 = y_predict[1]
    data1 = []
    data2 = []
    ANG = getAngle(mix)
    for i in range(len(s1)):
        A = s1[i]
        A = np.reshape(A,(1,len(A)))
        ang = ANG[i]
        A = sc.inverse_transform(A)
        x = A*np.exp(1j*ang)
        x = np.fft.ifft(x)
        x = x[0]
        #data1 = np.concatenate((data1,x),-1)
        
        if i == len(s1)-1:
            data1 = np.concatenate((data1,x),-1)
        else:
            data1 = np.concatenate((data1,x[0:int(len(x)/2)]),-1)
        
    for i in range(len(s2)):
        A = s2[i]
        A = np.reshape(A,(1,len(A)))
        ang = ANG[i]
        A = sc.inverse_transform(A)
        x = A*np.exp(1j*ang)
        x = np.fft.ifft(x)
        x = x[0]
        #data2 = np.concatenate((data2,x),-1)
        
        if i == len(s1)-1:
            data2 = np.concatenate((data2,x),-1)
        else:
            data2 = np.concatenate((data2,x[0:int(len(x)/2)]),-1)
        
    return data1,data2

def dataConvert3(y_predict,mix):
    s1 = y_predict[0]
    s2 = y_predict[1]
    data1 = []
    data2 = []
    ANG = getAngle(mix)
    for i in range(len(s1)):
        d = np.reshape(s1[i],(1,len(s1[i])))
        d = sc.inverse_transform(d)
        d = d[0]
        A = d[0:int(len(d)/2)]
        ang = d[int(len(d)/2):]
        x = A*np.exp(1j*ang)
        x = np.fft.ifft(x)
        
        #data1 = np.concatenate((data1,x),-1)
        
        if i == len(s1)-1:
            data1 = np.concatenate((data1,x),-1)
        else:
            data1 = np.concatenate((data1,x[0:int(len(x)/2)]),-1)
        
    for i in range(len(s2)):
        d = np.reshape(s2[i],(1,len(s2[i])))
        d = sc.inverse_transform(d)
        d = d[0]
        A = d[0:int(len(d)/2)]
        ang = d[int(len(d)/2):]
        x = A*np.exp(1j*ang)
        x = np.fft.ifft(x)
        #x = A*np.exp(1j*ang)
        #x = np.fft.ifft(x)
        #x = x[0]
        #data2 = np.concatenate((data2,x),-1)
        
        if i == len(s1)-1:
            data2 = np.concatenate((data2,x),-1)
        else:
            data2 = np.concatenate((data2,x[0:int(len(x)/2)]),-1)
        
    return data1,data2


input1 = keras.layers.Input(shape=(timestep*2,))
encoded = keras.layers.Dense(256, activation='relu')(input1)
#encoded = keras.layers.Dropout(0.2)(encoded)
#encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoder_output = keras.layers.Dense(timestep*2,name = 'ot1')(encoded)
encoder_output1 = keras.layers.Dense(timestep*2,name = 'ot2')(encoded)
model = tf.keras.Model(input1,[encoder_output,encoder_output1])





x_train,y_train,_ = getMix1()
x_test,y_test,y_data = getMix1()


#x_test,y_test = getTestData(path+fileNames[len(fileNames)-1])

model.compile(loss=tf.losses.mean_squared_error,optimizer=tf.optimizers.Adam(0.001))

checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

model.fit(x_train,y_train,128,epochs = 100,validation_split=0.2,callbacks=[cp_callback])
model.summary()
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

rs = model.predict(x_test)


mixs = y_data[0]
r1 = y_data[1]
r2 = y_data[2]


s1,s2 = dataConvert3(rs,mixs)




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