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
from tensorflow.python.ops.gen_math_ops import angle

timestep = 64
sc = MinMaxScaler(feature_range=(0, 1))

def getMix1():
    res = []
    t = np.linspace(0,10,640)
    s1 = np.sin(2*np.pi*450*t)
    s2 = np.cos(2*np.pi*400*t)
    noise = np.tan(2*np.pi*400*t)
    mix = 0.2*s1+0.8*s2
    res.append(mix)
    res.append(s1)
    res.append(s2)
    mix = np.reshape(mix,(1,len(mix)))
    s1 = np.reshape(s1,(1,len(s1)))
    s2 = np.reshape(s2,(1,len(s2)))
    x = []
    y1 = []
    y2 = []

    scope = []
    angle = []

    sco1 = []
    sco2 = []

    ang1 = []
    ang2 = []

    ##构造数据集
    for i in range(0,len(mix[0])-timestep,int(timestep/2)):
        sp = mix[0,i:i+timestep]
        spfft = np.fft.fft(sp)
        A = np.abs(spfft)
        ang = 180*np.angle(spfft)/np.pi
        scope.append(np.reshape(A,(1,len(A))))
        angle.append(np.reshape(ang,(1,len(ang))))

        ss1 = s1[0,i:i+timestep]
        s1fft = np.fft.fft(ss1)
        A = np.abs(s1fft)
        ang = 180*np.angle(s1fft)/np.pi
        sco1.append(np.reshape(A,(1,len(A))))
        ang1.append(np.reshape(ang,(1,len(ang))))


        ss2 = s2[0,i:i+timestep]
        s2fft = np.fft.fft(ss2)
        A = np.abs(s2fft)
        ang = 180*np.angle(s2fft)/np.pi
        
        sco2.append(np.reshape(A,(1,len(A))))
        ang2.append(np.reshape(ang,(1,len(ang))))
        
    c  = len(scope)
    s = sc.fit_transform(scope[0])
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []
    d6 = []
    for i in range(c):
        d1.append(sc.transform(scope[i]))
        d2.append(sc.transform(angle[i]))
        d3.append(sc.transform(sco1[i]))
        d4.append(sc.transform(sco2[i]))
        d5.append(sc.transform(ang1[i]))
        d6.append(sc.transform(ang2[i]))
    x_train1 = np.reshape(d1,(c,timestep,1))
    x_train2 = np.reshape(d2,(c,timestep,1))
    y_train1 = np.reshape(d3,(c,timestep,1))
    y_train2 = np.reshape(d4,(c,timestep,1))
    y_train3 = np.reshape(d5,(c,timestep,1))
    y_train4 = np.reshape(d6,(c,timestep,1))
    return (x_train1,x_train2),(y_train1,y_train2,y_train3,y_train4),res

def dataConvert(predict):
    a1 = predict[0]
    a2 = predict[1]
    ang1 = predict[2]
    ang2 = predict[3]
    c = len(a1)
    data1 = []
    data2 = []
    for i in range(c):
        ta = np.reshape(a1[i],(1,len(a1[i])))
        a = sc.inverse_transform(ta)
        tang = np.reshape(ang1[i],(1,len(ang1[i])))
        ang = sc.inverse_transform(tang)
        x = a*np.exp(1j*ang)
        x = np.fft.ifft(x)
        x = x[0]
        if i == c-1:
            data1 = np.concatenate((data1,x),-1)
        else:
            data1 = np.concatenate((data1,x[0:int(len(x)/2)]),-1)
        
        ta = np.reshape(a2[i],(1,len(a2[i])))
        a = sc.inverse_transform(ta)
        tang = np.reshape(ang2[i],(1,len(ang2[i])))
        ang = sc.inverse_transform(tang)
        x = a*np.exp(1j*ang)
        x = np.fft.ifft(x)
        x = x[0]
        if i == c-1:
            data2 = np.concatenate((data2,x),-1)
        else:
            data2 = np.concatenate((data2,x[0:int(len(x)/2)]),-1)
    return data1,data2





input1 = tf.keras.Input(shape=(timestep,))
input2 = tf.keras.Input(shape=(timestep,))
##训练幅值
a = tf.keras.layers.Dense(1024,activation='relu')(input1)
a = tf.keras.layers.Dropout(0.2)(a)
a = tf.keras.layers.Dense(1024,activation='relu')(a)
out1 = tf.keras.layers.Dense(timestep)(a)
out11 = tf.keras.layers.Dense(timestep)(a)
##训练相位
ang = tf.keras.layers.Dense(1024,activation='relu')(input2)
ang = tf.keras.layers.Dropout(0.2)(ang)
ang = tf.keras.layers.Dense(1024,activation='relu')(ang)
out2 = tf.keras.layers.Dense(timestep)(ang)
out22 = tf.keras.layers.Dense(timestep)(ang)

model = keras.Model([input1,input2],[out1,out11,out2,out22])

x_train,y_train,_= getMix1()
x_test,y_test,y_data = getMix1()

model.compile(loss=tf.losses.mean_squared_error,optimizer=tf.optimizers.Adam(0.001))

checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

model.fit(x_train,y_train,64,epochs = 10000,validation_split=0.2,callbacks=[cp_callback])
model.summary()
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

predict  = model.predict(x_test)
d1,d2 = dataConvert(predict)

mixs = y_data[0]
s1 = y_data[1]
s2 = y_data[2]

plt.subplot(5,1,1)
plt.plot(mixs)
plt.subplot(5,1,2)
plt.plot(s1)
plt.subplot(5,1,3)
plt.plot(d1)
plt.subplot(5,1,4)
plt.plot(s2)
plt.subplot(5,1,5)
plt.plot(d2)


plt.show()