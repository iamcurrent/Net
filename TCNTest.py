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
sc = MinMaxScaler(feature_range=(0, 1))
path = "I:\\电源测试数据\\CSVFile\\"
fileNames = os.listdir(path)
y_data  = []
mixs = []

##数据集构造
def dataDetail(path):
    file1 = path + fileNames[0]
    res = []
    with open(file1) as f:
        reader = csv.reader(f)
        mix = []
        s1 = []
        s2 = []
        for o in reader:
            mix.append(float(o[1]))
            s1.append(float(o[2]))
            s2.append(float(o[3]))
        mix  = mix - np.mean(mix)
        mix = np.reshape(mix,(1,len(mix)))
        s1 = s1 - np.mean(s1)
        s1 = np.reshape(s1,(1,len(s1)))
        s2 = s2 - np.mean(s2)
        s2 = np.reshape(s2,(1,len(s2)))
        #归一化处理
        '''
        mix = sc.fit_transform(mix)
        s1  = sc.transform(s1)
        s2 = sc.transform(s2)
        '''
        mix1 = []
        s11 = []
        s22 = []
        with open(path + fileNames[1]) as f1:
            reader1 = csv.reader(f1)
            for o1 in reader1:
                mix1.append(float(o1[1]))
                s11.append(float(o1[2]))
                s22.append(float(o1[3]))
        mix1 = mix1 - np.mean(mix1)
        mixs.append(mix1)
        mix1 = np.reshape(mix1,(1,len(mix1)))
        s11 = s11 - np.mean(s11)
        s11 = np.reshape(s11,(1,len(s11)))
        s22 = s22 - np.mean(s22)
        s22 = np.reshape(s22,(1,len(s22)))
        '''
        mix1 = sc.transform(mix1)
        s11 = sc.transform(s11)
        s22 = sc.transform(s22)
        '''

        xsamples1 = []
        ysamples1 = []
        ysamples11 = []

        xsamples2 = []
        ysamples2 = []
        ysamples22 = []
        
        y_data.append(s11)
        y_data.append(s22)
        for i in range(0,len(mix[0])-100,100):
            m = mix[0,i:i+200]
            m = np.fft.fft(m)
            #幅度和相位

            A = np.abs(m)
            AN=180*np.angle(m)/np.pi

            A = sc.fit_transform(np.reshape(A,(1,len(A))))
            AN = sc.transform(np.reshape(AN,(1,len(AN))))

            m = np.concatenate((A,AN),-1)
         
            xsamples1.append(m) ##合成信号

            ffts1 = np.fft.fft(s1[0,i:i+200])
            As1 = np.abs(ffts1)
            Ans1 = 180*np.angle(ffts1)/np.pi
            As1 = sc.transform(np.reshape(As1,(1,len(As1))))
            Ans1 = sc.transform(np.reshape(Ans1,(1,len(Ans1))))

            ffts2 = np.fft.fft(s2[0,i:i+200])
            As2 = np.abs(ffts2)
            Ans2 = 180*np.angle(ffts2)/np.pi
            As2 = sc.transform(np.reshape(As2,(1,len(As2))))
            Ans2 = sc.transform(np.reshape(Ans2,(1,len(Ans2))))

            d1 = np.concatenate((As1,Ans1),-1)
            d2 = np.concatenate((As2,Ans2),-1)
            ysamples1.append(d1) ##信号1
            ysamples11.append(d2) ##信号2

            m = mix1[0,i:i+200]
            m = np.fft.fft(m)
             #幅度和相位
            A = np.abs(m)
            AN=180*np.angle(m)/np.pi
            A = sc.fit_transform(np.reshape(A,(1,len(A))))
            AN = sc.transform(np.reshape(AN,(1,len(AN))))
            m = np.concatenate((A,AN),-1)
            xsamples2.append(m)

            ffts1 = np.fft.fft(s11[0,i:i+200])
            As1 = np.abs(ffts1)
            Ans1 = 180*np.angle(ffts1)/np.pi
            As1 = sc.transform(np.reshape(As1,(1,len(As1))))
            Ans1 = sc.transform(np.reshape(Ans1,(1,len(Ans1))))

            ffts2 = np.fft.fft(s22[0,i:i+200])
            As2 = np.abs(ffts2)
            Ans2 = 180*np.angle(ffts2)/np.pi
            As2 = sc.transform(np.reshape(As2,(1,len(As2))))
            Ans2 = sc.transform(np.reshape(Ans2,(1,len(Ans2))))

            d1 = np.concatenate((As1,Ans1),-1)
            d2 = np.concatenate((As2,Ans2),-1)

            ysamples2.append(d1)
            ysamples22.append(d2)
        c = len(xsamples1)
        x_train = np.reshape(xsamples1,(c,400,1))
        y_train1 = np.reshape(ysamples1,(c,400,1))
        y_train2 = np.reshape(ysamples11,(c,400,1))

        x_test = np.reshape(xsamples2,(c,400,1))
        y_test1 = np.reshape(ysamples2,(c,400,1))
        y_test2 = np.reshape(ysamples22,(c,400,1))
    return x_train,(y_train1,y_train2),x_test,(y_test1,y_test2)


def dataConvert1(y_predict):
    s1 = y_predict[0]
    s2 = y_predict[1]
    data1 = []
    data2 = []
    for i in range(len(s1)):
        
        A = s1[i][0:int(len(s1[i])/2)]
        ang = s1[i][int(len(s1[i])/2):]
        x = A*np.exp(1j*ang)
        if i == len(s1)-1:
            data1 = np.concatenate((data1,x),-1)
        else:
            data1 = np.concatenate((data1,x[0:int(len(x)/2)]),-1)
        
    for i in range(len(s2)):
        A = s2[i][0:int(len(s1[i])/2)]
        ang = s2[i][int(len(s1[i])/2):]
        x = A*np.exp(1j*ang)
        if i == len(s2)-1:
            data2 = np.concatenate((data2,x),-1)
        else:
            data2 = np.concatenate((data2,x[0:int(len(x)/2)]),-1)
    data1 = np.reshape(data1,(1,len(data1)))
    data2 = np.reshape(data2,(1,len(data2)))
    return data1,data2

input_img = keras.layers.Input(shape=(400*1,))
encoded = keras.layers.Dense(1024, activation='relu')(input_img)
#encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoder_output = keras.layers.Dense(400)(encoded)
encoder_output1 = keras.layers.Dense(400)(encoded)
model = tf.keras.Model(input_img,[encoder_output,encoder_output1])

x_train,y_train,x_test,y_test  = dataDetail(path)

#x_test,y_test = getTestData(path+fileNames[len(fileNames)-1])

model.compile(loss=tf.losses.mean_squared_error,optimizer=tf.optimizers.Adam(0.01))

checkpoint_save_path = "./checkpoint/LSTM_stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

model.fit(x_train,y_train,epochs = 10,validation_split=0.2,callbacks=[cp_callback])
model.summary()
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

rs = model.predict(x_test)
dataConvert1(rs)

r1 = y_data[0]
r2 = y_data[1]


s1,s2 = dataConvert1(rs)
s1 = sc.inverse_transform(s1)
s2 = sc.inverse_transform(s2)



plt.subplot(5,1,1)
plt.plot(mixs[0])
plt.subplot(5,1,2)
plt.plot(r1[0])
plt.subplot(5,1,3)
plt.plot(s1[0])
plt.subplot(5,1,4)
plt.plot(r2[0])
plt.subplot(5,1,5)
plt.plot(s2[0])
plt.show()







