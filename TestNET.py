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

timestep = 200
sc = MinMaxScaler(feature_range=(0, 1))
path = "I:\\traindata\\"
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
            mix.append(float(o[3]))
            s1.append(float(o[1]))
            s2.append(float(o[2]))
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

            A = np.reshape(A,(1,len(A)))
            AN = np.reshape(AN,(1,len(AN)))

            m = np.concatenate((A,AN),-1)
         
            xsamples1.append(m) ##合成信号

            ffts1 = np.fft.fft(s1[0,i:i+200])
            As1 = np.abs(ffts1)
            Ans1 = 180*np.angle(ffts1)/np.pi
            As1 = np.reshape(As1,(1,len(As1)))
            Ans1 =np.reshape(Ans1,(1,len(Ans1)))

            ffts2 = np.fft.fft(s2[0,i:i+200])
            As2 = np.abs(ffts2)
            Ans2 = 180*np.angle(ffts2)/np.pi
            As2 = np.reshape(As2,(1,len(As2)))
            Ans2 = np.reshape(Ans2,(1,len(Ans2)))

            d1 = np.concatenate((As1,Ans1),-1)
            d2 = np.concatenate((As2,Ans2),-1)
            ysamples1.append(d1) ##信号1
            ysamples11.append(d2) ##信号2

            m = mix1[0,i:i+200]
            m = np.fft.fft(m)
             #幅度和相位
            A = np.abs(m)
            AN=180*np.angle(m)/np.pi
            A = np.reshape(A,(1,len(A)))
            AN = np.reshape(AN,(1,len(AN)))
            m = np.concatenate((A,AN),-1)
            xsamples2.append(m)

            ffts1 = np.fft.fft(s11[0,i:i+200])
            As1 = np.abs(ffts1)
            Ans1 = 180*np.angle(ffts1)/np.pi
            As1 = np.reshape(As1,(1,len(As1)))
            Ans1 = np.reshape(Ans1,(1,len(Ans1)))

            ffts2 = np.fft.fft(s22[0,i:i+200])
            As2 = np.abs(ffts2)
            Ans2 = 180*np.angle(ffts2)/np.pi
            As2 = np.reshape(As2,(1,len(As2)))
            Ans2 = np.reshape(Ans2,(1,len(Ans2)))

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
            spfft = np.fft.fft(sp)
            spfft = np.reshape(spfft,(len(spfft),1))
            
            A = np.reshape(sp,(len(sp),1))
            x.append(A)

            ##信号1
            ss1 = s1[0,i:i+timestep]
            s1fft = np.fft.fft(ss1)
            s1fft = np.reshape(s1fft,(len(s1fft),1))
           
            A = np.reshape(ss1,(len(ss1),1))
            y1.append(A)

            ##信号2
            ss2 = s2[0,i:i+timestep]
            s2fft = np.fft.fft(ss2)
            s2fft = np.reshape(s2fft,(len(s2fft),1))
            
            A = np.reshape(ss2,(len(ss2),1))
            y2.append(A)
        c = len(x)
        '''
        ss = sc.fit_transform(x[0])
        tx = []
        ty1 = []
        ty2 = []
        for i in range(c):
            tx.append(sc.transform(x[i]))
            ty1.append(sc.transform(y1[i]))
            ty2.append(sc.transform(y2[i]))
        '''
        x_train = np.reshape(x,(c,200,1))
        y_train1 = np.reshape(y1,(c,200,1))
        y_train2 = np.reshape(y2,(c,200,1))
        return x_train,(y_train1,y_train2),res

def getAngle(path):
     with open(path) as f:
        reader = csv.reader(f)
        mix = []
        for o in reader:
            mix.append(float(o[1])) ##混合信号
        mix = np.reshape(mix,(1,len(mix)))
        ANG = []
        for i in range(0,len(mix[0])-timestep,int(timestep/2)):
            m = mix[0,i:i+timestep]
            fft = np.fft.fft(m)
            ang = 180*np.angle(fft)/np.pi
            ang = np.reshape(ang,(1,len(ang)))
            ANG.append(ang)
        return ANG



def dataConvert2(y_predict):
    s1 = y_predict[0]
    s2 = y_predict[1]
    data1 = []
    data2 = []
    #ANG = getAngle(path+fileNames[0])
    for i in range(len(s1)):
        A = s1[i]
        A = np.reshape(A,(1,len(A)))
        #ang = ANG[i]
        #A = sc.inverse_transform(A)
        #x = A*np.exp(1j*ang)
        x = np.fft.ifft(A)
        x = x[0]
        #data1 = np.concatenate((data1,x),-1)
        
        if i == len(s1)-1:
            data1 = np.concatenate((data1,x),-1)
        else:
            data1 = np.concatenate((data1,x[0:int(len(x)/2)]),-1)
        
    for i in range(len(s2)):
        A = s2[i]
        A = np.reshape(A,(1,len(A)))
        #ang = ANG[i]
        #A = sc.inverse_transform(A)
        #x = A*np.exp(1j*ang)
        x = np.fft.ifft(A)
        x = x[0]
        #data2 = np.concatenate((data2,x),-1)
        
        if i == len(s1)-1:
            data2 = np.concatenate((data2,x),-1)
        else:
            data2 = np.concatenate((data2,x[0:int(len(x)/2)]),-1)
        
    return data1,data2


input_img = keras.layers.Input(shape=(200,))
encoded = keras.layers.Dense(1024, activation='relu')(input_img)
encoded = keras.layers.Dropout(0.2)(encoded)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoder_output = keras.layers.Dense(200)(encoded)
encoder_output1 = keras.layers.Dense(200)(encoded)
model = tf.keras.Model(input_img,[encoder_output,encoder_output1])



x_train,y_train,_ = CreateData(path+fileNames[0])
x_train1,y_train1,_ = CreateData(path+fileNames[0])

x = np.concatenate((x_train,x_train1),0)
y1 = np.concatenate((y_train[0],y_train1[0]),0)
y2 = np.concatenate((y_train[1],y_train1[1]),0)
y = (y1,y2)
x_test,y_test,y_data = CreateData(path+fileNames[0])

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

model.fit(x,y,64,epochs = 10,validation_split=0.2,callbacks=[cp_callback])
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


s1,s2 = dataConvert2(rs)




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







