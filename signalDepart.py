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
timestep = 200
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
path = "I:\\traindata\\"
fileNames = os.listdir(path)
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
            mix.append(float(o[3]))
            s1.append(float(o[1]))
            s2.append(float(o[2]))
        mix  = mix - np.mean(mix)
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
        for i in range(0,len(mix[0])-timestep,int(timestep/2)):
            x.append(np.reshape(mix[0,i:i+timestep],(1,timestep)))
            y1.append(np.reshape(s1[0,i:i+timestep],(1,timestep)))
            y2.append(np.reshape(s2[0,i:i+timestep],(1,timestep)))
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



def dataConvert1(y_predict):
    s1 = y_predict[0]
    s2 = y_predict[1]
    data1 = []
    data2 = []
    c = len(s1)
    for i in range(c):
        a = s1[i]
        plt.plot(a)
        plt.show()
        a = np.reshape(a,(1,len(a)))
        a = sc.inverse_transform(a)
        a = a[0]
        if c-1==i:
            data1 = np.concatenate((data1,a),-1)
        else:
            data1 = np.concatenate((data1,a[0:int(len(a)/2)]),-1)

        a = s2[i]
        a = np.reshape(a,(1,len(a)))
        a = sc.inverse_transform(a)
        a = a[0]
        if c-1==i:
            data2 = np.concatenate((data2,a),-1)
        else:
            data2 = np.concatenate((data2,a[0:int(len(a)/2)]),-1)
    return data1,data2

'''
model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(400)
])
'''
'''
input_img = keras.layers.Input(shape=(200*1,))
encoded = keras.layers.Dense(1024, activation='relu')(input_img)
encoded = keras.layers.Dropout(0.2)(encoded)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoded = keras.layers.Dropout(0.2)(encoded)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoder_output = keras.layers.Dense(200)(encoded)
encoder_output1 = keras.layers.Dense(200)(encoded)
model = tf.keras.Model(input_img,[encoder_output,encoder_output1])
'''
input1 = keras.layers.Input(shape=(200,1))
o = keras.layers.Conv1D(32,3,activation='relu')(input1)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.MaxPool1D(3,3)(o)
o = keras.layers.Conv1D(32,3,activation='relu')(o)
o = keras.layers.MaxPool1D(3,3)(o)
out1 = keras.layers.Dense(200)(o)
out2 = keras.layers.Dense(200)(o)
model = keras.Model(input1,[out1,out2])

x_train,y_train,_ = dataDetail(path+fileNames[0])
x_test,y_test,y_data  = dataDetail(path+fileNames[1])

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

for i  in range(len(model.layers)-2):
    for j in range(i):
        model.layers[j].trainable = False
    model.fit(x_train,y_train,epochs = 10,validation_split=0.2,callbacks=[cp_callback])
model.summary()


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
s1,s2 = dataConvert1(rs)




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







