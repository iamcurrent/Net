from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import mean, shape
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
import mmap
import contextlib
import websocket
from sklearn import svm
from tensorflow.python.keras.activations import get
from sklearn.preprocessing import OneHotEncoder

from scipy import signal
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
import datahandleer
timestep = 10000
path = "I:\\频谱数据\\"
num_classes = 2
def getLstmModel():
    input1 = keras.Input(shape=(timestep,1))
    o = keras.layers.LSTM(32,return_sequences=True)(input1)
    o = keras.layers.LSTM(32,return_sequences=True)(o)
    #o = keras.layers.LSTM(256,return_sequences=True)(o)
    o = keras.layers.LSTM(32)(o)
    out = keras.layers.Dense(1,"softmax")(o)
    model = keras.Model(input1,out)
    return model

def getTcnModel():
    input1 = keras.Input(shape=(timestep,1))
    o = TCN(256)(input1)
    out = keras.layers.Dense(1,activation="sigmoid")(o)
    model = keras.Model(input1,out)
    return model

def getCovModel():
    input1 = keras.layers.Input(shape=(timestep,1))
    o = keras.layers.Conv1D(128,3,activation='relu')(input1)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.AveragePooling1D(3)(o)
    o = keras.layers.Conv1D(64,3,activation='relu')(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.MaxPool1D(3)(o)
    o = keras.layers.Conv1D(32,3,activation='relu')(o)
    o = keras.layers.Flatten()(o)
    out = keras.layers.Dense(num_classes,activation='softmax')(o)
    model = keras.Model(input1,out)
    return model

def getDenseModel():
    input1 = keras.layers.Input(shape=(timestep,1))
    o = keras.layers.Dense(512,activation="relu")(input1)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Dropout(0.2)(o)
    o = keras.layers.Dense(256,activation="relu")(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Dropout(0.2)(o)
    '''
    o = keras.layers.Dense(500,activation="relu")(o)
    o = keras.layers.BatchNormalization()(o)
    o = keras.layers.Dropout(0.2)(o)
    '''
    out = keras.layers.Dense(1,"sigmoid")(o)
    model = keras.Model(input1,out)
    return model

def getSvm():
    clf = svm.SVC(probability=True)
    return clf


'''
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
data = datahandleer.handleSimple(path,fileNames)
data1,yz = datahandleer.handleMix(path,fileNames3,yz)
dataTest,yTest = datahandleer.handleMix(path,fileNames4,yz)
train_data = data + data1
trainy = np.concatenate((ys,yz),0)
train_data = np.reshape(train_data,(len(train_data),10000))
trainy = np.reshape(trainy,(len(trainy),1))
dataTest = np.reshape(dataTest,(len(dataTest),10000))
yTest = np.reshape(yTest,(len(yTest),1))
'''
'''
paths = "./data.csv"
data = []
y = []
with open(paths) as f:
    reader = csv.reader(f)
    for o in reader:
        y.append(np.int(o[len(o)-1]))
        data.append(np.float32(o[:10000]))
train_data = data
train_y = y

test_data = data[int(len(data)/2):]
test_y = y[int(len(data)/2):]

train_data = np.reshape(train_data,(len(train_data),10000,1))
test_data = np.reshape(test_data,(len(test_data),10000,1))

train_y = np.reshape(train_y,(len(train_y),1))
test_y = np.reshape(test_y,(len(test_y),1))

fileNames4 = os.listdir(os.path.join(path,'ztest'))
fileNames4 = ["ztest/"+o for o in fileNames4]
ztestx,ztesty = datahandleer.handleMix(path,fileNames4,[2,3])
ztestx = np.reshape(ztestx,(len(ztestx),10000,1))
ztesty = np.reshape(ztesty,(len(ztesty),1))
onehot_encoder = OneHotEncoder(sparse=False)
trainy = onehot_encoder.fit_transform(train_y)
yTest = onehot_encoder.transform(test_y)
ztesty = onehot_encoder.transform(ztesty)
'''
#trainy = tf.one_hot(trainy,depth=4)
#yTest = tf.one_hot(yTest,depth=4)


model = getCovModel()
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(0.001))
checkpoint_save_path = "./checkpoint/sparate.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

#model.fit(train_data,trainy,epochs = 1000,validation_split=0.2,callbacks=[cp_callback])
'''
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
'''
with contextlib.closing(mmap.mmap(-1, 100, tagname="testMmf", access=mmap.ACCESS_WRITE)) as m:
    while True:
        try:
            da1 = datahandleer.getDataSimple("D:\\shared\\temp.csv")
            da2 = datahandleer.getDataSimple("D:\\shared\\temp1.csv")
            da = np.concatenate((da1,da2),0)
            da = np.reshape(da,(2,10000,1))
            y_predict= model.predict(da)
            s= ""
            for o in y_predict:
                index = np.argmax(o)
                if index == 0:
                    s = s+str(2)
                    print("2号电源")
                else:
                    print("3号电源")
                    s = s+str(3)
            m.seek(0)
            m.write(s.encode())
            m.flush()
            sleep(5)
        except:
            print("文件正在被占用")
        finally:
            print("开始下一轮查询")


    