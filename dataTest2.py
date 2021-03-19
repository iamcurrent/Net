
import scipy.io
import os
import re
import numpy as np
from numpy.random import rand,randn
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import csv


import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Input
from    tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.module.module import valid_identifier
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D

def VGG16(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)
    # 1st block
    x = Conv2D(64, (3,3), activation='relu', padding='same',name='conv1a')(input_tensor)
    x = Conv2D(64, (3,3), activation='relu', padding='same',name='conv1b')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool1')(x)
    # 2nd block
    x = Conv2D(128, (3,3), activation='relu', padding='same',name='conv2a')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same',name='conv2b')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool2')(x)
    # 3rd block
    x = Conv2D(256, (3,3), activation='relu', padding='same',name='conv3a')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same',name='conv3b')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same',name='conv3c')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool3')(x)
    # 4th block
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv4a')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv4b')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv4c')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool4')(x)
    # 5th block
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv5a')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv5b')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='conv5c')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'pool5')(x)
    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation='relu',  name='fc6')(x)
    # x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    # x = Dropout(0.5)(x)
    output_tensor = Dense(nb_classes, activation='softmax', name='fc8')(x)

    model = Model(input_tensor, output_tensor)
    return model






def getdata(file_path):
    data = []
    count = 0
    with open(file_path) as f:
        reader = csv.reader(f)
        for o in reader:
            if count==0:
                count+=1
                continue
            data_t = []
            for j in range(len(o)-1):
                data_t.append(float(o[j]))
            data.append(data_t)
    data_ss = np.array(data)
    return data_ss

path = "G:\\迅雷下载\\Electric_Motor_Temperature\\data\\data\\pmsm_temperature_data.csv"
data = getdata(path)
training_set = data[0:4000,0:1]  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set =data[4000:6000,0:1]   # 后300天的开盘价作为测试集
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化


x_train = []
y_train = []

x_test = []
y_test = []
# 测试集：csv表格中前2426-300=2126天数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
# 测试集：csv表格中后300天数据
# 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))


def mymodel():
    input_tensor  = Input(shape=(60,1))
    x = layers.Conv1D(32,3,activation='relu',padding='same')(input_tensor)
    x = layers.MaxPool1D(pool_size=3,strides = 3)(x)
    x = layers.Conv1D(64,3,strides = 1,activation='relu',padding='same')(x)
    x = keras.layers.MaxPool1D(pool_size=3,strides=3)(x)
    out_tensor = layers.Dense(1)(x)
    model = Model(input_tensor,out_tensor)
    return model



if __name__=="__main__":
    data_path = 'C:\\Users\\Admin\\Desktop\\nasa\\nasa'
    file_name = os.listdir(data_path)
    for o in file_name:
        fileName = data_path+'\\'+o
        data = scipy.io.loadmat(fileName)
        data = data[o.split('.')[0]]
        for i in range(len(data[0][0][0][0])):

            data1 = data[0][0][0][0,i]
            data1 = data1[3]
            data1 = data1[0,0]

            for i in range(5):
                plt.subplot(len(data1),1,i+1)
                plt.plot(data1[5][0],data1[i][0])
            plt.show()
    
    