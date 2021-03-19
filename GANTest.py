import re
import cv2
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
####################################################################### 实现数据生成
data = getdata("G:\\迅雷下载\\Electric_Motor_Temperature\\data\\data\\pmsm_temperature_data.csv")
training_set = data[0:4000,0:1]  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set =data[4000:6000,0:1]   # 后300天的开盘价作为测试集
sc = MinMaxScaler(feature_range=(0, 1)) 
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
#################################################################################################

#获取真实的数据样例
def get_real_samples(n):
    x1 = rand(n) - 0.5
    x2 = x1*x1 + x1*x1*x1 +0.1
    x1 = x1.reshape(n,1)
    x2 = x2.reshape(n,1)
    X = np.hstack((x1,x2)) ##对数据进行拼接
    y = np.ones((n,1))
    return X,y


#判别器
def disc_model(input_dim = 2):
    model = Sequential()
    model.add(Dense(25,activation='relu',input_dim=input_dim)) ##25个维度的输入层，输入使2维的
    model.add(Dense(1,activation='sigmoid')) ##1个维度的输出层
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc']) ##设置编译参数
    return model

#生成器
def get_fakae_samples(gen,noise_dim,n):
    x_input = noise_points(noise_dim,n)
    x = gen.predict(x_input) ##使用网络对数据进行预测,生成数据
    y = np.zeros((n,1))
    return x,y

##生成噪声数据

def noise_points(noise_dim,n):
    noise = randn(n*noise_dim)
    noise = noise.reshape(n,noise_dim)
    return noise
##生成器模型
def gen_model(input_dim,output_dim=2):
    model = Sequential()
    model.add(Dense(15,activation='relu',input_dim=input_dim))
    model.add(Dense(output_dim,activation='linear')) ##输出维度是2，就是一个坐标点
    return model

##对抗生成网络
def gan_model(disc,gen):
    disc.trainable = False
    model = Sequential()
    model.add(gen)
    model.add(disc)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model


def get_real_samples1(batch_size,i,data):
    y = np.ones((batch_size,1))
    r_x = data[batch_size*i:batch_size*i+batch_size,0:2]
    return r_x,y
##训练模型
def train(g_model,d_model,gan_model,noise_dim,epochs=20000,batch_size = 250,n_eval=1000):
    half_batch = batch_size//2
    for i in range(epochs):
        x_real,y_real = get_real_samples(half_batch)
        x_fake,y_fake = get_fakae_samples(g_model,noise_dim,half_batch)
        d_model.train_on_batch(x_real,y_real)
        g_model.train_on_batch(x_fake,y_fake)
        x_gen = noise_points(noise_dim,batch_size)
        y = np.ones((batch_size,1))
        gan_model.train_on_batch(x_gen,y)
        if (i+1)%n_eval == 0:
            show_performance(i+1,g_model,d_model,noise_dim)

def show_performance(epoch,g_model,d_model,noise_dim,n=100):
    x_real,y_real = get_real_samples(n)
    _,real_acc = d_model.evaluate(x_real,y_real,verbose=0)
    x_fake,y_fake = get_fakae_samples(g_model,noise_dim,n)
    _,fake_acc = d_model.evaluate(x_fake,y_fake,verbose=0)
    print(epoch,real_acc,fake_acc)
    plt.figure(figsize=(20,10))
    plt.scatter(x_real[:,0],x_real[:,1],color="red")
    plt.scatter(x_fake[:,0],x_fake[:,1],color='blue')
    plt.show()



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

def model1():
    model = tf.keras.Sequential()
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(0.001),metrics=['acc'])
    train_x = np.random.random((1000,72))
    train_y = np.random.random((1000,10))
    val_x = np.random.random((200,72))
    val_y = np.random.random((200,10))
    model.fit(train_x,train_y,epochs=10,batch_size=100,validation_data=(val_x,val_y))
    


# 采样网络
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon
# 编码器
class Encoder(layers.Layer):
    def __init__(self, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
        
# 解码器
class Decoder(layers.Layer):
    def __init__(self, original_dim, 
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)
    
# 变分自编码器
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32, 
                intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
    
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                              intermediate_dim=intermediate_dim)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        kl_loss = -0.5*tf.reduce_sum(
            z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstructed


def slfmodel(input_dim):
    input_layers = Input(shape=input_dim)
    x = keras.layers.Dense(600,activation='relu')(input_layers)
    x = keras.layers.Dense(300,activation='relu')(x)
    output_data = keras.layers.Dense(1,activation='linear')(x)
    model = keras.Model(input_layers,output_data)
    return model


if __name__=="__main__":
  
    '''
    path = 'I:\\Desk\\tf2.0study\\1.jpg'
    img = cv2.imread(path,0)
    img = cv2.resize(img,(28,28))
    img = img.reshape((1,784))
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    vae = VAE(784,32,64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError(),metrics=['acc'])
    vae.fit(x_train, x_train, epochs=3, batch_size=64)
    vae.evaluate(img)
    '''
    
    noise_dim = 5
    gen = gen_model(noise_dim)
    disc = disc_model()
    gan = gan_model(disc,gen)
    
    train(gen,disc,gan,noise_dim)
    
    '''
    model=VGG16(3, (224, 224, 3))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    model.fit()
    model.summary()
    '''

