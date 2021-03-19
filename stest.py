import matplotlib.pyplot as plt
import numpy as np
import csv
import os

from numpy.core.fromnumeric import mean
def getData(path):
    resData = []
    with open(path) as f:
        reader = csv.reader(f)
        for o in reader:
            resData.append(float(o[4]))
    return resData

def averFilter(data,n):
    return np.convolve(data, np.ones((n,))/n, mode='same')

if __name__ == "__main__":
    
    path = "D:\\TEK0002.CSV"
    res = getData(path)
    FS = 200
    x = np.linspace(0,10,2500)
    noise = np.random.normal(size=(1,len(x)))
    y1 = np.cos(2*np.pi*50*x)
    y2 = np.cos(2*np.pi*20*x)

    y = 0.5*y1 + 0.5*y2 + noise[0,:]

    fft = np.fft.fft(y)
    fft = np.abs(fft)
    x_axis = np.linspace(0,int(len(x)/2),int(len(x)/2))*FS/len(x)
    plt.plot(x_axis,fft[0:int(len(x)/2)])
    plt.show()

    '''
    fileNames = os.listdir("D:\\CSVData\\")
    res = getData(path+fileNames[len(fileNames)-3])
  


    
    FS = 500000
    x = np.linspace(1,10,500000)
    noise = np.random.randn(1,len(x))
    noise1 = np.random.normal(size=(1,len(x)))
    y = np.cos(2*np.pi*50*x)+10*np.cos(2*np.pi*100*x)+np.cos(2*np.pi*150*x)
    y1 = np.cos(2*np.pi*200000*x)
    freq = np.linspace(0,250,250)*FS/500
    #y = np.cos(2*np.pi*50*x)*np.sin(2*np.pi*150*x)+noise1[0,0:]
    
    fft = np.fft.fft(y1)
    abss = np.fft.fftshift(fft)
    abss = np.abs(abss)
    
    
   
  
    #plt.plot(x[0:int(len(x)/2)],abss[0:int(len(x)/2)])
    plt.subplot(2,1,1)
    plt.plot(y1)
    plt.subplot(2,1,2)
    plt.plot(abss[int(len(abss)/2):])
    plt.show()
    '''