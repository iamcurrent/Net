import numpy as np
import os
import csv
import sys
import matplotlib.pyplot as plt
path = "D:\\CSVData\\"
fileNames=  os.listdir(path)

def getData(path):
    FS = 500000
    
   
    with open(path) as f:
        reader = csv.reader(f)
        mix = []
        s1 = []
        s2 = []
        for o in reader:
            mix.append(float(o[1]))
            s1.append(float(o[2]))
            s2.append(float(o[3]))
        
        
        N = 100000 ##数据点数
        freq=np.linspace(0,int(N/2),int(N/2))*FS/N
        huis = 2*np.pi*freq/FS
        print(np.max(huis))
        fft1 = getFrouir(mix[0:N])
        fft2 = getFrouir(s1[0:N])
        fft3 = getFrouir(s2[0:N])
        print(np.pi)
        fft1[0] = 0
        fft2[0] = 0
        fft3[0] = 0

        f1 = getMainPre(fft1,huis,FS,N)
        f2 = getMainPre(fft2,huis,FS,N)
        f3 = getMainPre(fft3,huis,FS,N)
        print("混合信号主频率:",f1,"信号1主频率:",f2,"信号2主频率:",f3)
        showData(huis,(fft1[0:int(N/2)],fft2[0:int(N/2)],fft3[0:int(N/2)]))
        
def showData(x,pa):
    c = len(pa)
    for i in range(0,c):
        plt.subplot(c,1,i+1)
        plt.plot(x,pa[i])
    plt.show()

def getFrouir(data):
    fft = np.fft.fft(data)
    fft = np.abs(fft)
    return fft

def getMainPre(fft1,huis1,FS1,N1):
    a1 = np.argmax(fft1[0:int(N1/2)])
    k = huis1[a1]
    f1 = FS1/(2*np.pi*k)
    return f1

getData(path+fileNames[0])


