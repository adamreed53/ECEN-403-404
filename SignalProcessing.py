import pandas as pd
import numpy as np
import scipy.stats.morestats
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
from scipy import stats
from math import log10,sqrt
import sklearn.metrics
import matplotlib.pyplot as plt
from pywt import wavedec
import statistics
import copy
import pyyawt

def mse( original, denoised):
    sum = 0.0
    for i in range(len(original)):
        sum = sum + original[i]
    mean1 = sum / len(original)
    sum = 0.0
    for i in range(len(denoised)):
        sum = sum + denoised[i]
    mean2 = sum / len(denoised)
    mse1 = (mean1-mean2)**2
    return mse1

def PSNR(original, compressed):
    num = mse(original, compressed)
    max_pixel = np.max(original)**2
    psnr = 20 * log10(max_pixel / sqrt(num))
    return psnr

dataset = pd.read_csv(r"C:\Users\adamr\OneDrive\Desktop\403\pd30mm75mmdc.csv")
Time = dataset.iloc[:,0].values
Voltage = dataset.iloc[:,1].values
V = copy.deepcopy(Voltage)
print(Time)
print(Voltage)
##I need the voltage data as a list

# Number of sample points


Sigma = 0.5  # White noise
Noisy = Voltage + Sigma * np.random.uniform(-1000,1000, Voltage.size)  #Adds random gaussian white noise

for i in Voltage:  #removes random gaussian white noise centered at zero
    if abs(i) < 0.5:
        i = 0.0


# Daubuche and Symlet families of mother wavelets to compare
WName = ['db2','db4', 'db6', 'db8', 'db10', 'db12', 'db14', 'db16', 'db18', 'db20', 'sym2', 'sym3', 'sym4', 'sym5',
         'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17',
         'sym18', 'sym19', 'sym20']

ii = 0
WNamekept = WName[0]
MSEkept = sklearn.metrics.mean_squared_error(V,Noisy)
MSEA = mse(V, Noisy)
while ii < len(WName):
    [Denoised, C ,L] = pyyawt.denoising.wden(Noisy, 'rigrsure', 's', 'one', 7, WName[ii])  #performs actual denoising
    MSEB = mse(Voltage,Denoised)
    MSEC = MSEA-MSEB
    if MSEC < 0:
        ii+=1
        continue
    MSE = MSEC
    if ii == 0:
        MSEkept=MSE
    if MSE > MSEkept:
        MSEkept = MSE
        WNamekept = WName[ii]
        print("mse kept = ")
        print(MSEkept)
        psnrkept1 = PSNR(Noisy, Denoised)
        psnrkept2 = PSNR(Voltage, Denoised)
        psnrkept3 = PSNR(Voltage, Noisy)
    ii += 1
    #Denoised needs updating
started1 = mse(V,Noisy)
'''
print("Between Voltage and Noisy")
print('sklearn', Started)
print('mse', started1)
print('np', started2)
print('msekept ', MSEkept)
print("psnr noisy and denoised", psnrkept1)
print('psnr voltage and denoised', psnrkept2)
print('psnr voltage and noisy', psnrkept3)
'''
plt.plot(Time,Noisy)
plt.plot(Time,Denoised)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend(['Noisy', 'Denoised'])
plt.show()


print(WNamekept)
print(Started)
print(MSEkept)

#Feature Extraction - split into chunks of 500
chunksize = 500
chunkedlist = []
for i in range(0,len(Denoised) - len(Denoised)%chunksize ,chunksize):
    chunkedlist.append(Denoised[i:i+chunksize])

features = []
firstrow = np.arange(1, 43)   #Just the numbers 1 through 43


for list in chunkedlist:
    coeffs = wavedec(list,WNamekept,level=7)
    [C,C7,C6,C5,C4,C3,C2,C1] = coeffs  #levels 1 - 7  kurtosis, skewness, variance, mean, max, min,
    #level1
    K1 = scipy.stats.kurtosis(C1)
    S1 = scipy.stats.skew(C1)
    V1 = statistics.variance(C1)
    A1 = np.mean(C1)
    Max1 = np.max(C1)
    Min1 = np.min(C1)
    #level2
    K2 = scipy.stats.kurtosis(C2)
    S2 = scipy.stats.skew(C2)
    V2 = statistics.variance(C2)
    A2 = np.mean(C2)
    Max2 = np.max(C2)
    Min2 = np.min(C2)
    # level3
    K3 = scipy.stats.kurtosis(C3)
    S3 = scipy.stats.skew(C3)
    V3 = statistics.variance(C3)
    A3 = np.mean(C3)
    Max3 = np.max(C3)
    Min3 = np.min(C3)
    # level4
    K4 = scipy.stats.kurtosis(C4)
    S4 = scipy.stats.skew(C4)
    V4 = statistics.variance(C4)
    A4 = np.mean(C4)
    Max4 = np.max(C4)
    Min4 = np.min(C4)
    # level5
    K5 = scipy.stats.kurtosis(C5)
    S5 = scipy.stats.skew(C5)
    V5 = statistics.variance(C5)
    A5 = np.mean(C5)
    Max5 = np.max(C5)
    Min5 = np.min(C5)
    # level6
    K6 = scipy.stats.kurtosis(C6)
    S6 = scipy.stats.skew(C6)
    V6 = statistics.variance(C6)
    A6 = np.mean(C6)
    Max6 = np.max(C6)
    Min6 = np.min(C6)
    # level7
    K7 = scipy.stats.kurtosis(C7)
    S7 = scipy.stats.skew(C7)
    V7 = statistics.variance(C7)
    A7 = np.mean(C7)
    Max7 = np.max(C7)
    Min7 = np.min(C7)
    row = [K1,S1,V1,A1,Max1,Min1,K2,S2,V2,A2,Max2,Min2,K3,S3,V3,A3,Max3,Min3,K4,S4,V4,A4,Max4,Min4,K5,S5,V5,A5,Max5,Min5,K6,S6,V6,A6,Max6,Min6,K7,S7,V7,A7,Max7,Min7]
    features.append(row)

df = pd.DataFrame(features, columns=firstrow)
print(df)

