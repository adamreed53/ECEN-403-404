import pandas as pd
import numpy as np
import scipy.stats.morestats
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
from scipy import stats
import sklearn.metrics
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from pywt import wavedec
import statistics
import copy

dataset = pd.read_csv(r"C:\Users\adamr\OneDrive\Desktop\403\pddata.csv", header=None, index_col = False)
Time = dataset.iloc[:,0].values
Voltage = dataset.iloc[:,1].values
V = copy.deepcopy(Voltage)
##I need the voltage data as a list

# Number of sample points


sigma = 0.5  # White noise
Noisy = Voltage + sigma * np.random.randn(Voltage.size)  #Adds random gaussian white noise
print(len(Voltage))
for i in Voltage:  #removes random gaussian white noise centered at zero
    if abs(i) < 0.5:
        i = 0.0


# Daubuche and Symlet families of mother wavelets to compare
WName = ['db2','db4', 'db6', 'db8', 'db10', 'db12', 'db14', 'db16', 'db18', 'db20', 'sym2', 'sym3', 'sym4', 'sym5',
         'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17',
         'sym18', 'sym19', 'sym20']

ii = 0
WNamekept = WName[0]

while ii < len(WName):

    Denoised=denoise_wavelet(Noisy, wavelet=WName[ii], mode='hard', wavelet_levels=7, method='VisuShrink', rescale_sigma='True')
    MSE = sklearn.metrics.mean_squared_error(V,Denoised)

    if ii == 0:
        MSEkept=MSE
    if MSE < MSEkept:
        MSEkept = MSE
        WNamekept = WName[ii]
    ii += 1
Started = sklearn.metrics.mean_squared_error(V,Noisy)

plt.plot(Time,Noisy)
plt.plot(Time,Denoised)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend(['Noisy', 'Denoised'])
plt.show()

percentage_noise_removed = MSEkept/Started

#Feature Extraction - split into chunks of 500
chunksize = 500
chunkedlist = []
for i in range(0,len(Denoised)- len(Denoised)%chunksize ,chunksize):  #length of chunked list is 474
    chunkedlist.append(Denoised[i:i+chunksize])

features = []
firstrow = np.arange(1, 43)
#features.append(firstrow)

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