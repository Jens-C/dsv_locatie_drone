from numpy.random import seed
from numpy import fft
from scipy.signal import find_peaks


import scipy.io as sio
import numpy as np
import scipy
import matplotlib.pyplot as plt


def calculate_delay_between_samples(sample_frequencies):
    # Calculate the delays
    delays = 1 / sample_frequencies

    # Calculate the delay between samples
    delta_tau = np.diff(delays)

    return delta_tau


def channel2APDP(positie):
    num_positions =  positie.shape[1]
    APDP_list = []
    APDP_list = []
    PDP=np.zeros((100,200))
    for i in range(num_positions):

        impulsrespons = np.abs(scipy.fft.ifft(positie[:, i]))
        PDP[i] = np.square(impulsrespons)

    for i in range(200):
        APDP_list.append(np.sum(PDP[:,i])/len(PDP[:,i]))
    return APDP_list
def APDP2delays(APDP):
    # zoek de pieken
    pieken, _ = find_peaks(APDP)
    sorted_peaks = sorted(pieken, key=lambda x: APDP[x], reverse=True)#lambda nodig om de values te krijgen
    delays = np.abs(calculate_delay_between_samples(np.arange(1e9, 3e9 + 1e7, 1e7)))
    return delays[sorted_peaks[0]], delays[sorted_peaks[1]]


def calculate_location(tau0, tau1):
    afstand0= tau0*3e8
    afstand1= tau1*3e8

    yn=(np.square(afstand1)-np.square(afstand0))/4
    xn=np.sqrt(np.square(afstand0)-np.square(yn-1))
    
    return xn, yn

mat_contents = sio.loadmat('Dataset_1.mat')
matrix = mat_contents['H']   #200 frequentietonen, 25 posities, 100 kopies voor foutmeting


APDP_values = channel2APDP(matrix[:,1,:])
val0, val1 = APDP2delays(APDP_values)

delays = calculate_delay_between_samples(np.arange(1e9, 3e9 + 1e7, 1e7))


print(val0*3e8)
print(val1*3e8)



print(calculate_location(1/1e9+ val0*1e7,1/1e9+ val1*1e7))
plt.plot( delays,APDP_values)

plt.grid(True)
plt.show()