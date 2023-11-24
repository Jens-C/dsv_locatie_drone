from numpy.random import seed
from numpy import fft
from scipy.signal import find_peaks


import scipy.io as sio
import numpy as np
import scipy
import matplotlib.pyplot as plt

def channel2APDP(positie):
    num_positions =  positie.shape[1]
    APDP_list = []
    APDP_list = []
    PDP=np.zeros((100,200))
    print(num_positions)

    for i in range(num_positions):

        impulsrespons = np.abs(scipy.fft.ifft(positie[:, i]))
        PDP[i] = np.square(impulsrespons)

    for i in range(200):
        print(len(PDP[:,i]))
        APDP_list.append(np.sum(PDP[:,i])/len(PDP[:,i]))
    return APDP_list
def APDP2delays(APDP):
    # zoek de pieken
    pieken, _ = find_peaks(APDP)
    sorted_peaks = sorted(pieken, key=lambda x: APDP[x], reverse=True)#lambda nodig om de values te krijgen
    
    return APDP[sorted_peaks[0]], APDP[sorted_peaks[1]]

mat_contents = sio.loadmat('Dataset_1.mat')
matrix = mat_contents['H']   #200 frequentietonen, 25 posities, 100 kopies voor foutmeting


APDP_values = channel2APDP(matrix[:,1,:])
val0, val1 = APDP2delays(APDP_values)

print( val0)
print(val1)

plt.plot(APDP_values)

plt.grid(True)
plt.show()