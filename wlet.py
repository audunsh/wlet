from numpy.fft import *
from scipy import signal, integrate, io
import numpy as np
from scipy.io import wavfile

def mwave(K,f_a, f_s,t_, t):
    """
    Morlet wavelet with K periods at frequency f_a, samplerate f_S, centered at t_, returned at times t
    """
    C_1 = np.sqrt(np.sqrt(2*np.pi)*f_a/(K*f_s)) 
    return C_1 * np.exp(1j*2*np.pi*f_a*(t-t_))*np.exp(-.5*(2*np.pi*f_a*(t-t_)/K)**2)

def kwave(K,f_a, f_s,t):
    """
    the fourier transformed (analytical) of the above expression
    """
    N = 2**nextpow2(len(t))
    fut_n = np.linspace(1,N+1, int(N))
    eta = (fut_n - 1)*K*f_s/((N+1) *f_a)
    C_2 = np.sqrt(np.sqrt(2*np.pi)*f_a/(K*f_s))**-1
    
    return C_2*np.exp(-.5*(np.ones(int(N), dtype = float)*K-eta)**2.0)


def nextpow2(n):
    """
    Supporting function, returns next power of 2
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return m_i

def nu(K, f_a, f_s, N):
    """
    Supporting function to fourier transformed wavelet
    """
    d = K*f_s/((N+1)*f_a)
    return d*np.linspace(0,N-1,N)

def mwave_fft(t, signal, fs, fa, K, t0, t1):
    """
    FFT-based wavelet transform
    """
    t = np.linspace(t0,t1,int((t1-t0)*fs))
    L = len(t)
    N = int(2**nextpow2(L))
    fut = (fs/2) *np.linspace(0,1,int(N/2))
    data_f = fft(signal, N)
    
    return np.abs(ifft(data_f*kwave(K, fa, fs, t))) 

def mwave_scan(t, signal, K = 20, freq_a = [100]):
    """
    Scan signal for intensities in frequency range freq_a
    """
    Nt = 2**nextpow2(len(t))
    dt = t[1]-t[0]
    t_resample = np.linspace(0,dt*(Nt-1), int(Nt))

    signal_resample = np.zeros(len(t_resample))
    signal_resample[:len(signal)] = signal
    
    fs = len(t)/(t[-1]-t[0])
    Z = np.zeros((len(t), len(freq_a)))
    for i in np.arange(len(freq_a)):
        Z[:, i] = mwave_fft(t, signal, fs, freq_a[i], K, t[0], t[-1])[0:len(t)]
    return Z.T


def numint(x,f1,f2):
    """
    numint support
    """
    return (x[-1]-x[0])*mean(f1(x)*f2(x))

def dumint(D,f):
    """
    dumint support
    """
    return mean(f)*D

def bvec(fra, f0):
    """
    bvec support
    """
    return exp(-.01*(fra - f0)**2)
    
"""    
# Usage example

t = np.linspace(0,10, 2500) # time axis
f0 = 50                     # frequency
f = np.sin(2*np.pi*f0*t) +  .75*np.sin(np.pi*f0*t) # a signal 
freq = np.linspace(1, 100, 1400) #frequency range to scan


# scan signal (tip: try varying K to optimize time/frequency resolution)
X = mwave_scan(t, f, freq_a = freq, K = 20) 

# plot results
import matplotlib.pyplot as plt

plt.figure(1, figsize = (10,5))
plt.contourf(t, freq, X)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
"""