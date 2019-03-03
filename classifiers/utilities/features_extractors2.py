import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, iqr, kurtosis
from scipy.signal import welch
from statsmodels.robust import mad
from sklearn.preprocessing import MinMaxScaler
from utilities import tools
from utilities.detect_peaks import detect_peaks

## Useful functions

# To normalize data
def normalize(X_raw):
    X = []
    scaler = MinMaxScaler()
    for row in X_raw:
       scaler.fit(row)
       X.append(scaler.transform(row))
    return np.array(X)
 
# To compute Signal Magnitude Area       
def sma(x, y, z):
    sum = 0
    for i in range(len(x)):
        sum += abs(x[i]) + abs(y[i]) + abs(z[i])
    return sum/len(x)

# Fourier transform
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2) # Only take half (real part)
    fft_values_ = np.fft.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

# Power Spectral Density
def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

# To plot the FFT of a signal s
def plot_fft(s):    
    t_n = 2 # duration of sample (should ask)
    N = 128
    T = t_n / N
    f_s = 1/T
    
    f_values, fft_values = get_fft_values(s, T, N, f_s)

    plt.plot(f_values, fft_values, linestyle='-', color='blue')
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.title("Frequency domain of the signal", fontsize=16)
    plt.show()
  
# To plot the PSD of a signal s
def plot_psd(s):
    t_n = 2 # duration of sample (should ask)
    N = 128
    T = t_n / N
    f_s = 1/T
     
    f_values, psd_values = get_psd_values(s, T, N, f_s)
     
    plt.plot(f_values, psd_values, linestyle='-', color='blue')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2 / Hz]')
    plt.show()
    
# Computes autocorrelation
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(s, T, N, f_s):
    autocorr_values = autocorr(s)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

# Plots autocor
def plot_autocor(s):
    t_n = 2 # duration of sample (should ask)
    N = 128
    T = t_n / N
    f_s = 1/T
     
    t_values, autocorr_values = get_autocorr_values(s, T, N, f_s)
     
    plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
    plt.xlabel('time delay [s]')
    plt.ylabel('Autocorrelation amplitude')
    plt.show()
    
# To have a fixed number of peaks
def get_first_n_peaks(peaks, no_peaks=5):
    peaks = list(peaks)
    if len(peaks) >= no_peaks:
        return peaks[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(peaks)
        return peaks + [0]*missing_no_peaks
    
# Determines number of peaks in FFT, PSD and autocorr signals
def peaks(s):
    t_n = 2 # duration of sample (should ask)
    N = 128
    T = t_n / N
    f_s = 1/T
    peaks = []
    
    percentile = 5
    denominator = 3
    
    fun = [get_fft_values, get_psd_values, get_autocorr_values]
    for f in fun:
        new_s = f(s, T, N, f_s)[1]
        signal_min = np.nanpercentile(new_s, percentile)
        signal_max = np.nanpercentile(new_s, 100-percentile)
        mph = signal_min + (signal_max - signal_min)/denominator
        peaks.append(get_first_n_peaks(detect_peaks(new_s, mph=mph)))
    return peaks

# To have all possible labels
def features_labels():
    fl = []
    channels = ["O-x", "O-y", "O-z", "O-w", "AV-x", "AV-y", "AV-z", "LA-x", "LA-y", "LA-z"]
    labels = ['mean', 'var', 'std', 'min', 'max', 'skew', 'median', 'i_max', 'imin', 'inter_range', 'mad', 'kurto', 'rms']
    for channel in channels:
        for label in labels:
            fl.append("{} {}".format(channel, label))
    fl.append('AV sma')
    fl.append('LA sma')
    return fl


## Features exctraction

# With orientation
def features_extraction(X_raw):
    X = [[]]*len(X_raw)
    for i in range(len(X)):
        features = []
        # Mean
        for j in range(10):
            features.append(np.mean(X_raw[i, j]))
        # Variance
        for j in range(10):
            features.append(np.var(X_raw[i, j]))
        # Standard deviation
        for j in range(10):
            features.append(np.std(X_raw[i, j]))
        # Minimum
        for j in range(10):
            features.append(np.min(X_raw[i, j]))
        # Maximum
        for j in range(10):
            features.append(np.max(X_raw[i, j]))
        # Skewness
        for j in range(10):
            features.append(skew(X_raw[i, j]))
        # Median
        for j in range(10):
            features.append(np.median(X_raw[i, j]))
        # Index maximum
        for j in range(10):
            features.append(np.argmax(X_raw[i, j]))
        # Index minimum
        for j in range(10):
            features.append(np.argmin(X_raw[i, j]))
        # Interquartile range
        for j in range(10):
            features.append(iqr(X_raw[i, j]))
        # Mean absolute deviation
        for j in range(10):
            features.append(mad(X_raw[i, j]))
        # Kurtosis
        for j in range(10):
            features.append(kurtosis(X_raw[i, j]))
        # RMS
        for j in range(10):
            features.append(np.sqrt(np.mean(np.square(X_raw[i, j]))))      
        # Signal magnitude area
        features.append(sma(X_raw[i, 4], X_raw[i, 5], X_raw[i, 6]))
        features.append(sma(X_raw[i, 7], X_raw[i, 8], X_raw[i, 9]))
        # Local extremum
        for j in range(10):
            features.append(tools.count_local_maximum(tools.convolution_smooth(X_raw[i, j])))
        # Euler angles for every features
        for j in range(0, 140, 10):
            for angle in tools.quaternionToEulerAngles(features[j+0], features[j+1], features[j+2], features[j+3]):
                features.append(angle)
        # Peaks in frequential domain
#        for j in range(10):
#            peaks_index = peaks(X_raw[i, j])
#            for arr in peaks_index:
#                for peak in arr:
#                    features.append(peak)

        # Other extractors usable if we fix them
        # Entropy
        #X_normalized = normalize(X_raw)
        #for j in range(10):
        #    features.append(entropy(X_normalized[i, j]))
        
        X[i] = features
    
    return np.array(X)

# Without orientation
def features_extraction_no_ori(X_raw):
    X = [[]]*len(X_raw)
    for i in range(len(X)):
        features = []
        # Mean
        for j in range(4, 10):
            features.append(np.mean(X_raw[i, j]))
        # Variance
        for j in range(4, 10):
            features.append(np.var(X_raw[i, j]))
        # Standard deviation
        for j in range(4, 10):
            features.append(np.std(X_raw[i, j]))
        # Minimum
        for j in range(4, 10):
            features.append(np.min(X_raw[i, j]))
        # Maximum
        for j in range(4, 10):
            features.append(np.max(X_raw[i, j]))
        # Skewness
        for j in range(4, 10):
            features.append(skew(X_raw[i, j]))
        # Median
        for j in range(4, 10):
            features.append(np.median(X_raw[i, j]))
        # Index maximum
        for j in range(4, 10):
            features.append(np.argmax(X_raw[i, j]))
        # Index minimum
        for j in range(4, 10):
            features.append(np.argmin(X_raw[i, j]))
        # Interquartile range
        for j in range(4, 10):
            features.append(iqr(X_raw[i, j]))
        # Mean absolute deviation
        for j in range(4, 10):
            features.append(mad(X_raw[i, j]))
        # Kurtosis
        for j in range(4, 10):
            features.append(kurtosis(X_raw[i, j]))
        # RMS
        for j in range(4, 10):
            features.append(np.sqrt(np.mean(np.square(X_raw[i, j]))))      
        # Signal magnitude area
        features.append(sma(X_raw[i, 4], X_raw[i, 5], X_raw[i, 6]))
        features.append(sma(X_raw[i, 7], X_raw[i, 8], X_raw[i, 9]))
        # Local extremum
        for j in range(4, 10):
            features.append(tools.count_local_maximum(tools.convolution_smooth(X_raw[i, j])))
        # Peaks in frequential domain
        for j in range(4, 10):
            peaks_index = peaks(X_raw[i, j])
            for arr in peaks_index:
                for peak in arr:
                    features.append(peak)

        # Other extractors usable if we fix them
        # Entropy
        #X_normalized = normalize(X_raw)
        #for j in range(10):
        #    features.append(entropy(X_normalized[i, j]))

        X[i] = features
    
    return np.array(X)


