import librosa.core as core
import scipy.signal as signal
from scipy.fftpack import ifft
import numpy as np

genres = ['ChaChaCha', 'Jive', 'Quickstep', ('Rumba-American', 'Rumba-International', 'Rumba-Misc'), 'Samba', 'Tango',
          'VienneseWaltz', 'Waltz']
data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Ballroom/BallroomData'
label_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Ballroom/BallroomAnnotations/ballroomGroundTruth'

def spectral_flux(data, sr, hop_size, window_size, g, mean_size, lag=1):
    x = core.stft(data, n_fft=window_size, hop_length=hop_size)
    t = np.arange(x.shape[1]) * hop_size / sr

    y = np.log(1 + g * np.abs(x))
    spec_flux = np.maximum(0., y[:, lag:] - y[:, :-lag]).mean(axis=0)
    t2 = t[:-lag] + (t[lag:] - t[:-lag]) / 2

    # post-processing
    filter = np.ones(mean_size) / mean_size
    u = signal.fftconvolve(spec_flux, filter, 'same')
    spec_flux_enhance = np.maximum(0., spec_flux - u)
    return t2, spec_flux_enhance


def p_score(g, t1, t2, s1):
    p = 0
    if abs((g - t1) / g) <= 0.08:
        p += s1
    if abs((g - t2) / g) <= 0.08:
        p += 1 - s1
    return p


def alotc(g, t1, t2):
    if abs((g - t1) / g) <= 0.08:
        return 1
    elif abs((g - t2) / g) <= 0.08:
        return 1
    else:
        return 0


def stacf(data, sr, hop_size, window_size):
    if hop_size:
        noverlap = window_size-hop_size
    else:
        noverlap = None
    _, t, x = signal.stft(data, sr, nperseg=window_size, noverlap=noverlap, return_onesided=False)
    acf = ifft(np.abs(x) ** 2, axis=0).real
    acf = acf[:acf.shape[0]//2+1]
    lag = np.arange(acf.shape[0]) / sr
    return lag, t, acf


def bpm_tempogram(tpg, lag, bpm_max, bpm_min):
    if len(lag) != tpg.shape[0]:
        exit(1)

    bpm_tpg = np.zeros((bpm_max - bpm_min, tpg.shape[1]))
    bpm_idx = np.arange(bpm_min - 0.5, bpm_max + 0.5, 1)
    for i in range(bpm_max - bpm_min):
        c1, c2 = 60 / bpm_idx[i], 60 / bpm_idx[i + 1]
        p_idx = (lag < c1) & (lag > c2)
        p_value = lag[p_idx]
        if len(p_value) > 0:
            bpm_tpg[i, :] = np.mean(tpg[p_idx, :], axis=0)

    return bpm_tpg