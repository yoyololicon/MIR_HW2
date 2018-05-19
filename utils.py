import librosa.core as core
import scipy.signal as signal
from scipy.fftpack import ifft
from scipy.signal import argrelmax
import numpy as np

genres_dir = ['ChaChaCha', 'Jive', 'Quickstep', 'Rumba-American', 'Rumba-International', 'Rumba-Misc', 'Samba', 'Tango',
              'VienneseWaltz', 'Waltz']
genres = ['ChaChaCha', 'Jive', 'Quickstep', 'Rumba', 'Samba', 'Tango', 'VienneseWaltz', 'Waltz']
data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Ballroom/BallroomData'
bpm_label_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Ballroom/BallroomAnnotations/ballroomGroundTruth'
beat_label_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Ballroom/BallroomAnnotations-beat'


def spectral_flux(data, sr, hop_size, window_size, g, mean_size, lag=1):
    x = core.stft(data, n_fft=window_size, hop_length=hop_size)
    t = np.arange(x.shape[1]) * hop_size / sr

    y = np.log(1 + g * np.abs(x))
    spfx = np.maximum(0., y[:, lag:] - y[:, :-lag]).mean(axis=0)
    t2 = t[:-lag] + (t[lag:] - t[:-lag]) / 2

    # post-processing
    filter = np.ones(mean_size) / mean_size
    u = signal.fftconvolve(spfx, filter, 'same')
    spfx_enhance = np.maximum(0., spfx - u)
    spfx_enhance /= spfx_enhance.max()
    return t2, spfx_enhance


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


def stacf(data, sr, window_size, hop_size):
    if hop_size:
        noverlap = window_size - hop_size
    else:
        noverlap = None
    _, t, x = signal.stft(data, sr, nperseg=window_size, noverlap=noverlap, return_onesided=False)
    acf = ifft(np.abs(x) ** 2, axis=0).real
    acf = acf[:acf.shape[0] // 2 + 1]
    lag = np.arange(acf.shape[0]) / sr
    return lag, t, acf


def tempo_estimation(freq_scale, tempogram):
    tempo_vector = np.sum(tempogram, axis=1)
    peak_idx = argrelmax(tempo_vector)
    peaks = sorted(zip(tempo_vector[peak_idx], freq_scale[peak_idx]), key=lambda x: x[0], reverse=True)
    pack = peaks[:2]
    if pack[0][1] > pack[1][1]:
        pack = pack[::-1]
    s1 = pack[0][0] / (pack[0][0] + pack[1][0])
    # return t1, t2, s1
    return pack[0][1], pack[1][1], s1


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
