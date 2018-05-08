from librosa.core import load, stft
from librosa.util import normalize
from scipy.signal import argrelmax
import numpy as np
from utils import spectral_flux, p_score, genres, data_dir, label_dir
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def tempo_estimation(file_name, window1=2048, window2=1024, lw_sr=100, g=1, mean_size=25):
    data, sr = load(file_name, sr=None)
    hop_size = sr // lw_sr
    t, nv_curve = spectral_flux(data, sr, hop_size, window1, g, mean_size, lag=1)

    f_tpg = normalize(np.abs(stft(nv_curve, n_fft=window2)), axis=0)
    freq_scale = np.arange(window2) * lw_sr / window2 * 60

    tempo_vector = np.sum(f_tpg, axis=1)
    peak_idx = argrelmax(tempo_vector)
    peaks = sorted(zip(tempo_vector[peak_idx], peak_idx[0]), key=lambda x: x[0], reverse=True)
    pack = [(peaks[0][0], freq_scale[peaks[0][1]]), (peaks[1][0], freq_scale[peaks[1][1]])]
    if pack[0][1] > pack[1][1]:
        return reversed(pack)
    else:
        return pack


if __name__ == '__main__':
    table = PrettyTable(["Genre", "Average P-score", "Average ALOTC"])
    ratio_list = []
    for genre in genres:
        if type(genre) != type(''):
            score = []
            for sub_genre in genre:
                dir = os.path.join(data_dir, sub_genre)
                files = os.listdir(dir)
                for file_name in files:
                    ((t1v, t1), (t2v, t2)) = tempo_estimation(os.path.join(dir, file_name))
                    s1 = t1v / (t1v + t2v)

                    with open(os.path.join(label_dir, file_name.replace('.wav', '.bpm').split('/')[-1])) as label_file:
                        gt = int(label_file.readline())
                        score.append(p_score(gt, t1, t2, s1))
                        print(t2 / t1, t1 / gt, t2 / gt)

            score = np.array(score)
            average_p = score.mean()
            score[score > 0] = 1
            average_alotc = score.mean()
            table.add_row(["Rumba", "{:.4f}".format(average_p), "{:.4f}".format(average_alotc)])
        else:
            dir = os.path.join(data_dir, genre)
            files = os.listdir(dir)
            score = []
            for file_name in files:
                ((t1v, t1), (t2v, t2)) = tempo_estimation(os.path.join(dir, file_name))
                s1 = t1v / (t1v + t2v)

                with open(os.path.join(label_dir, file_name.replace('.wav', '.bpm').split('/')[-1])) as label_file:
                    gt = int(label_file.readline())
                    score.append(p_score(gt, t1, t2, s1))
                    print(t2 / t1, t1 / gt, t2 / gt)
            score = np.array(score)
            average_p = score.mean()
            score[score > 0] = 1
            average_alotc = score.mean()
            table.add_row([genre, "{:.4f}".format(average_p), "{:.4f}".format(average_alotc)])

    print(table)
