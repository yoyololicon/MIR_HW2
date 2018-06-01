from librosa.core import load, stft
from librosa.util import normalize
import numpy as np
from utils import spectral_flux, p_score, genres, data_dir, bpm_label_dir, genres_dir, tempo_estimation, \
    harmonic_sum_tempogram
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def tempogram_f(nv_curve, sr, window_size=512, hop_size=128):
    tpg = normalize(np.abs(stft(nv_curve, n_fft=window_size, hop_length=hop_size)))
    freq_scale = np.arange(window_size // 2 + 1) * sr / window_size * 60
    return freq_scale, tpg


if __name__ == '__main__':
    table = PrettyTable(["Genre", "P-score", "ALOTC", "1/2T P-score", "1/3T P-score", "1/4T P-score"])
    ratio_list = []

    window_size = 1024
    lw_sr = 100
    g = 1
    mean_size = 25
    visualize = False
    for genre in genres:
        score = []
        ratio = []
        for g_dir in genres_dir:
            if genre in g_dir and genre[:3] in g_dir[:3]:
                print("Running", g_dir, "...")
                dir = os.path.join(data_dir, g_dir)
                files = os.listdir(dir)
                for file_name in files:
                    data, sr = load(os.path.join(dir, file_name), sr=None)
                    hop_size = sr // lw_sr
                    t, nv_curve = spectral_flux(data, sr, hop_size, window_size, g, mean_size, lag=1)
                    f, tpg = tempogram_f(nv_curve, lw_sr, window_size=2000, hop_size=50)
                    f, tpg = harmonic_sum_tempogram(f, tpg, 4, 1.)
                    t1, t2, s1 = tempo_estimation(f, tpg)

                    with open(os.path.join(bpm_label_dir,
                                           file_name.replace('.wav', '.bpm').split('/')[-1])) as label_file:
                        truth = int(label_file.readline())
                        score.append([])
                        score[-1].append(p_score(truth, t1, t2, s1))
                        score[-1].append(p_score(truth, t1 / 2, t2 / 2, s1))
                        score[-1].append(p_score(truth, t1 / 3, t2 / 3, s1))
                        score[-1].append(p_score(truth, t1 / 4, t2 / 4, s1))
                        ratio.append((t2 / t1, t1 / truth, t2 / truth))
        score = np.array(score)
        ratio = np.array(ratio).T
        average_p = score.mean(axis=0)
        score[score > 0] = 1
        average_alotc = score.mean(axis=0)
        out = []
        out.append(average_p[0])
        out.append(average_alotc[0])
        out += average_p[1:].tolist()
        table_row = [genre]
        for p in out:
            table_row.append("{:.4f}".format(p))
        table.add_row(table_row)

        if visualize:
            fig, ax = plt.subplots(1, 3, sharey='row')
            fig.set_size_inches(8, 6)
            ax[0].hist(ratio[0])
            ax[0].set_title('T2/T1')
            ax[1].hist(ratio[1])
            ax[1].set_title('T1/G')
            ax[2].hist(ratio[2])
            ax[2].set_title('T2/G')
            fig.suptitle(genre)
            plt.show()

    print(table)
