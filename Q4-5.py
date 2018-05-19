from librosa.core import load
from librosa.util import normalize
import numpy as np
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils import data_dir, bpm_label_dir, genres, p_score, stacf, spectral_flux, genres_dir, tempo_estimation


def tempogram_t(nv_curve, sr, window_size=512, hop_size=None):
    lag, t2, tpg = stacf(nv_curve, sr, window_size, hop_size=hop_size)
    tpg = normalize(tpg)
    freq_scale = 60 / lag[1:]
    freq_scale = np.concatenate(([0], freq_scale))
    return freq_scale, tpg


if __name__ == '__main__':
    table = PrettyTable(
        ["Genre", "P-score", "ALOTC", "1/2T P-score", "1/3T P-score", "1/4T P-score", "2T P-score", "3T P-score",
         "4T P-score"])
    ratio_list = []

    window_size = 2048
    lw_sr = 100
    g = 1
    mean_size = 25
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
                    f, tpg = tempogram_t(nv_curve, lw_sr)
                    t1, t2, s1 = tempo_estimation(f, tpg)

                    with open(os.path.join(bpm_label_dir,
                                           file_name.replace('.wav', '.bpm').split('/')[-1])) as label_file:
                        truth = int(label_file.readline())
                        score.append([])
                        score[-1].append(p_score(truth, t1, t2, s1))
                        score[-1].append(p_score(truth, t1 / 2, t2 / 2, s1))
                        score[-1].append(p_score(truth, t1 / 3, t2 / 3, s1))
                        score[-1].append(p_score(truth, t1 / 4, t2 / 4, s1))
                        score[-1].append(p_score(truth, t1 * 2, t2 * 2, s1))
                        score[-1].append(p_score(truth, t1 * 3, t2 * 3, s1))
                        score[-1].append(p_score(truth, t1 * 4, t2 * 4, s1))
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

        '''
        fig, ax = plt.subplots(1, 3, sharey='all')
        ax[0].hist(ratio[0])
        ax[0].set_title('T2/T1')
        ax[1].hist(ratio[1])
        ax[1].set_title('T1/G')
        ax[2].hist(ratio[2])
        ax[2].set_title('T2/G')
        fig.suptitle(genre)
        plt.show()
        '''
    print(table)
