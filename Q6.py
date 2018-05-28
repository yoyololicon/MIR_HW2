from librosa.core import load, stft
from librosa.util import normalize
import numpy as np
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils import data_dir, bpm_label_dir, genres, p_score, stacf, spectral_flux, genres_dir, tempo_estimation, \
    harmonics_sum_tempogram


def CFP(nv_curve, sr, wsize_f, wsize_t, hop_f, hop_t=None):
    tpg_f = np.abs(stft(nv_curve, n_fft=wsize_f, hop_length=hop_f))
    freq_f = np.arange(wsize_f // 2 + 1) * sr / wsize_f * 60
    freq_f, tpg_f = harmonics_sum_tempogram(freq_f, tpg_f)
    tpg_f = normalize(tpg_f)

    lag, _, tpg = stacf(nv_curve, sr, wsize_t, hop_size=hop_t)
    tpg_t = normalize(tpg)
    freq_t = 60 / lag[1:]
    freq_t = np.concatenate(([0], freq_t))

    tpg_f = tpg_f.mean(axis=1)
    tpg_t = tpg_t.mean(axis=1)

    pool_scale = (freq_f[1:] + freq_f[:-1]) / 2
    transformed_tpg_t = np.zeros(len(pool_scale) - 1)
    for i in range(len(pool_scale) - 1):
        f1, f2 = pool_scale[i], pool_scale[i + 1]
        p_idx = (freq_t > f1) & (freq_t < f2)
        p_value = freq_t[p_idx]
        if len(p_value) > 0:
            transformed_tpg_t[i] = np.mean(tpg_t[p_idx])

    cfp = tpg_f[1:-1] * transformed_tpg_t
    return freq_f[1:-1], cfp[:, None]


if __name__ == '__main__':
    table = PrettyTable(["Genre", "P-score", "ALOTC"])
    ratio_list = []

    window_size = 1024
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
                    f, tpg = CFP(nv_curve, lw_sr, wsize_f=1000, wsize_t=512, hop_f=50)
                    t1, t2, s1 = tempo_estimation(f, tpg)

                    with open(os.path.join(bpm_label_dir,
                                           file_name.replace('.wav', '.bpm').split('/')[-1])) as label_file:
                        truth = int(label_file.readline())
                        # print(truth, t1, t2, s1)
                        score.append(p_score(truth, t1, t2, s1))
                        ratio.append((t2 / t1, t1 / truth, t2 / truth))
        score = np.array(score)
        ratio = np.array(ratio).T
        average_p = score.mean()
        score[score > 0] = 1
        average_alotc = score.mean()

        table.add_row([genre, "{:.4f}".format(average_p), "{:.4f}".format(average_alotc)])

        '''
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
        '''

    print(table)
