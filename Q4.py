from librosa.core import load
from librosa.util import normalize
import numpy as np
import os
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils import data_dir, label_dir, genres, p_score, stacf, spectral_flux, genres_dir, tempo_estimation


def tempogram_t(file_name, window1=2048, window2=512, lw_sr=100, g=1, mean_size=25):
    data, sr = load(file_name, sr=None)
    hop_size = sr // lw_sr
    t, nv_curve = spectral_flux(data, sr, hop_size, window1, g, mean_size, lag=1)
    offset = t[0]

    lag, t2, tpg = stacf(nv_curve, lw_sr, None, window2)
    tpg = normalize(tpg, axis=0)
    t2 += offset
    freq_scale = 60 / lag[1:]
    freq_scale = np.concatenate(([0], freq_scale))
    return freq_scale, t2, tpg


if __name__ == '__main__':
    table = PrettyTable(["Genre", "Average P-score", "Average ALOTC"])
    ratio_list = []
    for genre in genres:
        score = []
        for g_dir in genres_dir:
            if genre in g_dir and genre[:3] in g_dir[:3]:
                print("Running", g_dir, "...")
                dir = os.path.join(data_dir, g_dir)
                files = os.listdir(dir)
                for file_name in files:
                    f, t, tpg = tempogram_t(os.path.join(dir, file_name))
                    ((t1v, t1), (t2v, t2)) = tempo_estimation(f, tpg)
                    s1 = t1v / (t1v + t2v)

                    with open(os.path.join(label_dir, file_name.replace('.wav', '.bpm').split('/')[-1])) as label_file:
                        gt = int(label_file.readline())
                        score.append(p_score(gt, t1, t2, s1))
                        #print(t2 / t1, t1 / gt, t2 / gt)
        score = np.array(score)
        average_p = score.mean()
        score[score > 0] = 1
        average_alotc = score.mean()
        table.add_row([genre, "{:.4f}".format(average_p), "{:.4f}".format(average_alotc)])

    print(table)
