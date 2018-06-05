from librosa.core import load, autocorrelate
import numpy as np
import os
from prettytable import PrettyTable
from scipy.signal import fftconvolve, argrelmax
from utils import data_dir, beat_label_dir, genres, spectral_flux_with_stft, genres_dir, tempo_estimation
from Q6 import CFP
from Q7 import optimal_beats_sequence, evaluate

if __name__ == '__main__':
    table = PrettyTable(["Genre", "Precision", "Recall", "F-scores"])
    lw_sr = 100
    ld = 20

    for genre in genres:
        score = []
        for g_dir in genres_dir:
            if genre in g_dir and genre[0] == g_dir[0]:
                print("Running", g_dir, "...")
                dir = os.path.join(data_dir, g_dir)
                files = os.listdir(dir)
                for file_name in files:
                    data, sr = load(os.path.join(dir, file_name), sr=None)
                    hop_size = sr // lw_sr
                    t, nv_curve, spec = spectral_flux_with_stft(data, sr, hop_size, 1024, 1, 25, lag=1)
                    f, tpg = CFP(nv_curve, lw_sr, 2000, 512, 50)
                    t1, t2, s1 = tempo_estimation(f, tpg)
                    if s1 > 0.5:
                        delta = t1
                    else:
                        delta = t2
                    delta = 60 / delta * lw_sr
                    beats = optimal_beats_sequence(nv_curve, delta, ld)

                    spec = spec[:, beats]
                    t = t[beats]
                    beat_nv_curve = np.maximum(0., spec[:, 1:] - spec[:, :-1]).mean(axis=0)
                    u = fftconvolve(beat_nv_curve, np.array([0.3, 0.4, 0.3]), 'same')
                    downbeats = argrelmax(u)[0] + 1

                    label = np.loadtxt(os.path.join(beat_label_dir, file_name.replace('.wav', '.beats').split('/')[-1]))
                    label = np.squeeze(label[np.where(label[:, 1] == 1.), 0])
                    score.append(evaluate(label, t[downbeats]))

        score = np.array(score).sum(axis=0)
        p = score[0] / (score[0] + score[1])
        r = score[0] / (score[0] + score[2])
        f = 2 * p * r / (p + r)
        table_row = [genre]
        table_row.append("{:.4f}".format(p))
        table_row.append("{:.4f}".format(r))
        table_row.append("{:.4f}".format(f))
        table.add_row(table_row)

    print(table)
