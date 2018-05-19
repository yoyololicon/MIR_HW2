from librosa.core import load
from librosa.util import normalize
from scipy.signal import argrelmax
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from utils import data_dir, beat_label_dir, genres, spectral_flux, genres_dir, stacf


def penalty_func(d, d_):
    return -math.log(d / d_, 2) ** 2


vpenalty_func = np.vectorize(penalty_func)


def tempo_estimation(nv_curve, sr, window_size):
    lag, t2, tpg = stacf(nv_curve, sr, window_size, hop_size=None)
    tpg = normalize(tpg, axis=0).sum(axis=1)
    mask_tpg = np.zeros(tpg.shape)
    idx = argrelmax(tpg)
    mask_tpg[idx] = tpg[idx]
    delta = np.argmax(mask_tpg)
    return delta


def optimal_beats_sequence(nv_curve, d_, ld):
    D = np.zeros(len(nv_curve) + 1)
    P = np.zeros(len(nv_curve) + 1, dtype=np.int32)
    penalty = vpenalty_func(np.arange(len(nv_curve) - 1, 0, -1), d_) * ld

    D[1] = nv_curve[0]
    for i in range(2, len(nv_curve) + 1):
        D[i] = nv_curve[i - 1]
        obj = D[1:i] + penalty[-i + 1:]
        maxterm = max(0, np.max(obj))
        if maxterm > 0:
            P[i] = np.argmax(obj) + 1
            D[i] += maxterm
        else:
            P[i] = 0

    al = np.argmax(D)
    seq = []
    while P[al] != 0:
        seq.append(al)
        al = P[al]
    if al == 0:
        return []
    else:
        return np.array(seq[::-1]) - 1


def evaluate(label_seq, pred_seq):
    tol = 0.07
    p_pool = np.zeros(pred_seq.shape)
    r_pool = np.zeros(label_seq.shape)

    for i in range(len(label_seq)):
        p_idx = np.where((pred_seq > label_seq[i] - tol) & (pred_seq < label_seq[i] + tol))[0]
        p_beats = pred_seq[p_idx]
        if len(p_beats) > 1:
            dist = np.abs(p_beats - label_seq[i])
            best_idx = p_idx[np.argmax(dist)]
        elif len(p_beats) == 1:
            best_idx = p_idx[0]
        else:
            continue

        if p_pool[best_idx] > 0:
            continue
        else:
            p_pool[best_idx] = 1
            r_pool[i] = 1
    tp = np.count_nonzero(p_pool)
    fp = len(p_pool) - tp
    fn = len(r_pool) - tp
    return tp, fp, fn


if __name__ == '__main__':
    table = PrettyTable(["Genre", "Precision", "Recall", "F-scores"])
    lw_sr = 100
    ld = 10

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
                    t, nv_curve = spectral_flux(data, sr, hop_size, 1024, 1, 25, lag=1)
                    delta = tempo_estimation(nv_curve, lw_sr, 512)
                    beats = optimal_beats_sequence(nv_curve, delta, ld)
                    label = np.loadtxt(os.path.join(beat_label_dir, file_name.replace('.wav', '.beats').split('/')[-1]))
                    score.append(evaluate(label[:, 0], t[beats]))

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
