import numpy as np

def mca(gt_label, pred_label, n_eval_cls):
    n_hit       = np.zeros(n_eval_cls)
    n_total     = np.zeros(n_eval_cls)
    for i, j in zip(gt_label, pred_label):
        if i == j:
            n_hit[i] += 1
        n_total[i] += 1
    _mca = n_hit / (n_total + 1e-15)
    return np.mean(_mca)

def harmonic_mean(gt_label, pred_label, n_eval_cls, n_tr):
    n_hit       = np.zeros(n_eval_cls)
    n_total     = np.zeros(n_eval_cls)
    for i, j in zip(gt_label, pred_label):
        if i == j:
            n_hit[i] += 1
        n_total[i] += 1 
    _mca = n_hit / (n_total + 1e-15)
    tr = _mca[:n_tr].mean()
    ts = _mca[n_tr:].mean()
    H  = 2 * tr * ts / (tr + ts + 1e-16)
    return tr, ts, H
