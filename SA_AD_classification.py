# -*- coding: utf-8 -*-

"""Train and freeze ARC. See --help. Inputs are one row per baseline participant.
NPZ arrays: X (Fisher-z FC, rows in CSV order), row_id, edge_id (ordered names).
CSV: row_id, cohort, participant_id, scan_date; anchors also require dx=SA/AD.
Use verified, date-matched baseline diagnoses and the same atlas/edge convention.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def load_baseline(csv, npz):
    t = pd.read_csv(csv, dtype=str, keep_default_na=False)
    fields = ['row_id', 'cohort', 'participant_id', 'scan_date']
    for key in fields:
        t[key] = t[key].str.strip()
        if t[key].eq('').any():
            raise ValueError(f'Empty {key}')
    t['scan_date'] = pd.to_datetime(t.scan_date, format='%Y-%m-%d', errors='raise')
    if t.scan_date.isna().any():
        raise ValueError("Missing baseline scan date")
    if t.row_id.duplicated().any() or t.duplicated(['cohort', 'participant_id']).any():
        raise ValueError('Provide exactly one verified baseline FC per participant')
    with np.load(npz, allow_pickle=False) as a:
        x = np.asarray(a['X'], dtype=float)
        edge = a['edge_id'].astype(str)
        rows = a['row_id'].astype(str)
    if not np.array_equal(rows, t.row_id.to_numpy()):
        raise ValueError('FC and metadata row IDs do not agree in order')
    if (edge.ndim != 1 or x.shape != (len(t), len(edge))
            or len(edge) < 2 or len(set(edge)) != len(edge)
            or any(not e.strip() for e in edge) or not np.isfinite(x).all()):
        raise ValueError('Invalid feature matrix or edge IDs')
    return t, x, edge


def keys(t):
    return set(zip(t.cohort, t.participant_id))


def anchor_labels(t):
    if not t.dx.isin(['SA', 'AD']).all():
        raise ValueError('Anchor dx must be SA or AD')
    return t.dx.eq('AD').to_numpy(dtype=int)


def metrics(y, score):
    return {'n': len(y), 'auc': float(roc_auc_score(y, score)),
            'balanced_accuracy': float(balanced_accuracy_score(y, score > 0))}


def tune(x, y, seed, folds):
    n = min(folds, int(np.bincount(y, minlength=2).min()))
    if n < 2:
        raise ValueError('Each class needs at least two training participants')
    pipe = Pipeline([('scale', StandardScaler()),
                     ('edges', SelectKBest(f_classif)),
                     ('svm', LinearSVC(penalty='l1', dual=False, max_iter=20000,
                                       random_state=seed))])
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    grid = {'edges__k': sorted({min(k, x.shape[1]) for k in [50, 200, 500]}),
            'svm__C': [.01, .1, .7, 1.], 'svm__class_weight': [None, 'balanced']}
    return GridSearchCV(pipe, grid, scoring='balanced_accuracy', cv=cv,
                        error_score='raise').fit(x, y)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('discovery-csv', 'discovery-npz', 'external-csv', 'external-npz', 'out'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--outer', type=int, default=10)
    p.add_argument('--inner', type=int, default=5)
    p.add_argument('--seed', type=int, default=9)
    a = p.parse_args()
    t, x, edges = load_baseline(a.discovery_csv, a.discovery_npz)
    u, z, ext_edges = load_baseline(a.external_csv, a.external_npz)
    if (keys(t) & keys(u) or set(t.cohort) & set(u.cohort)
            or not np.array_equal(edges, ext_edges)):
        raise ValueError('External cohort must be separate and have identical ordered edges')
    y = anchor_labels(t)
    if not 2 <= a.outer <= np.bincount(y, minlength=2).min() or a.inner < 2:
        raise ValueError('Invalid fold counts for class sizes')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fold_stats = []
    t['fold'], t['arc_score'] = -1, np.nan
    outer = StratifiedKFold(n_splits=a.outer, shuffle=True, random_state=a.seed)
    for i, (tr, te) in enumerate(outer.split(x, y)):
        if keys(t.iloc[tr]) & keys(t.iloc[te]):
            raise RuntimeError('Participant overlap')
        fit = tune(x[tr], y[tr], a.seed+i, a.inner)
        score = fit.decision_function(x[te])
        t.loc[te, ['fold', 'arc_score']] = np.column_stack([np.full(len(te), i), score])
        fold_stats.append({'fold': i, **metrics(y[te], score), 'params': fit.best_params_})
    frozen = tune(x, y, a.seed, a.inner)
    artifact = {'model': frozen.best_estimator_, 'edges': edges, 'threshold': 0.,
                'anchor_keys': keys(t) | keys(u), 'params': frozen.best_params_}
    joblib.dump(artifact, out/'arc_model.joblib')
    u['arc_score'] = frozen.decision_function(z)
    u['predicted_dx'] = np.where(u.arc_score > 0, 'AD', 'SA')
    t.to_csv(out/'discovery_oof.csv', index=False)
    u.to_csv(out/'external_predictions.csv', index=False)
    report = {'discovery_folds': fold_stats, 'frozen_params': frozen.best_params_,
              'mean_fold_auc': float(np.mean([f['auc'] for f in fold_stats])),
              'external': metrics(anchor_labels(u), u.arc_score)}
    (out/'metrics.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

