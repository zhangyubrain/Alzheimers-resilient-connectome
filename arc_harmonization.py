"""Fold-local, label-blind ComBat with a separately reserved ADNI calibration set."""
import os
import numpy as np
import pandas as pd


def clean_ids(values):
    return np.asarray([str(x).strip() for x in values])


def load_adni_calibration(evaluation_ids, n_edges):
    # The calibration matrix must use the same FC edge order and per-scan
    # normalization as the discovery arrays. One row per reserved participant.
    with np.load(os.environ["ARC_ADNI_CALIBRATION"], allow_pickle=False) as data:
        x = np.asarray(data["fc"], float)
        ids = clean_ids(data["participant_id"])
    if (x.ndim != 2 or x.shape != (len(ids), n_edges)
            or len(ids) < 3 or len(set(ids)) != len(ids)
            or "" in ids or not np.isfinite(x).all()):
        raise ValueError("Invalid ADNI calibration FC/IDs")
    if set(ids) & set(clean_ids(evaluation_ids)):
        raise ValueError("ADNI calibration and evaluation participants overlap")
    return x


class FoldCombat:
    """Estimate location/scale on training participants; apply frozen transforms.

    Site-only model: no diagnosis, demographics, or outcomes are passed to
    ComBat. Keep interpretation conditional on cohort confounding.
    """
    def fit(self, fc, sites):
        from neuroCombat import neuroCombat
        x = np.asarray(fc, float)
        sites = np.asarray(sites)
        if (x.ndim != 2 or len(x) != len(sites) or not np.isfinite(x).all()
                or np.any(np.unique(sites, return_counts=True)[1] < 3)):
            raise ValueError("ComBat requires finite FC and >=3 training people per batch")
        e = neuroCombat(dat=x.T, covars=pd.DataFrame({"site": sites}),
                        batch_col="site", categorical_cols=[], continuous_cols=[])["estimates"]
        self.index = {float(b): i for i, b in enumerate(e["batches"])}
        self.center = np.asarray(e["stand.mean"])[:, 0]
        self.scale = np.sqrt(np.asarray(e["var.pooled"]).reshape(-1))
        self.shift = np.asarray(e["gamma.star"])
        self.spread = np.sqrt(np.asarray(e["delta.star"]))
        if (not all(np.isfinite(a).all() for a in (self.center, self.scale, self.shift, self.spread))
                or np.any(self.scale <= 0) or np.any(self.spread <= 0)):
            raise ValueError("ComBat produced invalid estimates; inspect constant edges and batches")
        return self

    def transform(self, fc, sites):
        x = np.asarray(fc, float)
        if x.ndim != 2 or x.shape != (len(sites), len(self.center)) or not np.isfinite(x).all():
            raise ValueError("Invalid FC for frozen ComBat")
        try:
            j = np.array([self.index[float(s)] for s in sites], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Batch was not fitted: {exc}") from exc
        return (((x - self.center) / self.scale - self.shift[j]) /
                self.spread[j]) * self.scale + self.center


def participant_scores(ids, labels, scores):
    """Average held-out decision margins, not predicted class labels."""
    ids = clean_ids(ids)
    labels, scores = np.asarray(labels).ravel(), np.asarray(scores).ravel()
    if not (len(ids) == len(labels) == len(scores)) or not np.isfinite(scores).all():
        raise ValueError("Misaligned participant scoring inputs")
    unique, inverse = np.unique(ids, return_inverse=True)
    y = np.empty(len(unique), dtype=labels.dtype)
    for j in range(len(unique)):
        values = np.unique(labels[inverse == j])
        if len(values) != 1:
            raise ValueError(f"Conflicting test labels for participant {unique[j]}")
        y[j] = values[0]
    return y, np.bincount(inverse, weights=scores) / np.bincount(inverse)
