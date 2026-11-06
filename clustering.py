"""Score independent baseline CU participants with a frozen ARC model.
Use the same baseline CSV/NPZ schema as SA_AD_classification.py, with dx=CU.
Optional long outcome CSV: cohort, participant_id, outcome, measured_date, value..
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from SA_AD_classification import keys, load_baseline


def score_cu(t, x, edges, artifact):
    if not np.array_equal(edges, artifact['edges']):
        raise ValueError('Ordered edge IDs differ from the training input')
    if not t.dx.eq('CU').all():
        raise ValueError('Supply only independently confirmed baseline CU participants')
    if keys(t) & artifact['anchor_keys']:
        raise ValueError('CU participants overlap discovery or external anchors')
    t = t.copy()
    t['arc_score'] = artifact['model'].decision_function(x)
    t['arc_group'] = np.where(t.arc_score > artifact['threshold'], 'AD-like', 'SA-like')
    return t


def longitudinal(scored, outcomes, out):
    import statsmodels.formula.api as smf
    import statsmodels.stats.multitest as mt
    if 'baseline_age' not in scored:
        raise ValueError('Longitudinal analyses require baseline_age in the baseline CSV')
    y = pd.read_csv(outcomes, dtype={'cohort': str, 'participant_id': str})
    for key in ['cohort', 'participant_id', 'outcome']:
        if y[key].isna().any() or y[key].str.strip().eq('').any():
            raise ValueError(f'Invalid outcome key: {key}')
        y[key] = y[key].str.strip()
    y['measured_date'] = pd.to_datetime(y.measured_date, format='%Y-%m-%d', errors='raise')
    if y.measured_date.isna().any() or y.duplicated(
            ['cohort', 'participant_id', 'outcome', 'measured_date']).any():
        raise ValueError('Missing dates or duplicate outcome visits')
    y['value'] = pd.to_numeric(y.value, errors='raise')
    k = ['cohort', 'participant_id']
    m = y.merge(scored[k+['scan_date', 'arc_score', 'baseline_age']],
                on=k, how='inner', validate='many_to_one')
    m['years'] = (m.measured_date-m.scan_date).dt.days/365.25
    m['baseline_age'] = pd.to_numeric(m.baseline_age, errors='raise')
    m = m[(m.years > 0) & np.isfinite(m.value) & np.isfinite(m.baseline_age)].copy()
    m.to_csv(out/'followup_used.csv', index=False)
    rows = []
    for (cohort, outcome), d in m.groupby(['cohort', 'outcome']):
        row = {'cohort': cohort, 'outcome': outcome,
               'participants': d.participant_id.nunique(), 'visits': len(d),
               'participants_with_repeat': int((d.groupby('participant_id').size() > 1).sum())}
        rows.append(row)
        if row['participants_with_repeat'] < 5 or d.arc_score.nunique() < 2:
            row['status'] = 'insufficient repeated observations or ARC variation'
            continue
        d = d.assign(age0=d.baseline_age-d.baseline_age.mean())
        term = 'arc_score:years'
        formula = 'value ~ arc_score * years + age0 * years'
        if d.age0.nunique() == 1:
            formula = 'value ~ arc_score * years'
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                fit = smf.mixedlm(formula, data=d, groups=d.participant_id).fit(
                    reml=False, method='powell', disp=False)
            lo, hi = fit.conf_int().loc[term]
            if not fit.converged or not np.isfinite([lo, hi, fit.pvalues[term]]).all():
                raise ValueError('nonconvergence or nonfinite interval')
            row.update(status='ok', beta=fit.params[term], low=lo, high=hi,
                       p=fit.pvalues[term], warnings=' | '.join(str(v.message) for v in w))
        except (ValueError, np.linalg.LinAlgError) as exc:
            row['status'] = f'fit failed: {exc}'
    r = pd.DataFrame(rows, columns=['cohort', 'outcome', 'participants', 'visits',
                                    'participants_with_repeat', 'status',
                                    'beta', 'low', 'high', 'p', 'warnings'])
    r['q_within_cohort'] = np.nan
    for _, g in r.groupby('cohort'):
        good = g.index[g.p.notna()]
        if len(good):
            r.loc[good, 'q_within_cohort'] = mt.multipletests(r.loc[good, 'p'], method='fdr_bh')[1]
    r.to_csv(out/'longitudinal_effects.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('baseline-csv', 'baseline-npz', 'model', 'out'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--outcomes')
    a = p.parse_args()
    t, x, edges = load_baseline(a.baseline_csv, a.baseline_npz)
    scored = score_cu(t, x, edges, joblib.load(a.model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out/'cu_baseline_scores.csv', index=False)
    if a.outcomes:
        longitudinal(scored, a.outcomes, out)


if __name__ == '__main__':
    main()
