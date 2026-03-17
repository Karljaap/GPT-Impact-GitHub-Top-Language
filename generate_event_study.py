# -*- coding: utf-8 -*-
"""
generate_event_study.py
-----------------------
Standalone script: runs SDID estimation only and generates the
event study (pre-trend validation) figures for all 10 languages.
Reads from the already-built balanced panel (output/data/data_langs_balanced.csv).
Much faster than running the full pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from synthdid.synthdid import Synthdid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(*parts):
    return os.path.join(BASE_DIR, *parts)


LANGUAGES_5 = ["C", "C#", "C++", "Go", "Java", "JavaScript",
               "PHP", "Python", "Ruby", "TypeScript"]

LANG_SAFE = {
    "C":          "C",
    "C#":         "C_hashtag",
    "C++":        "C_plus",
    "Go":         "Go",
    "Java":       "Java",
    "JavaScript": "JavaScript",
    "PHP":        "PHP",
    "Python":     "Python",
    "Ruby":       "Ruby",
    "TypeScript": "TypeScript",
}

TREAT_START = 12      # Q4-2022
N_BOOT      = 100     # for SDID SE (required by .vcov)
N_BOOT_ES   = 200     # for event study CIs


def _quarter_structure(df_lang):
    all_q    = sorted(df_lang['quarter'].unique())
    pre_q    = [q for q in all_q if q < TREAT_START]
    post_q   = [q for q in all_q if q >= TREAT_START]
    co_units = sorted(df_lang[df_lang['gpt_available'] == 0]['iso2_code'].unique())
    tr_units = sorted(df_lang[df_lang['gpt_available'] == 1]['iso2_code'].unique())
    return all_q, pre_q, post_q, co_units, tr_units


def _estimate_sdid(df_lang, outcome_col='num_pushers_pc'):
    df_s = df_lang[['iso2_code', 'quarter', outcome_col, 'gpt_available']].copy()
    df_s['treat'] = (
        (df_s['gpt_available'] == 1) & (df_s['quarter'] >= TREAT_START)
    ).astype(int)
    r = (
        Synthdid(df_s, 'iso2_code', 'quarter', 'treat', outcome_col)
        .fit()
        .vcov(method='bootstrap', n_reps=N_BOOT)
        .summary()
    )
    omega = np.array(r.weights['omega'][0])
    all_units = list(r.Y_units[0])
    co_units_ordered = all_units[:len(omega)]
    return omega, co_units_ordered


def _sdid_event_study(df_lang, omega, co_units, outcome_col='num_pushers_pc'):
    all_q, pre_q, _, _, tr_units = _quarter_structure(df_lang)
    T  = len(all_q)
    T0 = len(pre_q)

    def _piv(units):
        return (
            df_lang[df_lang['iso2_code'].isin(units)]
            .pivot(index='iso2_code', columns='quarter', values=outcome_col)
            .reindex(index=list(units), columns=all_q)
            .fillna(0).values
        )

    Y_co = _piv(co_units)
    Y_tr = _piv(tr_units)
    synthetic = omega @ Y_co

    def _gap(Y_tr_):
        g = Y_tr_.mean(axis=0) - synthetic
        g = g - g[:T0].mean()
        return g

    gap_est  = _gap(Y_tr)
    rng      = np.random.default_rng(42)
    boot_mat = np.zeros((N_BOOT_ES, T))
    for b in range(N_BOOT_ES):
        idx = rng.integers(0, len(tr_units), size=len(tr_units))
        boot_mat[b] = _gap(Y_tr[idx])

    lower = np.percentile(boot_mat, 2.5,  axis=0)
    upper = np.percentile(boot_mat, 97.5, axis=0)
    rel   = [q - TREAT_START for q in all_q]
    return rel, gap_est, lower, upper


def _plot_event_study(lang, rel, gap, lower, upper):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.array(rel)
    ax.fill_between(x, lower, upper, alpha=0.20, color='steelblue', label='IC 95%')
    ax.plot(x, gap, marker='o', color='steelblue', lw=1.8, ms=5, label='Gap SDID')
    ax.axhline(0,    color='black', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(-0.5, color='red',   lw=1.2, ls='--', label='Inicio trat. (Q4-2022)')
    if any(r < 0 for r in rel):
        ax.axvspan(min(x) - 0.5, -0.5, alpha=0.06, color='grey')
    ax.set_xticks(x)
    xlabels = [str(r) if i % 2 == 0 else '' for i, r in enumerate(rel)]
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel('Trimestres relativos al tratamiento (0 = Q4-2022)', fontsize=10)
    ax.set_ylabel('Gap (Tratado \u2212 Control Sint\u00e9tico)', fontsize=10)
    ax.set_title(f'{lang} \u2014 An\u00e1lisis de evento SDID (pre-tendencias)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    fname = p("output", "figures", f"{LANG_SAFE[lang]}sdid_event_study.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname


# ── main ──────────────────────────────────────────────────────────────────────

panel_path = p("output", "data", "data_langs_balanced.csv")
if not os.path.isfile(panel_path):
    print(f"ERROR: {panel_path} not found. Run the full pipeline first.")
    sys.exit(1)

df_panel = pd.read_csv(panel_path)
df_panel = df_panel[df_panel['iso2_code'] != 'HK'].copy()

print("=" * 60)
print("Generating SDID event study figures")
print("=" * 60)

for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l = df_panel[df_panel['language'] == lang].copy()
    try:
        omega, co_units = _estimate_sdid(df_l)
        rel, gap, lo, hi = _sdid_event_study(df_l, omega, co_units)
        _plot_event_study(lang, rel, gap, lo, hi)
    except Exception as exc:
        print(f"  FAILED: {exc}")

print("\n" + "=" * 60)
print("Done. Now run: pdflatex Tesis.tex")
print("=" * 60)
