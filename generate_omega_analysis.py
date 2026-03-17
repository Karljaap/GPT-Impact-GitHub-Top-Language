# -*- coding: utf-8 -*-
"""
generate_omega_analysis.py
--------------------------
Extracts SDID omega weights per control country for each language.
Key question: do the "hardline censors" (CN, RU, IR, CU, BY, etc.)
receive near-zero omega weights? If so, SDID self-corrects for
the non-random control group composition.

Output: output/tables/omega_weights_table.tex
        output/data/omega_weights.csv
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

LANG_TEX = {"C": "C", "C#": r"C\#", "C++": "C++", "Go": "Go", "Java": "Java",
            "JavaScript": "JavaScript", "PHP": "PHP", "Python": "Python",
            "Ruby": "Ruby", "TypeScript": "TypeScript"}

TREAT_START = 12
N_BOOT = 100

# Countries with extreme internet censorship (Freedom on the Net 2022 score < 25)
HARDLINE = {"CN", "IR", "SY", "BY", "RU", "CU", "TJ", "VN", "TM"}

def _stars(att, se):
    if np.isnan(se) or se <= 0: return ''
    from scipy.stats import norm
    pv = 2.0 * (1.0 - norm.cdf(abs(att / se)))
    if pv < 0.01: return '***'
    if pv < 0.05: return '**'
    if pv < 0.10: return '*'
    return ''

# ── load panel ────────────────────────────────────────────────────────────────
df_panel = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))
df_panel = df_panel[df_panel['iso2_code'] != 'HK'].copy()

print("=" * 60)
print("Extracting SDID omega weights per control country")
print("=" * 60)

all_weights = {}  # {lang: {country: weight}}

for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l = df_panel[df_panel['language'] == lang].copy()
    df_s = df_l[['iso2_code', 'quarter', 'num_pushers_pc', 'gpt_available']].copy()
    df_s['treat'] = ((df_s['gpt_available'] == 1) & (df_s['quarter'] >= TREAT_START)).astype(int)

    try:
        r = (Synthdid(df_s, 'iso2_code', 'quarter', 'treat', 'num_pushers_pc')
             .fit()
             .vcov(method='bootstrap', n_reps=N_BOOT)
             .summary())
        omega = np.array(r.weights['omega'][0])
        all_units = list(r.Y_units[0])
        co_units = all_units[:len(omega)]
        weights_dict = dict(zip(co_units, omega))
        all_weights[lang] = weights_dict
        # Show hardline weights
        for c in sorted(HARDLINE):
            w = weights_dict.get(c, 0.0)
            if w > 0.001:
                print(f"    {c}: omega = {w:.4f}  *** non-zero ***")
            else:
                print(f"    {c}: omega = {w:.4f}")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        all_weights[lang] = {}

# ── Save CSV ──────────────────────────────────────────────────────────────────
# Get all control countries
all_co = sorted(set(c for wd in all_weights.values() for c in wd))
rows = []
for co in all_co:
    row = {'iso2_code': co, 'is_hardline': co in HARDLINE}
    for lang in LANGUAGES_5:
        row[lang] = all_weights.get(lang, {}).get(co, 0.0)
    row['avg_weight'] = np.mean([row[l] for l in LANGUAGES_5])
    rows.append(row)
df_omega = pd.DataFrame(rows).sort_values('avg_weight', ascending=False)
df_omega.to_csv(p("output", "data", "omega_weights.csv"), index=False)
print(f"\n  Saved: output/data/omega_weights.csv")

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n  Average omega weights by group:")
hardline_avg = df_omega[df_omega['is_hardline']]['avg_weight'].mean()
other_avg    = df_omega[~df_omega['is_hardline']]['avg_weight'].mean()
print(f"    Hardline censors (N=9):    {hardline_avg:.5f}")
print(f"    Other control (N=20):      {other_avg:.5f}")
print(f"    Ratio (hardline/other):    {hardline_avg/other_avg:.3f}")

# ── LaTeX table: omega weights for hardline countries across languages ─────────
print("\n  Writing omega_weights_table.tex ...")

hardline_rows = df_omega[df_omega['is_hardline']].sort_values('avg_weight', ascending=False)

lines = [
    r"\begin{table}[H]\centering",
    r"\caption{Pesos $\omega$ SDID para países con restricciones extremas de internet}",
    r"\label{tab:omega_weights}",
    r"\begin{threeparttable}",
    r"{\small",
    r"\begin{tabular}{lccccccccccr}",
    r"\toprule",
    r"País & C & C\# & C++ & Go & Java & JS & PHP & Py & Ruby & TS & Promedio \\",
    r"\midrule",
]

country_names = {
    'CN': 'China', 'RU': 'Rusia', 'IR': 'Irán', 'CU': 'Cuba',
    'BY': 'Bielorrusia', 'SY': 'Siria', 'TJ': 'Tayikistán',
    'VN': 'Vietnam', 'TM': 'Turkmenistán'
}

for _, row in hardline_rows.iterrows():
    co   = row['iso2_code']
    name = country_names.get(co, co)
    vals = [f"{row[l]:.3f}" for l in LANGUAGES_5]
    avg  = f"{row['avg_weight']:.3f}"
    lines.append(f"{name} & " + " & ".join(vals) + f" & {avg} \\\\")

# Add separator and average non-hardline
avg_other_by_lang = {l: df_omega[~df_omega['is_hardline']][l].mean() for l in LANGUAGES_5}
other_vals = [f"{avg_other_by_lang[l]:.3f}" for l in LANGUAGES_5]
other_avg_str = f"{other_avg:.3f}"
lines += [
    r"\midrule",
    r"\textit{Resto de controles (N=20)} & " + " & ".join(other_vals) + f" & {other_avg_str} \\\\",
    r"\bottomrule",
    r"\end{tabular}}",
    r"\begin{tablenotes}",
    r"\footnotesize",
    r"\item \textit{Nota.} Pesos $\omega$ asignados por el estimador SDID a cada país del grupo de control. "
    r"Los países listados fueron clasificados como \textit{Not Free} con puntaje inferior a 25 en el índice "
    r"\textit{Freedom on the Net} 2022 (Freedom House). Un peso cercano a cero implica que el país no "
    r"contribuye a la construcción del contrafactual sintético. JS = JavaScript, Py = Python, TS = TypeScript.",
    r"\end{tablenotes}",
    r"\end{threeparttable}",
    r"\end{table}",
    "",
]

outpath = p("output", "tables", "omega_weights_table.tex")
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"  Saved: {outpath}")

# ── Heatmap figure ────────────────────────────────────────────────────────────
print("\n  Generating omega heatmap ...")
all_co_sorted = df_omega.sort_values('avg_weight', ascending=False)['iso2_code'].tolist()
mat = np.array([[all_weights.get(lang, {}).get(co, 0.0) for lang in LANGUAGES_5]
                for co in all_co_sorted])

fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=mat.max())
ax.set_xticks(range(len(LANGUAGES_5)))
ax.set_xticklabels(LANGUAGES_5, fontsize=9)
ax.set_yticks(range(len(all_co_sorted)))
yticklabels = [f"{'* ' if co in HARDLINE else '  '}{co}" for co in all_co_sorted]
ax.set_yticklabels(yticklabels, fontsize=7)
ax.set_title('Pesos $\\omega$ SDID por país y lenguaje\n(* = censor severo, Freedom on the Net < 25)',
             fontsize=11)
plt.colorbar(im, ax=ax, label='Peso $\\omega$')
plt.tight_layout()
fname = p("output", "figures", "omega_weights_heatmap.png")
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fname}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
