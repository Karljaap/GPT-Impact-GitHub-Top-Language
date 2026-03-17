# -*- coding: utf-8 -*-
"""
generate_robustness_restricted.py
----------------------------------
Re-runs SDID excluding the 9 "hardline censors" from the control group.
These are countries classified as "Not Free" with Freedom on the Net 2022
score < 25 (Freedom House): CN, IR, SY, BY, RU, CU, TJ, VN, TM.

If results are similar to the main specification, it confirms that the
estimates are not driven by the choice of these structurally dissimilar
control units.

Output: output/tables/robustness_restricted_control.tex
"""

import os
import sys
import numpy as np
import pandas as pd
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

# Countries to exclude from control group (Freedom on the Net 2022 score < 25)
HARDLINE = {"CN", "IR", "SY", "BY", "RU", "CU", "TJ", "VN", "TM"}

# Main SDID results (no controls, for comparison)
MAIN_SDID = {
    "C":          (1.276, 0.459),
    "C#":         (0.443, 0.464),
    "C++":        (1.131, 0.293),
    "Go":         (0.720, 0.247),
    "Java":       (1.725, 0.835),
    "JavaScript": (8.303, 2.974),
    "PHP":        (0.916, 0.422),
    "Python":     (4.525, 1.057),
    "Ruby":       (0.808, 0.282),
    "TypeScript": (3.078, 0.833),
}

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

# Remove hardline censors from control group
df_restricted = df_panel[~((df_panel['gpt_available'] == 0) &
                            (df_panel['iso2_code'].isin(HARDLINE)))].copy()

n_control_main = df_panel[df_panel['gpt_available'] == 0]['iso2_code'].nunique()
n_control_rest = df_restricted[df_restricted['gpt_available'] == 0]['iso2_code'].nunique()
print(f"Control group: {n_control_main} -> {n_control_rest} countries after removing hardline censors")
print(f"Removed: {sorted(HARDLINE)}")

print("=" * 60)
print("SDID with restricted control group")
print("=" * 60)

results = {}
for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l = df_restricted[df_restricted['language'] == lang].copy()
    df_s = df_l[['iso2_code', 'quarter', 'num_pushers_pc', 'gpt_available']].copy()
    df_s['treat'] = ((df_s['gpt_available'] == 1) & (df_s['quarter'] >= TREAT_START)).astype(int)
    try:
        r = (Synthdid(df_s, 'iso2_code', 'quarter', 'treat', 'num_pushers_pc')
             .fit()
             .vcov(method='bootstrap', n_reps=N_BOOT)
             .summary())
        att = float(r.att)
        se  = float(r.se)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        att, se = float('nan'), float('nan')
    results[lang] = (att, se)
    main_att, main_se = MAIN_SDID[lang]
    diff_pct = ((att - main_att) / abs(main_att) * 100) if main_att != 0 else float('nan')
    print(f"    Restricted ATT = {att:.3f}  SE = {se:.3f}  {_stars(att, se)}")
    print(f"    Main ATT       = {main_att:.3f}  SE = {main_se:.3f}  {_stars(main_att, main_se)}")
    print(f"    Difference     = {diff_pct:+.1f}%")

# ── Write LaTeX table ─────────────────────────────────────────────────────────
print("\n  Writing robustness_restricted_control.tex ...")

NOTE = (
    r"Estimaciones SDID con grupo de control restringido: se excluyen los 9 países clasificados "
    r"como \textit{Not Free} con puntaje inferior a 25 en el índice \textit{Freedom on the Net} "
    r"2022 (Freedom House): China, Irán, Siria, Bielorrusia, Rusia, Cuba, Tayikistán, Vietnam y "
    r"Turkmenistán. El grupo de control pasa de 29 a 20 países. Las especificaciones principales "
    r"corresponden al Cuadro 3. Los errores est\'{a}ndar se obtienen por \textit{bootstrap} (100 "
    r"replicaciones). * p$<$0.10, ** p$<$0.05, *** p$<$0.01."
)

lines = [
    r"\begin{table}[H]\centering",
    r"\caption{Robustez: SDID con grupo de control restringido (excluye censores severos)}",
    r"\label{tab:tabla6}",
    r"\begin{threeparttable}",
    r"{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Lenguaje & \shortstack{ATT Restringido \\ (SE)} & Sig. & \shortstack{ATT Principal \\ (SE)} & \shortstack{Diferencia \\ (\%)} \\",
    r"\midrule",
]

for lang in LANGUAGES_5:
    tname = LANG_TEX[lang]
    att_r, se_r   = results[lang]
    att_m, se_m   = MAIN_SDID[lang]
    s_r = _stars(att_r, se_r)
    s_m = _stars(att_m, se_m)
    if not np.isnan(att_r) and att_m != 0:
        diff_pct = (att_r - att_m) / abs(att_m) * 100
        diff_str = f"{diff_pct:+.1f}\\%"
    else:
        diff_str = "---"
    lines.append(
        f"{tname} & {att_r:.3f}{s_r} & & {att_m:.3f}{s_m} & {diff_str} \\\\"
    )
    lines.append(
        f"         & ({se_r:.3f}) & & ({se_m:.3f}) & \\\\"
    )
    lines.append(r"\addlinespace")

lines += [
    r"\bottomrule",
    r"\end{tabular}}",
    r"\begin{tablenotes}",
    r"\footnotesize",
    rf"\item \textit{{Nota.}} {NOTE}",
    r"\end{tablenotes}",
    r"\end{threeparttable}",
    r"\end{table}",
    "",
]

outpath = p("output", "tables", "robustness_restricted_control.tex")
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"  Saved: {outpath}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
