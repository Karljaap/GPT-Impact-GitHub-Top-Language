"""
generate_placebo.py
--------------------
Placebo test for the ChatGPT/GitHub study.
- Uses only pre-treatment data: quarters 1-11 (Q1-2020 to Q3-2022)
- Assigns a false treatment at quarter 6 (Q2-2021)
- Runs SDID for each of the 10 languages
- Writes results to output/tables/placebo_test.tex
"""

from synthdid.synthdid import Synthdid
import numpy as np
import pandas as pd
import os
import scipy.stats as stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def p(*parts):
    return os.path.join(BASE_DIR, *parts)


LANGUAGES_5 = ["C", "C#", "C++", "Go", "Java", "JavaScript", "PHP", "Python", "Ruby", "TypeScript"]

LANG_SAFE = {
    "C": "C", "C#": "C_hashtag", "C++": "C_plus", "Go": "Go", "Java": "Java",
    "JavaScript": "JavaScript", "PHP": "PHP", "Python": "Python", "Ruby": "Ruby",
    "TypeScript": "TypeScript"
}

LANG_TEX = {
    "C": "C", "C#": r"C\#", "C++": "C++", "Go": "Go", "Java": "Java",
    "JavaScript": "JavaScript", "PHP": "PHP", "Python": "Python", "Ruby": "Ruby",
    "TypeScript": "TypeScript"
}

TREAT_START_PLACEBO = 6
N_BOOT = 100

# Real SDID results (no controls, from main analysis)
REAL_SDID = {
    "C": (1.276, 0.459),
    "C#": (0.443, 0.464),
    "C++": (1.131, 0.293),
    "Go": (0.720, 0.247),
    "Java": (1.725, 0.835),
    "JavaScript": (8.303, 2.974),
    "PHP": (0.916, 0.422),
    "Python": (4.525, 1.057),
    "Ruby": (0.808, 0.282),
    "TypeScript": (3.078, 0.833),
}


def _stars(att, se):
    if np.isnan(se) or se <= 0:
        return ''
    pv = 2.0 * (1.0 - stats.norm.cdf(abs(att / se)))
    if pv < 0.01:
        return '***'
    if pv < 0.05:
        return '**'
    if pv < 0.10:
        return '*'
    return ''


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading panel data...")
df_panel = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))
df_panel = df_panel[df_panel['iso2_code'] != 'HK'].copy()

# Use only pre-treatment quarters (1-11)
df_placebo = df_panel[df_panel['quarter'] <= 11].copy()

print(f"Placebo sample: {len(df_placebo)} rows, quarters 1-11")
print(f"Placebo treatment assigned at quarter {TREAT_START_PLACEBO}")

# ── Run SDID for each language ─────────────────────────────────────────────────
results = {}
for lang in LANGUAGES_5:
    print(f"\nRunning SDID placebo for {lang}...")
    df_l = df_placebo[df_placebo['language'] == lang].copy()
    df_s = df_l[['iso2_code', 'quarter', 'num_pushers_pc', 'gpt_available']].copy()
    df_s['treat'] = (
        (df_s['gpt_available'] == 1) & (df_s['quarter'] >= TREAT_START_PLACEBO)
    ).astype(int)

    try:
        r = (
            Synthdid(df_s, 'iso2_code', 'quarter', 'treat', 'num_pushers_pc')
            .fit()
            .vcov(method='bootstrap', n_reps=N_BOOT)
            .summary()
        )
        att = float(r.att)
        se = float(r.se)
    except Exception as exc:
        print(f"  {lang}: FAILED ({exc})")
        att, se = float('nan'), float('nan')

    results[lang] = (att, se)
    print(f"  [{lang}] Placebo ATT = {att:.3f}  SE = {se:.3f}  {_stars(att, se)}")


# ── Write LaTeX table ──────────────────────────────────────────────────────────
print("\nWriting placebo_test.tex...")

os.makedirs(p("output", "tables"), exist_ok=True)

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(r"\begin{threeparttable}")
lines.append(r"\caption{Prueba placebo: tratamiento ficticio en Q2-2021}")
lines.append(r"\label{tab:tabla5}")
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
lines.append(
    r"Lenguaje & ATT Placebo & SE & Significancia & ATT Real (SDID) & SE Real \\"
)
lines.append(r"\midrule")

for lang in LANGUAGES_5:
    att_p, se_p = results[lang]
    att_r, se_r = REAL_SDID[lang]
    stars_p = _stars(att_p, se_p)
    lang_tex = LANG_TEX[lang]

    att_p_str = f"{att_p:.3f}" if not np.isnan(att_p) else "---"
    se_p_str = f"{se_p:.3f}" if not np.isnan(se_p) else "---"
    att_r_str = f"{att_r:.3f}"
    se_r_str = f"{se_r:.3f}"

    lines.append(
        f"{lang_tex} & {att_p_str} & {se_p_str} & {stars_p} & {att_r_str} & {se_r_str} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\begin{tablenotes}")
lines.append(r"\small")
lines.append(
    r"\item \textit{Nota.} El tratamiento ficticio se asigna en Q2-2021, "
    r"período anterior al lanzamiento de ChatGPT. La muestra se restringe a los "
    r"11 trimestres pretratamiento (2020-Q1--2022-Q3). Si la estrategia de "
    r"identificación es válida, los ATT placebo deben ser estadísticamente "
    r"indistinguibles de cero."
)
lines.append(r"\item $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.")
lines.append(r"\end{tablenotes}")
lines.append(r"\end{threeparttable}")
lines.append(r"\end{table}")

out_path = p("output", "tables", "placebo_test.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Done! Table written to: {out_path}")
