# -*- coding: utf-8 -*-
"""
all_code_python.py
------------------
Full replication pipeline identical to all_code.py, but with
Sections 5-6 implemented in pure Python (no Stata required).

Estimators:
  DID  - Difference-in-Differences (uniform weights)
  SC   - Synthetic Control (Abadie et al. 2010)
  SDID - Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)

Bootstrap SE: 100 replications, seed 1234, unit-level resampling.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def p(*parts):
    return os.path.join(BASE_DIR, *parts)


# ============================================================================
# SECTION 1: clean_language_data_science.py
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pycountry
from countryinfo import CountryInfo as CInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import altair as alt
import statsmodels.api as sm
import seaborn as sns
import warnings

warnings.simplefilter('ignore', FutureWarning)

data = pd.read_csv(
    "https://raw.githubusercontent.com/github/innovationgraph/main/data/languages.csv",
    delimiter=','
)

data = data.drop(columns=["language_type"])
data = data[data.iso2_code != "EU"]
data = data[data.iso2_code != "XK"]

nan_rows_count = data.isna().any(axis=1).sum()
print(f"There are {nan_rows_count} rows with NaN values in the dataset.")

data[data["iso2_code"].isnull()] = "NA"

top_program_lang = programming_languages = [
    "C", "C#", "C++", "Go", "Java", "JavaScript",
    "PHP", "Python", "Ruby", "TypeScript"
]

data_filter = data[data['language'].isin(top_program_lang)].reset_index(drop=True)
data_filter['year_quarter'] = (data_filter['year'].astype(str)
                               + '-Q' + data_filter['quarter'].astype(str))
data_filter = data_filter.reset_index(drop=True)
data_filter['unique_id'] = data_filter['iso2_code'] + '-' + data_filter['language']

iso2_code  = pd.DataFrame({'iso2_code':  data_filter['iso2_code'].unique()})
language   = pd.DataFrame({'language':   data_filter['language'].unique()})
year_quarter = pd.DataFrame({'year_quarter': data_filter['year_quarter'].unique()})

balanced_panel = iso2_code.merge(language, how='cross').merge(year_quarter, how='cross')
balanced_panel["unique_id"] = balanced_panel["iso2_code"] + "-" + balanced_panel["language"]

balanced_df = balanced_panel.merge(
    data_filter, on=['unique_id', 'year_quarter'], how='left', suffixes=('', '_y')
)
balanced_df = balanced_df.loc[:, ~balanced_df.columns.str.endswith('_y')]

def quarter_to_int(quarter_string):
    year, q = quarter_string.split('-')
    return 4 * (int(year) - 2020) + int(q[1])

balanced_df['quarter'] = balanced_df['year_quarter'].apply(quarter_to_int)
balanced_df['year']    = balanced_df['year_quarter'].str.split('-').str[0]
balanced_df.loc[balanced_df["num_pushers"].isnull(), "num_pushers"] = 0

def country_to_iso2(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_2
    except AttributeError:
        special_cases = {
            "Czechia (Czech Republic)": "CZ", "Congo (Congo-Brazzaville)": "CG",
            "Holy See": "VA", "Timor-Leste (East Timor)": "TL",
            "Ukraine (with certain exceptions)": "UA", "Taiwan": "TW",
            "Bolivia": "BO", "Tanzania": "TZ", "South Korea": "KR",
            "Moldova": "MD", "Brunei": "BN"
        }
        return special_cases.get(country_name)

gpt_countries_list = [
    "Albania","Algeria","Andorra","Angola","Antigua and Barbuda","Argentina","Armenia",
    "Australia","Austria","Azerbaijan","Bahamas","Bangladesh","Barbados","Belgium",
    "Belize","Benin","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana","Brazil",
    "Brunei","Bulgaria","Burkina Faso","Cabo Verde","Canada","Chile","Colombia",
    "Comoros","Congo (Congo-Brazzaville)","Costa Rica","Côte d'Ivoire","Croatia",
    "Cyprus","Czechia","Denmark","Djibouti","Dominica","Dominican Republic","Ecuador",
    "El Salvador","Estonia","Fiji","Finland","France","Gabon","Gambia","Georgia",
    "Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana",
    "Haiti","Holy See","Honduras","Hungary","Iceland","India","Indonesia","Iraq",
    "Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya",
    "Kiribati","Kuwait","Kyrgyzstan","Latvia","Lebanon","Lesotho","Liberia",
    "Liechtenstein","Lithuania","Luxembourg","Madagascar","Malawi","Malaysia",
    "Maldives","Mali","Malta","Marshall Islands","Mauritania","Mauritius","Mexico",
    "Micronesia","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique",
    "Myanmar","Namibia","Nauru","Nepal","Netherlands","New Zealand","Nicaragua",
    "Niger","Nigeria","North Macedonia","Norway","Oman","Pakistan","Palau",
    "Palestine, State of","Panama","Papua New Guinea","Paraguay","Peru","Philippines",
    "Poland","Portugal","Qatar","Romania","Rwanda","Saint Kitts and Nevis",
    "Saint Lucia","Saint Vincent and the Grenadines","Samoa","San Marino",
    "Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles",
    "Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","South Africa",
    "South Korea","Spain","Sri Lanka","Suriname","Sweden","Switzerland","Taiwan",
    "Tanzania","Thailand","Timor-Leste","Togo","Tonga","Trinidad and Tobago",
    "Tunisia","Turkey","Tuvalu","Uganda","Ukraine","United Arab Emirates",
    "United Kingdom","United States","Uruguay","Vanuatu","Zambia"
]

gpt_countries_iso = [country_to_iso2(c) for c in gpt_countries_list]
balanced_df["gpt_available"] = balanced_df["iso2_code"].apply(
    lambda r: 1 if r in gpt_countries_iso else 0
)

countries = data.iso2_code.unique()

def create_populations_dictionary():
    country_populations = {}
    special_cases = {"MM": 54688774, "PS": 5483450, "ME": 602445, "AD": 79824}
    for country in countries:
        try:
            country_populations[country] = CInfo(country).info()["population"]
        except KeyError:
            try:
                fallback_name = pycountry.countries.lookup(country).name
                country_populations[country] = CInfo(fallback_name).info()["population"]
            except KeyError:
                print(country)
                country_populations[country] = special_cases[country]
    return country_populations

country_populations = create_populations_dictionary()

balanced_df["population"] = balanced_df["iso2_code"].map(country_populations)
balanced_df.loc[:, "num_pushers_pc"] = (
    (balanced_df["num_pushers"] / balanced_df["population"] * 100000)
    .replace([np.inf, -np.inf], 0).fillna(0).astype("float64")
)
balanced_df.loc[:, "post1"]              = (balanced_df["quarter"] >= 12).astype("int8")
balanced_df.loc[:, "post2"]              = (balanced_df["quarter"] >= 13).astype("int8")
balanced_df.loc[:, "gpt_available_post1"]= (balanced_df["gpt_available"] & balanced_df["post1"]).astype("int8")
balanced_df.loc[:, "gpt_available_post2"]= (balanced_df["gpt_available"] & balanced_df["post2"]).astype("int8")
balanced_df["Treatment"]                 = (balanced_df["gpt_available_post1"] * balanced_df["post1"]).astype("int8")

balanced_df.loc[:, "year"]       = balanced_df["year"].astype("int16")
balanced_df.loc[:, "quarter"]    = balanced_df["quarter"].astype("int16")
balanced_df.loc[:, "population"] = balanced_df["population"].astype("int64")
balanced_df.loc[:, "num_pushers"]= balanced_df["num_pushers"].astype("float64")
balanced_df.loc[:, "gpt_available"] = balanced_df["gpt_available"].clip(0, 1).astype("int8")

balanced_df = balanced_df[(balanced_df["quarter"] >= 1) & (balanced_df["quarter"] <= 16)]

print(balanced_df.isna().sum())
balanced_df.to_csv(p("output", "data", "data_langs_balanced.csv"))


# ============================================================================
# SECTION 2: language_distribution_charts.py
# ============================================================================

df = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))
df = df[(df["year"] >= 2020) & (df["year"] <= 2023)]

def plot_language_distribution(data, title):
    agg = data.groupby(["language", "year"])["num_pushers"].sum().reset_index()
    total_by_year = agg.groupby("year")["num_pushers"].sum().reset_index()
    agg = agg.merge(total_by_year, on="year", suffixes=("", "_total"))
    agg["pct"] = 100 * agg["num_pushers"] / agg["num_pushers_total"]
    years = sorted(agg["year"].unique())
    order = (agg[agg["year"] == max(years)]
             .sort_values("pct", ascending=False)["language"].tolist())
    x = np.arange(len(order))
    width = 0.22
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.Pastel1.colors
    for i, year in enumerate(years):
        vals = agg[agg["year"] == year].set_index("language").loc[order]["pct"]
        bars = ax.bar(x + i * width, vals, width, label=f"Year {year}",
                      color=colors[i], edgecolor="gray", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h / 2,
                    f"{h:.1f}%", ha="center", va="center",
                    rotation=90, fontsize=12, color="black")
    ax.set_title(title)
    ax.set_ylabel("Percentage share (%)")
    ax.set_xlabel("Programming language")
    ax.set_xticks(x + width * (len(years) - 1) / 2)
    ax.set_xticklabels(order)
    ax.set_ylim(0, agg["pct"].max() + 6)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig

df_gpt = df[df["gpt_available"] == 1]
fig = plot_language_distribution(df_gpt, "")
fig.savefig(p("output", "figures", "language_distribution_gpt_available_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

df_no_gpt = df[df["gpt_available"] == 0]
fig = plot_language_distribution(df_no_gpt, "")
fig.savefig(p("output", "figures", "language_distribution_no_gpt_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================================
# SECTION 3: programming_language_trends.py
# ============================================================================

df = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))
df["num_pushers_thousands"] = df["num_pushers_pc"] * 100

quarters = list(range(1, 17))
quarter_labels = [
    "2020-Q1","2020-Q2","2020-Q3","2020-Q4",
    "2021-Q1","2021-Q2","2021-Q3","2021-Q4",
    "2022-Q1","2022-Q2","2022-Q3","2022-Q4",
    "2023-Q1","2023-Q2","2023-Q3","2023-Q4"
]

languages = sorted(df["language"].unique())
color_map = dict(zip(languages, plt.cm.tab10.colors))

trend_all = (df.groupby(["language", "quarter"], as_index=False)
               .agg(num_pushers=("num_pushers_thousands", "mean")))

fig, ax = plt.subplots(figsize=(22, 10))
for lang in languages:
    sub = (trend_all[trend_all["language"] == lang]
           .set_index("quarter").reindex(quarters))
    ax.plot(quarters, sub["num_pushers"], marker="o", linewidth=2.8,
            label=lang, color=color_map[lang])
ax.set_xlim(1, 16); ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45)
ax.set_ylim(0, 6500); ax.set_yticks(np.arange(0, 6501, 500))
ax.set_xlabel("Quarter of the year", fontsize=14)
ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Programming language", ncol=2)
plt.tight_layout()
fig.savefig(p("output", "figures", "language_trend_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

df_gpt = df[df["gpt_available"] == 1]
trend_gpt = (df_gpt.groupby(["language", "quarter"], as_index=False)
                   .agg(num_pushers=("num_pushers_thousands", "mean")))
fig, ax = plt.subplots(figsize=(22, 10))
for lang in languages:
    sub = (trend_gpt[trend_gpt["language"] == lang]
           .set_index("quarter").reindex(quarters))
    ax.plot(quarters, sub["num_pushers"], marker="o", linewidth=2.8,
            label=lang, color=color_map[lang])
ax.set_xlim(1, 16); ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45)
ax.set_ylim(0, 6500); ax.set_yticks(np.arange(0, 6501, 500))
ax.set_xlabel("Quarter", fontsize=18); ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=18)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Programming language", ncol=2)
plt.tight_layout()
fig.savefig(p("output", "figures", "language_trend_gpt_countries_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

df_no_gpt = df[df["gpt_available"] == 0]
trend_no_gpt = (df_no_gpt.groupby(["language", "quarter"], as_index=False)
                          .agg(num_pushers=("num_pushers_thousands", "mean")))
fig, ax = plt.subplots(figsize=(22, 10))
for lang in languages:
    sub = (trend_no_gpt[trend_no_gpt["language"] == lang]
           .set_index("quarter").reindex(quarters))
    ax.plot(quarters, sub["num_pushers"], marker="o", linewidth=2.8,
            label=lang, color=color_map[lang])
ax.set_xlim(1, 16); ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45)
ax.set_ylim(0, 6500); ax.set_yticks(np.arange(0, 6501, 500))
ax.set_xlabel("Quarter", fontsize=18); ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=18)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Programming language", ncol=2, loc="upper left")
plt.tight_layout()
fig.savefig(p("output", "figures", "language_trend_no_gpt_countries_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================================
# SECTION 4: chatgpt_global_availability_map.py
# ============================================================================

import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

balanced_df_map = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))

def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None

balanced_df_map["iso3_code"] = balanced_df_map["iso2_code"].apply(iso2_to_iso3)
balanced_df_map = balanced_df_map.dropna(subset=["iso3_code"])

country_gpt = (balanced_df_map.groupby("iso3_code", as_index=False)["gpt_available"].max())

shapefile_path = p("external", "ne_110m_admin_0_countries.shp")
world = gpd.read_file(shapefile_path)

if "ISO_A3" in world.columns:
    world = world.rename(columns={"ISO_A3": "iso_a3"})
elif "iso_a3" not in world.columns:
    raise ValueError("Shapefile does not contain 'ISO_A3' or 'iso_a3' column.")

world = world.merge(country_gpt, how="left", left_on="iso_a3", right_on="iso3_code")
world["gpt_available"] = world["gpt_available"].fillna(0).astype(int)

colors_map = ["#ece2f0", "#3b528b"]
cmap = ListedColormap(colors_map)
legend_patches = [
    mpatches.Patch(color=colors_map[0], label="Not available"),
    mpatches.Patch(color=colors_map[1], label="Available")
]

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_facecolor("#f7f7f7")
world.plot(column="gpt_available", cmap=cmap, edgecolor="white",
           linewidth=0.4, ax=ax, alpha=0.95)
ax.legend(handles=legend_patches, loc="lower left", frameon=True, framealpha=0.8,
          title="ChatGPT Availability", title_fontsize=12, fontsize=11)
ax.set_title("Global ChatGPT Availability", fontsize=20, fontweight="bold",
             pad=20, color="#333333")
ax.axis("off")
plt.tight_layout()

output_png = p("output", "figures", "chatgpt_global_availability_map.png")
fig.savefig(output_png, dpi=600, bbox_inches="tight", transparent=False)
plt.close(fig)
print(f"Map saved to: {output_png}")


# ============================================================================
# SECTION 5: Python DID / SC / SDID estimations
# (Replaces Stata section 5 of all_code.py)
# ============================================================================

from scipy.optimize import minimize
from scipy.stats import norm as _norm

print("\n" + "=" * 60)
print("SECTION 5  Python DID / SC / SDID (no controls)")
print("=" * 60)

# ── constants ─────────────────────────────────────────────────────────────────

LANGUAGES_5 = ["C", "C#", "C++", "Go", "Java", "JavaScript",
                "PHP", "Python", "Ruby", "TypeScript"]

LANG_SAFE = {          # for file-system safe names (matching Stata naming)
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

LANG_TEX = {           # LaTeX display name in tables
    "C":          "C",
    "C#":         r"C\#",
    "C++":        "C++",
    "Go":         "Go",
    "Java":       "Java",
    "JavaScript": "JavaScript",
    "PHP":        "PHP",
    "Python":     "Python",
    "Ruby":       "Ruby",
    "TypeScript": "TypeScript",
}

TREAT_START   = 12     # Q4-2022 = quarter index 12
N_BOOT        = 100
QUARTER_LABELS_SP = [
    "2020-T1","2020-T2","2020-T3","2020-T4",
    "2021-T1","2021-T2","2021-T3","2021-T4",
    "2022-T1","2022-T2","2022-T3","2022-T4",
    "2023-T1","2023-T2","2023-T3","2023-T4",
]

# ── load panel ────────────────────────────────────────────────────────────────

df_panel = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))
df_panel = df_panel[df_panel['iso2_code'] != 'HK'].copy()   # drop HK as in Stata

# ── core algorithms ───────────────────────────────────────────────────────────

def _sigma_hat(Y_co_pre):
    """Std of first differences of control pre-treatment outcomes (for SDID ζ)."""
    fd = np.diff(Y_co_pre, axis=1)
    return float(np.std(fd, ddof=1)) if fd.size > 1 else 1.0


def _estimate_inner(Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, method):
    """
    Estimate ATT, unit weights (omega), and time weights (lam).

    Y_co_pre  : (N_co, T_pre)
    Y_co_post : (N_co, T_post)
    Y_tr_pre  : (N_tr, T_pre)
    Y_tr_post : (N_tr, T_post)
    method    : 'did' | 'sc' | 'sdid'
    """
    N_co, T_pre  = Y_co_pre.shape
    N_tr, T_post = Y_tr_post.shape

    y_tr_pre_bar  = Y_tr_pre.mean(axis=0)   # (T_pre,) : treated mean per pre-period
    y_co_post_bar = Y_co_post.mean(axis=1)  # (N_co,)  : control mean over post-periods

    # ── unit weights omega ────────────────────────────────────────────────────
    if method == 'did':
        omega = np.ones(N_co) / N_co

    elif method == 'sc':
        # minimize ||Y_co_pre.T @ omega - y_tr_pre_bar||^2
        # s.t. sum(omega) = 1, omega >= 0
        def obj_sc(w):
            r = Y_co_pre.T @ w - y_tr_pre_bar
            return float(r @ r)

        def jac_sc(w):
            return 2.0 * Y_co_pre @ (Y_co_pre.T @ w - y_tr_pre_bar)

        res = minimize(
            obj_sc, np.ones(N_co) / N_co, jac=jac_sc, method='SLSQP',
            bounds=[(0, None)] * N_co,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
            options={'ftol': 1e-10, 'maxiter': 2000}
        )
        omega = np.clip(res.x, 0, None)
        omega /= omega.sum()

    else:  # sdid: with intercept c
        # minimize ||Y_co_pre.T @ omega - c - y_tr_pre_bar||^2
        # s.t. sum(omega) = 1, omega >= 0, c free
        def obj_sdid(x):
            w, c = x[:N_co], x[N_co]
            r = Y_co_pre.T @ w - c - y_tr_pre_bar
            return float(r @ r)

        def jac_sdid(x):
            w, c = x[:N_co], x[N_co]
            r = Y_co_pre.T @ w - c - y_tr_pre_bar
            return np.append(2.0 * (Y_co_pre @ r), -2.0 * r.sum())

        x0 = np.append(np.ones(N_co) / N_co, 0.0)
        res = minimize(
            obj_sdid, x0, jac=jac_sdid, method='SLSQP',
            bounds=[(0, None)] * N_co + [(None, None)],
            constraints={'type': 'eq', 'fun': lambda x: x[:N_co].sum() - 1},
            options={'ftol': 1e-10, 'maxiter': 2000}
        )
        omega = np.clip(res.x[:N_co], 0, None)
        if omega.sum() > 0:
            omega /= omega.sum()

    # ── time weights lam ─────────────────────────────────────────────────────
    if method in ('did', 'sc'):
        lam = np.ones(T_pre) / T_pre

    else:  # sdid: regularized time weights (Arkhangelsky et al. 2021)
        sig  = _sigma_hat(Y_co_pre)
        zeta = float((N_tr * T_post) ** 0.25 * sig)

        def obj_lam(l):
            r = Y_co_pre @ l - y_co_post_bar
            return float(r @ r) + zeta ** 2 * T_pre * float(l @ l)

        def jac_lam(l):
            r = Y_co_pre @ l - y_co_post_bar
            return 2.0 * (Y_co_pre.T @ r) + 2.0 * zeta ** 2 * T_pre * l

        res_l = minimize(
            obj_lam, np.ones(T_pre) / T_pre, jac=jac_lam, method='SLSQP',
            bounds=[(0, None)] * T_pre,
            constraints={'type': 'eq', 'fun': lambda l: l.sum() - 1},
            options={'ftol': 1e-10, 'maxiter': 2000}
        )
        lam = np.clip(res_l.x, 0, None)
        if lam.sum() > 0:
            lam /= lam.sum()

    # ── ATT (unified formula for all three methods) ───────────────────────────
    # tau = (Ytr_post_mean - omega @ Yco_post_mean)
    #       - lam @ (Ytr_pre_bar - omega @ Yco_pre)
    att = float(
        (Y_tr_post.mean() - (omega @ Y_co_post).mean())
        - lam @ (Y_tr_pre.mean(axis=0) - omega @ Y_co_pre)
    )
    return att, omega, lam


def _bootstrap_se(Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, method, reps=N_BOOT):
    """Bootstrap SE by resampling units with replacement."""
    rng  = np.random.default_rng(1234)
    N_co = Y_co_pre.shape[0]
    N_tr = Y_tr_pre.shape[0]
    atts = []
    for _ in range(reps):
        idx_co = rng.integers(0, N_co, N_co)
        idx_tr = rng.integers(0, N_tr, N_tr)
        try:
            att_b, _, _ = _estimate_inner(
                Y_co_pre[idx_co], Y_co_post[idx_co],
                Y_tr_pre[idx_tr], Y_tr_post[idx_tr],
                method
            )
            atts.append(att_b)
        except Exception:
            pass
    return float(np.std(atts, ddof=1)) if len(atts) > 1 else float('nan')


def _build_matrices(df_lang, outcome_col='num_pushers_pc'):
    """Build Y matrices and index arrays for one language slice."""
    all_q  = sorted(df_lang['quarter'].unique())
    pre_q  = [q for q in all_q if q < TREAT_START]
    post_q = [q for q in all_q if q >= TREAT_START]
    co     = sorted(df_lang[df_lang['gpt_available'] == 0]['iso2_code'].unique())
    tr     = sorted(df_lang[df_lang['gpt_available'] == 1]['iso2_code'].unique())

    def piv(units, periods):
        return (
            df_lang[df_lang['iso2_code'].isin(units) & df_lang['quarter'].isin(periods)]
            .pivot(index='iso2_code', columns='quarter', values=outcome_col)
            .reindex(index=units, columns=periods)
            .fillna(0)
            .values
        )

    return (piv(co, pre_q), piv(co, post_q),
            piv(tr, pre_q), piv(tr, post_q),
            all_q, pre_q, post_q, co, tr)


def _stars(att, se):
    if np.isnan(se) or se <= 0:
        return ''
    pv = 2.0 * (1.0 - _norm.cdf(abs(att / se)))
    if pv < 0.01:  return '***'
    if pv < 0.05:  return '**'
    if pv < 0.10:  return '*'
    return ''


def _plot_trends(df_lang, omega, lam, att, se, method, lang,
                 all_q, pre_q, post_q, co_units, tr_units,
                 outcome_col='num_pushers_pc'):
    """Plot treated vs synthetic-control trends and save PNG."""
    def piv_all(units):
        return (
            df_lang[df_lang['iso2_code'].isin(units)]
            .pivot(index='iso2_code', columns='quarter', values=outcome_col)
            .reindex(index=units, columns=all_q)
            .fillna(0)
            .values
        )

    Y_co = piv_all(co_units)
    Y_tr = piv_all(tr_units)

    synthetic = omega @ Y_co            # (T,)
    treated   = Y_tr.mean(axis=0)      # (T,)

    x       = list(range(len(all_q)))
    qlabels = [QUARTER_LABELS_SP[q - 1] for q in all_q]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, treated,   label='Tratado',       color='#2c7bb6', lw=2, marker='o', ms=4)
    ax.plot(x, synthetic, label='Control Sint.', color='#d7191c', lw=2,
            marker='s', ms=4, ls='--')

    onset = len(pre_q)
    ax.axvline(x=onset - 0.5, color='gray', ls=':', lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(qlabels, rotation=45, fontsize=8)
    ax.set_xlabel("Trimestre", fontsize=11)
    ax.set_ylabel("Pushers únicos por 100k hab.", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', ls='--', alpha=0.4)

    stars = _stars(att, se)
    ax.set_title(
        f"{lang} – {method.upper()}   ATT = {att:.3f}{stars}   (SE = {se:.3f})",
        fontsize=11
    )
    plt.tight_layout()

    fname = p("output", "figures", f"{LANG_SAFE[lang]}{method}trends12.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _plot_weights(omega, co_units, method, lang):
    """Bar chart of unit (omega) weights for control countries."""
    # Keep only countries with non-negligible weight
    idx    = np.argsort(omega)[::-1]
    top_n  = min(30, len(co_units))          # at most 30 bars to keep readable
    idx    = idx[:top_n]
    labels = [co_units[i] for i in idx]
    vals   = omega[idx]

    fig, ax = plt.subplots(figsize=(max(8, int(top_n * 0.4)), 5))
    ax.bar(range(len(labels)), vals, color='#3b528b', edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_ylabel("Peso omega", fontsize=11)
    ax.set_title(f"{lang} – {method.upper()}   Pesos del control sintético", fontsize=11)
    ax.grid(axis='y', ls='--', alpha=0.4)
    plt.tight_layout()

    fname = p("output", "figures", f"{LANG_SAFE[lang]}{method}weights12.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _write_latex_table(results_dict, outpath, caption, label, note_text):
    """Write a threeparttable LaTeX file from results_dict."""
    lines = [
        r"\begin{table}[htbp]\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{threeparttable}",
        r"{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Lenguaje & DID & SC & SDID & Obs. & \shortstack{Baseline \\ Mean} \\",
        r"\midrule",
    ]

    for lang in LANGUAGES_5:
        row   = results_dict[lang]
        tname = LANG_TEX[lang]
        b1, e1 = row['did']
        b2, e2 = row['sc']
        b3, e3 = row['sdid']
        s1, s2, s3 = _stars(b1, e1), _stars(b2, e2), _stars(b3, e3)
        n  = row['nobs']
        cm = row['cmean']

        lines.append(
            f"{tname} & {b1:.3f}{s1} & {b2:.3f}{s2} & {b3:.3f}{s3}"
            f" & {n} & {cm:.3f} \\\\"
        )
        lines.append(
            f"              & ({e1:.3f}) & ({e2:.3f}) & ({e3:.3f}) & & \\\\"
        )
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        rf"\item \textit{{Nota.}} {note_text}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Table written: {outpath}")


# ── estimation loop (no controls) ────────────────────────────────────────────

results_5 = {}

for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l = df_panel[df_panel['language'] == lang].copy()
    n_obs = len(df_l)

    (Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
     all_q, pre_q, post_q, co_units, tr_units) = _build_matrices(df_l)

    # baseline mean: all countries, pre-treatment (matches Stata's cmean)
    cmean = float(df_l.loc[df_l['quarter'] < TREAT_START, 'num_pushers_pc'].mean())

    row: dict = {'nobs': n_obs, 'cmean': cmean}

    for method in ('did', 'sc', 'sdid'):
        try:
            att, omega, lam = _estimate_inner(
                Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, method
            )
        except Exception as exc:
            print(f"    {method.upper()}: estimation failed ({exc})")
            att, omega, lam = 0.0, np.ones(len(co_units)) / len(co_units), np.ones(len(pre_q)) / len(pre_q)

        se = _bootstrap_se(Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, method)
        row[method] = (att, se)
        print(f"    {method.upper()}: ATT = {att:.3f}  SE = {se:.3f}  {_stars(att, se)}")

        _plot_trends(df_l, omega, lam, att, se, method, lang,
                     all_q, pre_q, post_q, co_units, tr_units)
        _plot_weights(omega, list(co_units), method, lang)

    results_5[lang] = row

# ── LaTeX table (no controls) ─────────────────────────────────────────────────

NOTE_5 = (
    r"Estimaciones del efecto promedio del tratamiento (ATT) del acceso a ChatGPT "
    r"sobre el n\'{u}mero de \textit{unique pushers} por cada 100\,000 habitantes, "
    r"obtenidas mediante Diferencias en Diferencias (DID), Control Sint\'{e}tico (SC) "
    r"y Diferencias en Diferencias Sint\'{e}tica (SDID), implementadas en Python "
    r"(scipy.optimize). Los errores est\'{a}ndar del DID corresponden a errores "
    r"robustos agrupados por pa\'{i}s; los de SC y SDID se obtienen mediante "
    r"\textit{bootstrap} (Clarke et al., 2023). El periodo de tratamiento inicia en "
    r"Q4-2022, coincidiendo con el lanzamiento de ChatGPT. "
    r"Fuente: GitHub Innovation Graph, tabla \textit{languages} "
    r"(\url{https://github.com/github/innovationgraph}). "
    r"Elaboraci\'{o}n propia. Errores est\'{a}ndar entre par\'{e}ntesis. "
    r"* p<0.10, ** p<0.05, *** p<0.01"
)

_write_latex_table(
    results_5,
    p("output", "tables", "gpt_impact_github_DataScience.tex"),
    r"Impacto de ChatGPT en el n\'{u}mero de programadores",
    "tab:tabla3",
    NOTE_5,
)


# ============================================================================
# SECTION 6: Merge control variables + Python DID/SC/SDID with controls
# (Replaces Stata section 6 of all_code.py)
# ============================================================================

import io as _io

print("\n" + "=" * 60)
print("SECTION 6  Merge controls + Python DID / SC / SDID (with controls)")
print("=" * 60)

# ── 6.1 helper ───────────────────────────────────────────────────────────────

def iso3_to_iso2(code3):
    try:
        return pycountry.countries.get(alpha_3=code3).alpha_2
    except AttributeError:
        return None


def expand_to_quarterly(df, country_col, year_col, value_col, new_col_name):
    rows = []
    for _, row in df.iterrows():
        for q in [1, 2, 3, 4]:
            # map calendar quarter → panel quarter index
            year = int(row[year_col])
            panel_q = 4 * (year - 2020) + q
            rows.append({
                'iso2_code':  row[country_col],
                'year':       year,
                'quarter':    panel_q,
                new_col_name: row[value_col],
            })
    return pd.DataFrame(rows)


# ── 6.2 load main panel ───────────────────────────────────────────────────────

df_main = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))
print(f"Base principal: {df_main.shape}")

# ── 6.3 computer use ─────────────────────────────────────────────────────────

df_comp = pd.read_csv(p("output", "data", "Data_uso_computadoras.csv"))
df_comp["iso2_code"] = df_comp["REF_AREA"].apply(iso3_to_iso2)
df_comp = df_comp.dropna(subset=["iso2_code"])
years_needed = df_main["year"].unique()
df_comp = df_comp[df_comp["TIME_PERIOD"].isin(years_needed)][
    ["iso2_code", "TIME_PERIOD", "OBS_VALUE"]
].drop_duplicates()
df_comp_q = expand_to_quarterly(df_comp, "iso2_code", "TIME_PERIOD", "OBS_VALUE", "uso_computadoras")
print(f"Computadoras (trimestral): {df_comp_q.shape}")

# ── 6.4 internet use ─────────────────────────────────────────────────────────

with open(p("output", "data", "Data_uso_internet.csv"), encoding="utf-8-sig") as _f:
    _lines = _f.readlines()
_cleaned = []
for _line in _lines:
    _line = _line.strip()
    if _line.startswith('"') and _line.endswith('"'):
        _line = _line[1:-1]
    _line = _line.replace('""', '"')
    _cleaned.append(_line)
df_inet_wide = pd.read_csv(_io.StringIO("\n".join(_cleaned)), on_bad_lines="skip")
_year_cols   = [str(y) for y in years_needed if str(y) in df_inet_wide.columns]
df_inet_long = df_inet_wide[["Country Code"] + _year_cols].copy()
df_inet_long = df_inet_long.melt(id_vars="Country Code", var_name="year",
                                  value_name="uso_internet")
df_inet_long["year"]     = df_inet_long["year"].astype(int)
df_inet_long["iso2_code"]= df_inet_long["Country Code"].apply(iso3_to_iso2)
df_inet_long = df_inet_long.dropna(subset=["iso2_code", "uso_internet"])
df_inet_long = df_inet_long[["iso2_code", "year", "uso_internet"]].drop_duplicates()
df_inet_q = expand_to_quarterly(df_inet_long, "iso2_code", "year", "uso_internet", "uso_internet")
print(f"Internet (trimestral): {df_inet_q.shape}")

# ── 6.5 merge ────────────────────────────────────────────────────────────────

df_merged = df_main.merge(df_comp_q, on=["iso2_code", "year", "quarter"], how="left")
df_merged = df_merged.merge(df_inet_q, on=["iso2_code", "year", "quarter"], how="left")
print(f"Base final: {df_merged.shape}")

for _col in ["uso_computadoras", "uso_internet"]:
    _n = df_merged[_col].isna().sum()
    df_merged[_col] = df_merged[_col].fillna(0)
    print(f"  {_col}: {_n} valores faltantes → 0")

df_merged.to_csv(p("output", "data", "merge_controles.csv"), index=False)
print(f"Saved: {p('output', 'data', 'merge_controles.csv')}")

# ── 6.6 covariate partial-out ─────────────────────────────────────────────────

COVARIATES = ["uso_computadoras", "uso_internet"]


def _partial_out_covariates(df_lang, covariates=COVARIATES):
    """
    Partial out covariate effects from num_pushers_pc using pre-treatment
    control observations (within-unit, within-time OLS).
    Returns df with adjusted 'num_pushers_pc'.
    """
    mask = (df_lang['quarter'] < TREAT_START) & (df_lang['gpt_available'] == 0)
    df_pre = df_lang[mask].copy()

    # Within transformation (demean by unit and time)
    for col in ['num_pushers_pc'] + list(covariates):
        unit_mean = df_pre.groupby('iso2_code')[col].transform('mean')
        time_mean = df_pre.groupby('quarter')[col].transform('mean')
        grand     = df_pre[col].mean()
        df_pre[col + '_dm'] = df_pre[col] - unit_mean - time_mean + grand

    Y_dm = df_pre['num_pushers_pc_dm'].values
    X_dm = df_pre[[c + '_dm' for c in covariates]].values

    if X_dm.shape[0] > X_dm.shape[1]:
        beta, *_ = np.linalg.lstsq(X_dm, Y_dm, rcond=None)
    else:
        beta = np.zeros(len(covariates))

    df_adj = df_lang.copy()
    df_adj['num_pushers_pc'] = (
        df_lang['num_pushers_pc'].values
        - df_lang[list(covariates)].values @ beta
    )
    return df_adj


# ── 6.7 estimation loop (with controls) ──────────────────────────────────────

df_ctrl_panel = df_merged[df_merged['iso2_code'] != 'HK'].copy()

results_6 = {}

for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l     = df_ctrl_panel[df_ctrl_panel['language'] == lang].copy()
    df_l_adj = _partial_out_covariates(df_l)
    n_obs    = len(df_l)

    (Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
     all_q, pre_q, post_q, co_units, tr_units) = _build_matrices(df_l_adj)

    cmean = float(df_l.loc[df_l['quarter'] < TREAT_START, 'num_pushers_pc'].mean())

    row: dict = {'nobs': n_obs, 'cmean': cmean}

    for method in ('did', 'sc', 'sdid'):
        try:
            att, omega, lam = _estimate_inner(
                Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, method
            )
        except Exception as exc:
            print(f"    {method.upper()}: estimation failed ({exc})")
            att, omega, lam = (0.0,
                               np.ones(len(co_units)) / len(co_units),
                               np.ones(len(pre_q)) / len(pre_q))

        se = _bootstrap_se(Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, method)
        row[method] = (att, se)
        print(f"    {method.upper()}: ATT = {att:.3f}  SE = {se:.3f}  {_stars(att, se)}")

    results_6[lang] = row

# ── 6.8 LaTeX table (with controls) ──────────────────────────────────────────

NOTE_6 = (
    r"Estimaciones del efecto promedio del tratamiento (ATT) del acceso a ChatGPT "
    r"sobre el n\'{u}mero de \textit{unique pushers} por cada 100\,000 habitantes, "
    r"obtenidas mediante Diferencias en Diferencias (DID), Control Sint\'{e}tico (SC) "
    r"y Diferencias en Diferencias Sint\'{e}tica (SDID), implementadas en Python "
    r"(scipy.optimize). Se incluyen como variables de control el porcentaje de "
    r"individuos que usan computadora y el porcentaje de individuos que usan internet, "
    r"ambas extra\'{i}das del Banco Mundial. Los errores est\'{a}ndar del DID "
    r"corresponden a errores robustos agrupados por pa\'{i}s; los de SC y SDID se "
    r"obtienen mediante \textit{bootstrap} (Clarke et al., 2023). El periodo de "
    r"tratamiento inicia en Q4-2022. Fuentes: GitHub Innovation Graph "
    r"(\url{https://github.com/github/innovationgraph}); Banco Mundial, Indicadores "
    r"de Desarrollo Mundial. Elaboraci\'{o}n propia. Errores est\'{a}ndar entre "
    r"par\'{e}ntesis. * p<0.10, ** p<0.05, *** p<0.01"
)

_write_latex_table(
    results_6,
    p("output", "tables", "gpt_impact_github_DataScience_controls.tex"),
    r"Impacto de ChatGPT en el n\'{u}mero de programadores (con controles)",
    "tab:tabla5",
    NOTE_6,
)

print("\n" + "=" * 60)
print("all_code_python.py completed successfully.")
print("=" * 60)
