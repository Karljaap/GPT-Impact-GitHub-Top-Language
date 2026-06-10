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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import altair as alt
import statsmodels.api as sm
import seaborn as sns
import warnings

warnings.simplefilter('ignore', FutureWarning)

# Skip URL fetch + countryinfo step when the balanced panel already exists.
# countryinfo is incompatible with Python 3.12 (uses deprecated pkgutil.ImpImporter).
_BALANCED_CSV = p("output", "data", "data_langs_balanced.csv")
_REBUILD_PANEL = not os.path.exists(_BALANCED_CSV)
if not _REBUILD_PANEL:
    print(f"SECTION 1  Skipped: balanced panel cached at {_BALANCED_CSV}")

if _REBUILD_PANEL:
    from countryinfo import CountryInfo as CInfo
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

    # Restrict to countries that actually appear in 2020-2023 data (quarters 1-16)
    # Prevents countries that only appear in 2024+ from entering the balanced panel as all-zeros
    data_filter_1_16 = data_filter[(data_filter['year'] >= 2020) & (data_filter['year'] <= 2023)]
    iso2_code  = pd.DataFrame({'iso2_code':  data_filter_1_16['iso2_code'].unique()})
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
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = plt.cm.Pastel1.colors
    for i, year in enumerate(years):
        vals = agg[agg["year"] == year].set_index("language").loc[order]["pct"]
        bars = ax.bar(x + i * width, vals, width, label=f"Año {year}",
                      color=colors[i], edgecolor="gray", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h / 2,
                    f"{h:.1f}%", ha="center", va="center",
                    rotation=90, fontsize=12, color="black")
    ax.set_title(title)
    ax.set_ylabel("Participación porcentual (%)", fontsize=15)
    ax.set_xlabel("Lenguaje de programación", fontsize=15)
    ax.set_xticks(x + width * (len(years) - 1) / 2)
    ax.set_xticklabels(order, fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(0, agg["pct"].max() + 6)
    ax.legend(fontsize=13)
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

fig, ax = plt.subplots(figsize=(13, 6.5))
for lang in languages:
    sub = (trend_all[trend_all["language"] == lang]
           .set_index("quarter").reindex(quarters))
    ax.plot(quarters, sub["num_pushers"], marker="o", linewidth=2.8,
            label=lang, color=color_map[lang])
ax.set_xlim(1, 16); ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45, fontsize=13)
ax.axvline(x=12, color='red', linestyle='--', linewidth=2.5,
           label='Lanzamiento ChatGPT (Q4-2022)', zorder=2.5)
ax.set_ylim(0, 6500); ax.set_yticks(np.arange(0, 6501, 500))
ax.set_xlabel("Trimestre", fontsize=14)
ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Lenguaje de programación", ncol=2, fontsize=13, title_fontsize=14)
plt.tight_layout()
fig.savefig(p("output", "figures", "language_trend_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

df_gpt = df[df["gpt_available"] == 1]
trend_gpt = (df_gpt.groupby(["language", "quarter"], as_index=False)
                   .agg(num_pushers=("num_pushers_pc", "mean")))
fig, ax = plt.subplots(figsize=(13, 6.5))
for lang in languages:
    sub = (trend_gpt[trend_gpt["language"] == lang]
           .set_index("quarter").reindex(quarters))
    ax.plot(quarters, sub["num_pushers"], marker="o", linewidth=2.8,
            label=lang, color=color_map[lang])
ax.set_xlim(1, 16); ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45, fontsize=13)
ax.axvline(x=12, color='red', linestyle='--', linewidth=2.5,
           label='Lanzamiento ChatGPT (Q4-2022)', zorder=2.5)
ax.set_ylim(0, 70); ax.set_yticks(np.arange(0, 71, 5))
ax.set_xlabel("Trimestre", fontsize=16); ax.set_ylabel("Unique pushers por 100k hab.", fontsize=16)
ax.tick_params(axis='both', labelsize=13)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Lenguaje de programación", ncol=2, fontsize=13, title_fontsize=14)
plt.tight_layout()
fig.savefig(p("output", "figures", "language_trend_gpt_countries_2020_2023.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)

df_no_gpt = df[df["gpt_available"] == 0]
trend_no_gpt = (df_no_gpt.groupby(["language", "quarter"], as_index=False)
                          .agg(num_pushers=("num_pushers_pc", "mean")))
fig, ax = plt.subplots(figsize=(13, 6.5))
for lang in languages:
    sub = (trend_no_gpt[trend_no_gpt["language"] == lang]
           .set_index("quarter").reindex(quarters))
    ax.plot(quarters, sub["num_pushers"], marker="o", linewidth=2.8,
            label=lang, color=color_map[lang])
ax.set_xlim(1, 16); ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45, fontsize=13)
ax.axvline(x=12, color='red', linestyle='--', linewidth=2.5,
           label='Lanzamiento ChatGPT (Q4-2022)', zorder=2.5)
ax.set_ylim(0, 70); ax.set_yticks(np.arange(0, 71, 5))
ax.set_xlabel("Trimestre", fontsize=16); ax.set_ylabel("Unique pushers por 100k hab.", fontsize=16)
ax.tick_params(axis='both', labelsize=13)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Lenguaje de programación", ncol=2, fontsize=13, title_fontsize=14, loc="upper left")
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
# SECTION 4.5: Timeline of ChatGPT improvements and limitations (2020-2023)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 4.5  ChatGPT timeline figure")
print("=" * 60)

# Map calendar date to fractional year for plotting
def _date_to_x(year, month):
    return year + (month - 1) / 12.0

# Timeline window
_x_min, _x_max = _date_to_x(2020, 1), _date_to_x(2024, 1)

fig, ax = plt.subplots(figsize=(15, 7.5))

# ── Treatment start (Q4-2022) vertical band ──────────────────────────────────
ax.axvspan(_date_to_x(2022, 10), _x_max, alpha=0.06, color='red', zorder=0)
ax.axvline(_date_to_x(2022, 11), color='red', linewidth=2.5, linestyle='--',
           zorder=2, label='Lanzamiento ChatGPT (Nov 2022)')

# ── Lane 1: GitHub Copilot events (top) ──────────────────────────────────────
y_copilot = 3.2
ax.hlines(y_copilot, _date_to_x(2021, 6), _x_max, color='#5b8def',
          linewidth=4, alpha=0.55, zorder=1)
ax.plot(_date_to_x(2021, 6), y_copilot, marker='o', markersize=11,
        color='#1f4ea8', zorder=3)
ax.annotate('Copilot\n(preview)\nJun 2021',
            xy=(_date_to_x(2021, 6), y_copilot),
            xytext=(_date_to_x(2021, 6), y_copilot + 0.45),
            ha='center', fontsize=9, fontweight='bold',
            color='#1f4ea8')
ax.plot(_date_to_x(2022, 6), y_copilot, marker='o', markersize=11,
        color='#1f4ea8', zorder=3)
ax.annotate('Copilot\ncomercial\n10 USD/mes\nJun 2022',
            xy=(_date_to_x(2022, 6), y_copilot),
            xytext=(_date_to_x(2022, 6), y_copilot + 0.45),
            ha='center', fontsize=9, fontweight='bold',
            color='#1f4ea8')

# ── Lane 2: ChatGPT events (middle) ──────────────────────────────────────────
y_gpt = 2.0
ax.hlines(y_gpt, _date_to_x(2022, 11), _x_max, color='#2ca02c',
          linewidth=4, alpha=0.55, zorder=1)
ax.plot(_date_to_x(2022, 11), y_gpt, marker='*', markersize=22,
        color='#1a7a1a', zorder=3)
ax.annotate('ChatGPT (GPT-3.5)\ngratuito y masivo\nNov 2022',
            xy=(_date_to_x(2022, 11), y_gpt),
            xytext=(_date_to_x(2022, 11), y_gpt + 0.45),
            ha='center', fontsize=9.5, fontweight='bold',
            color='#1a7a1a')
ax.plot(_date_to_x(2023, 3), y_gpt, marker='*', markersize=18,
        color='#1a7a1a', zorder=3)
ax.annotate('GPT-4 vía Plus\n20 USD/mes\nMar 2023',
            xy=(_date_to_x(2023, 3), y_gpt),
            xytext=(_date_to_x(2023, 3), y_gpt + 0.45),
            ha='center', fontsize=9, fontweight='bold',
            color='#1a7a1a')
ax.plot(_date_to_x(2023, 7), y_gpt, marker='*', markersize=16,
        color='#1a7a1a', zorder=3)
ax.annotate('Code Interpreter\n+ plugins\nJul 2023',
            xy=(_date_to_x(2023, 7), y_gpt),
            xytext=(_date_to_x(2023, 7), y_gpt + 0.45),
            ha='center', fontsize=9, color='#1a7a1a')

# ── Lane 3: Limitations (bottom) ─────────────────────────────────────────────
y_lim = 0.7
# Hallucinations span: from launch through GPT-4 adoption
ax.hlines(y_lim, _date_to_x(2022, 11), _date_to_x(2023, 4),
          color='#d62728', linewidth=8, alpha=0.45, zorder=1)
ax.annotate('Alucinaciones\nfrecuentes',
            xy=(_date_to_x(2023, 1), y_lim),
            xytext=(_date_to_x(2023, 1), y_lim - 0.55),
            ha='center', fontsize=8.5, color='#a31a1a', fontweight='bold')

# Training cutoff line (Sep 2021)
ax.plot(_date_to_x(2021, 9), y_lim, marker='v', markersize=10,
        color='#a31a1a', zorder=3)
ax.annotate('Corte de\nentrenamiento\n(GPT-3.5)\nSep 2021',
            xy=(_date_to_x(2021, 9), y_lim),
            xytext=(_date_to_x(2021, 9), y_lim - 0.55),
            ha='center', fontsize=8.5, color='#a31a1a')

# Context window evolution
ax.annotate('Contexto: 4{,}096 tokens (GPT-3.5)\n$\\rightarrow$ 8{,}192--32{,}768 (GPT-4)',
            xy=(_date_to_x(2023, 8), y_lim),
            xytext=(_date_to_x(2023, 8), y_lim - 0.55),
            ha='center', fontsize=8.5, color='#a31a1a', fontweight='bold')

# ── Axis: years and quarters ────────────────────────────────────────────────
year_ticks  = [_date_to_x(y, 1) for y in (2020, 2021, 2022, 2023, 2024)]
year_labels = ['2020', '2021', '2022', '2023', '2024']
ax.set_xticks(year_ticks)
ax.set_xticklabels(year_labels, fontsize=11)

# Quarter minor grid
quarter_ticks = [_date_to_x(y, m) for y in (2020, 2021, 2022, 2023) for m in (1, 4, 7, 10)]
ax.set_xticks(quarter_ticks, minor=True)
ax.grid(which='major', axis='x', linestyle='-', alpha=0.25)
ax.grid(which='minor', axis='x', linestyle=':',  alpha=0.18)

# Lane labels
ax.text(_x_min - 0.05, y_copilot, 'GitHub\nCopilot',
        fontsize=10.5, fontweight='bold', va='center', ha='right',
        color='#1f4ea8')
ax.text(_x_min - 0.05, y_gpt, 'ChatGPT',
        fontsize=10.5, fontweight='bold', va='center', ha='right',
        color='#1a7a1a')
ax.text(_x_min - 0.05, y_lim, 'Limitaciones',
        fontsize=10.5, fontweight='bold', va='center', ha='right',
        color='#a31a1a')

# Pre-period shading (pretratamiento)
ax.axvspan(_x_min, _date_to_x(2022, 10), alpha=0.04, color='grey', zorder=0)
ax.text(_date_to_x(2021, 6), 4.3, 'Pre-tratamiento (Q1-2020 -- Q3-2022)',
        ha='center', fontsize=10, color='dimgrey', style='italic')
ax.text(_date_to_x(2023, 6), 4.3, 'Post-tratamiento (Q4-2022 -- Q4-2023)',
        ha='center', fontsize=10, color='#a31a1a', style='italic',
        fontweight='bold')

ax.set_xlim(_x_min - 0.5, _x_max + 0.05)
ax.set_ylim(-0.2, 4.6)
ax.set_yticks([])
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#888')

ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.set_title('Línea de tiempo: mejoras y limitaciones de ChatGPT (2020--2023)',
             fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
timeline_path = p("output", "figures", "chatgpt_timeline.png")
fig.savefig(timeline_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Timeline saved: {timeline_path}")


# ============================================================================
# SECTION 5: Python DID / SC / SDID estimations
# (Replaces Stata section 5 of all_code.py)
# ============================================================================

from scipy.stats import norm as _norm
from synthdid.synthdid import Synthdid

print("\n" + "=" * 60)
print("SECTION 5  DID / SC / SDID (no controls)")
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
N_BOOT_SC     = 2000   # larger for SC stability; parallelised so still fast
N_BOOT_ES     = 200    # bootstrap reps for event-study CIs (matrix ops only)
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

def _estimate_synthdid(df_lang, method, outcome_col='num_pushers_pc'):
    """
    DID, SC or SDID estimate via synthdid (d2cml-ai).
    method: 'did' | 'sc' | 'sdid'
    Returns (att, se, omega, lam, co_units_ordered)
    """
    df_s = df_lang[['iso2_code', 'quarter', outcome_col, 'gpt_available']].copy()
    df_s['treat'] = (
        (df_s['gpt_available'] == 1) & (df_s['quarter'] >= TREAT_START)
    ).astype(int)
    if method == 'sc':
        fit_kwargs = {'synth': True}
    elif method == 'did':
        fit_kwargs = {'did': True}
    else:
        fit_kwargs = {}
    estimator = (
        Synthdid(df_s, 'iso2_code', 'quarter', 'treat', outcome_col)
        .fit(**fit_kwargs)
    )
    # Fix seed per method for reproducible and distinct bootstrap SEs
    _method_seeds = {'did': 1234, 'sdid': 1235, 'sc': 1236}
    np.random.seed(_method_seeds.get(method, 1234))
    r = estimator.vcov(method='bootstrap', n_reps=N_BOOT).summary()
    att   = float(r.att)
    se    = float(r.se)
    omega = np.array(r.weights['omega'][0])
    lam   = np.array(r.weights['lambda'][0])
    all_units = list(r.Y_units[0])
    co_units_ordered = all_units[:len(omega)]
    return att, se, omega, lam, co_units_ordered


def _solve_sc_osqp(co_pre, tr_pre_mean):
    """SC QP via OSQP: min ||co_pre.T @ w - tr_pre_mean||^2  s.t. sum(w)=1, w>=0.
    co_pre: N_co × T_pre,  tr_pre_mean: T_pre vector.
    ~20-50x faster than SLSQP for this problem size.
    """
    import osqp
    import scipy.sparse as sp
    n = co_pre.shape[0]
    P = sp.csc_matrix(2.0 * co_pre @ co_pre.T)
    q = -2.0 * co_pre @ tr_pre_mean
    A = sp.csc_matrix(np.vstack([np.ones((1, n)), np.eye(n)]))
    l = np.concatenate([[1.0], np.zeros(n)])
    u = np.concatenate([[1.0], np.full(n, np.inf)])
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False,
               eps_abs=1e-9, eps_rel=1e-9, max_iter=10000, polish=True)
    return np.maximum(prob.solve().x, 0.0)


def _sc_boot_one(co_pre, co_post, tr_pre, tr_post, n_co, n_tr, seed_i):
    """Single SC bootstrap iteration. Module-level so joblib can pickle it."""
    rng     = np.random.default_rng(int(seed_i))
    ci      = rng.integers(0, n_co, size=n_co)
    ti      = rng.integers(0, n_tr, size=n_tr)
    cb_pre  = co_pre[ci]
    cb_post = co_post[ci]
    tb_pre  = tr_pre[ti].mean(0)
    tb_post = tr_post[ti].mean(0)
    w = _solve_sc_osqp(cb_pre, tb_pre)
    return float(np.mean(tb_post - cb_post.T @ w))


def _estimate_sc_stata(df_lang, outcome_col='num_pushers_pc', n_boot=N_BOOT_SC, seed=1234):
    """SC matching Stata sdid method(sc): no centering, lambda_pre=0.
    Uses OSQP (fast QP solver) + joblib parallel bootstrap for speed.
    With n_boot=2000 and 12 cores, runs faster than the old 100-rep SLSQP serial loop.
    """
    from joblib import Parallel, delayed

    co_units = sorted(df_lang[df_lang['gpt_available'] == 0]['iso2_code'].unique())
    tr_units = sorted(df_lang[df_lang['gpt_available'] == 1]['iso2_code'].unique())
    all_q    = sorted(df_lang['quarter'].unique())
    pre_q    = [q for q in all_q if q < TREAT_START]
    post_q   = [q for q in all_q if q >= TREAT_START]

    def _piv(units, quarters):
        return (df_lang[df_lang['iso2_code'].isin(units)]
                .pivot(index='iso2_code', columns='quarter', values=outcome_col)
                .reindex(index=units, columns=quarters).fillna(0).values)

    Y_co_pre  = _piv(co_units, pre_q)
    Y_co_post = _piv(co_units, post_q)
    Y_tr_pre  = _piv(tr_units, pre_q)
    Y_tr_post = _piv(tr_units, post_q)

    # Point estimate with OSQP
    omega = _solve_sc_osqp(Y_co_pre, Y_tr_pre.mean(0))
    att   = float(np.mean(Y_tr_post.mean(0) - Y_co_post.T @ omega))

    # Parallel bootstrap: each iteration is independent → embarrassingly parallel
    n_co = len(co_units)
    n_tr = len(tr_units)
    rng_main = np.random.default_rng(seed)
    seeds_i  = rng_main.integers(0, 2**31, size=n_boot)
    boot_atts = Parallel(n_jobs=-1, backend='loky')(
        delayed(_sc_boot_one)(Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
                              n_co, n_tr, int(s))
        for s in seeds_i
    )
    boot_atts = [b for b in boot_atts if np.isfinite(b)]
    se = float(np.std(boot_atts, ddof=1)) if len(boot_atts) > 1 else float('nan')

    return att, se, omega, co_units


def _quarter_structure(df_lang):
    """Return all_q, pre_q, post_q, co_units, tr_units for plot functions."""
    all_q    = sorted(df_lang['quarter'].unique())
    pre_q    = [q for q in all_q if q < TREAT_START]
    post_q   = [q for q in all_q if q >= TREAT_START]
    co_units = sorted(df_lang[df_lang['gpt_available'] == 0]['iso2_code'].unique())
    tr_units = sorted(df_lang[df_lang['gpt_available'] == 1]['iso2_code'].unique())
    return all_q, pre_q, post_q, co_units, tr_units


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

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, treated,   label='Tratado',       color='#2c7bb6', lw=2.6, marker='o', ms=6)
    ax.plot(x, synthetic, label='Control Sint.', color='#d7191c', lw=2.6,
            marker='s', ms=6, ls='--')

    onset = len(pre_q)
    ax.axvline(x=onset, color='gray', ls=':', lw=1.8)

    # SDID: shaded band of temporal (lambda) weights over pre-treatment periods.
    # Replicates the canonical synthdid plot: band height linear in lambda,
    # anchored to the data range (lambda * height/3 + base). Not drawn for DiD
    # (uniform lambda) or SC (lambda = 0).
    if method == 'sdid' and lam is not None and len(lam) == len(pre_q):
        vals   = np.concatenate([treated, synthetic])
        y_min, y_max = vals.min(), vals.max()
        height = y_max - y_min
        base   = y_min - height / 5
        band   = np.asarray(lam) * height / 3 + base
        xpre   = list(range(len(pre_q)))
        ax.fill_between(xpre, base, band, alpha=0.6, color='gray',
                        zorder=1, label='Pesos temporales λ')

    ax.set_xticks(x)
    ax.set_xticklabels(qlabels, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel("Trimestre", fontsize=15)
    ax.set_ylabel("Unique pushers por 100k hab.", fontsize=15)
    # SDID (panel e): pin legend upper-left so all figures 5-14 are consistent.
    ax.legend(fontsize=14, loc='upper left' if method == 'sdid' else 'best')
    ax.grid(axis='y', ls='--', alpha=0.4)

    stars = _stars(att, se)
    ax.set_title(
        f"{lang} – {method.upper()}   ATT = {att:.3f}{stars}   (SE = {se:.3f})",
        fontsize=14
    )
    plt.tight_layout()

    fname = p("output", "figures", f"{LANG_SAFE[lang]}{method}trends12.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _plot_weights(omega, co_units, method, lang, att=None, se=None):
    """Bubble chart of unit (omega) weights for control countries.
    Style: large blue circles for high-weight countries, small red dots
    for negligible-weight countries, matching the reference scatter style.
    """
    from matplotlib.lines import Line2D

    n       = len(co_units)
    uniform = 1.0 / n

    # Sort alphabetically so x-axis is consistent across methods
    idx    = np.argsort(co_units)
    labels = [co_units[i] for i in idx]
    vals   = np.array([omega[i] for i in idx])

    # Classify: significant (≥ 30 % of uniform weight) vs negligible
    threshold = uniform * 0.30
    colors = ['#2c7bb6' if v >= threshold else '#d7191c' for v in vals]

    # Bubble size proportional to weight (min 20 for visibility)
    max_w  = max(vals) if max(vals) > 0 else 1.0
    sizes  = [max(45, (v / max_w) * 520) for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.scatter(range(n), vals, c=colors, s=sizes,
               alpha=0.85, edgecolors='white', linewidth=0.6, zorder=3)

    # Reference lines (style matches the reference image)
    ax.axhline(y=0,       color='#2c7bb6', linewidth=1.2, alpha=0.7, zorder=2)
    ax.axhline(y=uniform, color='purple',  linewidth=1.2, linestyle='--',
               alpha=0.8, zorder=2)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=75, fontsize=10, ha='right')
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylabel("Peso ω", fontsize=15)
    meth_label = {'did': 'DiD', 'sc': 'CS', 'sdid': 'SDiD'}.get(method, method.upper())
    ax.set_title(f"{lang} — {meth_label}   Pesos del control sintético", fontsize=14)
    ax.grid(axis='y', ls='--', alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='purple', linewidth=1.2, linestyle='--',
               label=f'Peso uniforme: {uniform:.3f}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c7bb6',
               markersize=13, label='Peso significativo'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d7191c',
               markersize=7,  label='Peso negligible'),
    ]
    if att is not None and not np.isnan(se if se is not None else float('nan')):
        legend_elements.insert(0,
            Line2D([0], [0], color='w', label=f'ATT: {att:.3f}'))
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right',
              framealpha=0.9)

    plt.tight_layout()
    fname = p("output", "figures", f"{LANG_SAFE[lang]}{method}weights12.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _sdid_event_study(df_lang, omega, co_units, outcome_col='num_pushers_pc'):
    """
    Period-by-period gap for SDID event study using fixed omega weights.
    Bootstraps treated units (no SDID re-fit) for 95% CI bands.
    Returns: rel_periods, gap_est, lower_ci, upper_ci
    """
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

    Y_co = _piv(co_units)          # (N_co, T)
    Y_tr = _piv(tr_units)          # (N_tr, T)
    synthetic = omega @ Y_co       # (T,)

    def _gap(Y_tr_):
        g = Y_tr_.mean(axis=0) - synthetic
        g = g - g[:T0].mean()      # normalise: pre-treatment mean → 0
        return g

    gap_est = _gap(Y_tr)

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
    """
    Event study plot: SDID gap per period with 95% CI band.
    Pre-treatment periods hovering near zero validates the SDID counterfactual.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.array(rel)
    ax.fill_between(x, lower, upper, alpha=0.20, color='steelblue', label='IC 95%')
    ax.plot(x, gap, marker='o', color='steelblue', lw=2.4, ms=7, label='Gap SDID')
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6)
    ax.axvline(0, color='red',   lw=2.0, ls='--', label='Inicio trat. (Q4-2022)')
    if any(r < 0 for r in rel):
        ax.axvspan(min(x) - 0.5, 0, alpha=0.06, color='grey')
    ax.set_xticks(x)
    xlabels = [str(r) if i % 2 == 0 else '' for i, r in enumerate(rel)]
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel('Trimestres relativos al tratamiento (0 = Q4-2022)', fontsize=14)
    ax.set_ylabel('Gap (Tratado \u2212 Control Sint\u00e9tico)', fontsize=14)
    ax.set_title(f'{lang} \u2014 An\u00e1lisis de evento SDID (pre-tendencias)', fontsize=15)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(axis='y', ls='--', alpha=0.3)
    plt.tight_layout()
    fname = p("output", "figures", f"{LANG_SAFE[lang]}sdid_event_study.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def _fmt_es(x, n=3):
    """Format float with comma as decimal separator (Spanish convention).
    The ``{,}`` token keeps proper LaTeX math spacing.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:.{n}f}".replace(".", "{,}")


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
            f"{tname} & {_fmt_es(b1)}{s1} & {_fmt_es(b2)}{s2} & {_fmt_es(b3)}{s3}"
            f" & {n} & {_fmt_es(cm)} \\\\"
        )
        lines.append(
            f"              & ({_fmt_es(e1)}) & ({_fmt_es(e2)}) & ({_fmt_es(e3)}) & & \\\\"
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
    cmean = float(df_l.loc[df_l['quarter'] < TREAT_START, 'num_pushers_pc'].mean())
    all_q, pre_q, post_q, co_units, tr_units = _quarter_structure(df_l)

    row: dict = {'nobs': n_obs, 'cmean': cmean}
    omega_sdid, co_sdid = None, None

    # ── DID / SDID: synthdid (d2cml-ai) ─────────────────────────────────
    for method in ('did', 'sdid'):
        try:
            att, se, omega, lam, co_sd = _estimate_synthdid(df_l, method)
        except Exception as exc:
            print(f"    {method.upper()}: estimation failed ({exc})")
            att, se = 0.0, float('nan')
            omega, lam, co_sd = (np.ones(len(co_units)) / len(co_units),
                                 np.ones(len(pre_q)) / len(pre_q),
                                 list(co_units))
        row[method] = (att, se)
        print(f"    {method.upper()}: ATT = {att:.3f}  SE = {se:.3f}  {_stars(att, se)}")
        _plot_trends(df_l, omega, lam, att, se, method, lang,
                     all_q, pre_q, post_q, co_sd, tr_units)
        _plot_weights(omega, co_sd, method, lang, att=att, se=se)
        if method == 'sdid':
            omega_sdid, co_sdid = omega, co_sd

    # ── SC: uncentered (Stata-compatible, no demeaning) ───────────────
    try:
        att_sc, se_sc, omega_sc, co_sc = _estimate_sc_stata(df_l)
    except Exception as exc:
        print(f"    SC: estimation failed ({exc})")
        att_sc, se_sc = 0.0, float('nan')
        omega_sc, co_sc = (np.ones(len(co_units)) / len(co_units),
                           list(co_units))
    row['sc'] = (att_sc, se_sc)
    print(f"    SC: ATT = {att_sc:.3f}  SE = {se_sc:.3f}  {_stars(att_sc, se_sc)}")
    lam_dummy = np.ones(len(pre_q)) / len(pre_q)
    _plot_trends(df_l, omega_sc, lam_dummy, att_sc, se_sc, 'sc', lang,
                 all_q, pre_q, post_q, co_sc, tr_units)
    _plot_weights(omega_sc, co_sc, 'sc', lang, att=att_sc, se=se_sc)

    # ── SDID event study (pre-trend validation) ───────────────────────────
    if omega_sdid is not None:
        try:
            rel, gap, lo, hi = _sdid_event_study(df_l, omega_sdid, co_sdid)
            _plot_event_study(lang, rel, gap, lo, hi)
            print(f"    Event study: saved")
        except Exception as exc:
            print(f"    Event study: failed ({exc})")

    results_5[lang] = row

# ── LaTeX table (no controls) ─────────────────────────────────────────────────

NOTE_5 = (
    r"Estimaciones del efecto promedio del tratamiento (ATT) del acceso a ChatGPT "
    r"sobre el n\'{u}mero de \textit{unique pushers} por cada 100\,000 habitantes. "
    r"Las columnas DID, SC y SDID corresponden a Diferencias en Diferencias, "
    r"Control Sint\'{e}tico (Abadie et al., 2010) y Diferencias en Diferencias "
    r"Sint\'{e}tica (Arkhangelsky et al., 2021), respectivamente, todos estimados "
    r"mediante la librer\'{i}a \texttt{synthdid} de Python (Clarke et al., 2023). "
    r"Los errores est\'{a}ndar se obtienen por \textit{bootstrap} (100 replicaciones). "
    r"El periodo de tratamiento inicia en Q4-2022, coincidiendo con el lanzamiento de ChatGPT. "
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
    print(f"  {_col}: {_n} valores faltantes -> 0")

df_merged.to_csv(p("output", "data", "merge_controles.csv"), index=False)
print(f"Saved: {p('output', 'data', 'merge_controles.csv')}")

# ── 6.6 covariate partial-out ─────────────────────────────────────────────────

COVARIATES = ["uso_computadoras", "uso_internet"]


def _partial_out(df_lang, covariates=COVARIATES):
    """Partial out covariates matching Stata sdid covariates(, projected).

    Stata projected(): OLS with explicit unit + time FE dummies on control
    units across ALL periods, then subtracts X*beta from the full panel.
    """
    # Control units only (gpt_available == 0), all periods -- matches Stata
    df_co = df_lang[df_lang['gpt_available'] == 0].copy()
    df_co = df_co.dropna(subset=list(covariates))

    y = df_co['num_pushers_pc'].values
    X_cov = df_co[list(covariates)].values

    # Unit and time FE dummies (drop one to avoid multicollinearity)
    unit_dummies = pd.get_dummies(df_co['iso2_code'], drop_first=True).astype(float).values
    time_dummies = pd.get_dummies(df_co['quarter'],   drop_first=True).astype(float).values

    # Design matrix: [covariates | time FE | unit FE | constant]
    X_full = np.column_stack([X_cov, time_dummies, unit_dummies,
                               np.ones(len(y))])

    # OLS -- extract only the covariate betas (first len(covariates) cols)
    beta_full, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    beta = beta_full[:len(covariates)]

    df_adj = df_lang.copy()
    df_adj['num_pushers_pc'] = (df_lang['num_pushers_pc'].values
                                - df_lang[list(covariates)].values @ beta)
    return df_adj


# ── 6.7 estimation loop (with controls) ──────────────────────────────────────

df_ctrl_panel = df_merged[df_merged['iso2_code'] != 'HK'].copy()

results_6 = {}

for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l     = df_ctrl_panel[df_ctrl_panel['language'] == lang].copy()
    df_l_adj = _partial_out(df_l)
    n_obs    = len(df_l)

    all_q, pre_q, post_q, co_units, tr_units = _quarter_structure(df_l_adj)

    cmean = float(df_l.loc[df_l['quarter'] < TREAT_START, 'num_pushers_pc'].mean())

    row: dict = {'nobs': n_obs, 'cmean': cmean}

    # ── DID / SDID: synthdid (d2cml-ai) ─────────────────────────────────
    for method in ('did', 'sdid'):
        try:
            att, se, omega, lam, co_sd = _estimate_synthdid(df_l_adj, method)
        except Exception as exc:
            print(f"    {method.upper()}: estimation failed ({exc})")
            att, se = 0.0, float('nan')
            omega = np.ones(len(co_units)) / len(co_units)
            lam   = np.ones(len(pre_q)) / len(pre_q)
            co_sd = list(co_units)
        row[method] = (att, se)
        print(f"    {method.upper()}: ATT = {att:.3f}  SE = {se:.3f}  {_stars(att, se)}")

    # ── SC: uncentered (Stata-compatible, no demeaning) ───────────────
    try:
        att_sc, se_sc, omega_sc, co_sc = _estimate_sc_stata(df_l_adj)
    except Exception as exc:
        print(f"    SC: estimation failed ({exc})")
        att_sc, se_sc = 0.0, float('nan')
        omega_sc, co_sc = (np.ones(len(co_units)) / len(co_units),
                           list(co_units))
    row['sc'] = (att_sc, se_sc)
    print(f"    SC: ATT = {att_sc:.3f}  SE = {se_sc:.3f}  {_stars(att_sc, se_sc)}")

    results_6[lang] = row

# ── 6.8 LaTeX table (with controls) ──────────────────────────────────────────

NOTE_6 = (
    r"Estimaciones del efecto promedio del tratamiento (ATT) del acceso a ChatGPT "
    r"sobre el n\'{u}mero de \textit{unique pushers} por cada 100\,000 habitantes. "
    r"Las columnas DID, SC y SDID corresponden a Diferencias en Diferencias, "
    r"Control Sint\'{e}tico (Abadie et al., 2010) y Diferencias en Diferencias "
    r"Sint\'{e}tica (Arkhangelsky et al., 2021), respectivamente, todos estimados "
    r"mediante la librer\'{i}a \texttt{synthdid} de Python (Clarke et al., 2023). "
    r"Se incluyen como variables de control el porcentaje de individuos que usan "
    r"computadora y el porcentaje que usa internet, extra\'{i}das del Banco Mundial, "
    r"incorporadas mediante un \textit{partial-out} de efectos fijos de unidad y "
    r"tiempo. Los errores est\'{a}ndar se obtienen por \textit{bootstrap} (100 replicaciones). "
    r"El periodo de tratamiento inicia en Q4-2022. "
    r"Fuentes: GitHub Innovation Graph "
    r"(\url{https://github.com/github/innovationgraph}); Banco Mundial, Indicadores "
    r"de Desarrollo Mundial. Elaboraci\'{o}n propia. Errores est\'{a}ndar entre "
    r"par\'{e}ntesis. * p<0.10, ** p<0.05, *** p<0.01"
)

_write_latex_table(
    results_6,
    p("output", "tables", "gpt_impact_github_DataScience_controls.tex"),
    r"Impacto de ChatGPT en el n\'{u}mero de programadores (con controles)",
    "tab:tabla4",
    NOTE_6,
)


# ============================================================================
# SECTION 7: Robustness — SDID with restricted control group
# (Excludes 6 severe internet censors: CN, CU, IR, BY, RU, SY)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 7  Robustness: SDID restricted control group")
print("=" * 60)

SEVERE_CENSORS = ['CN', 'CU', 'IR', 'BY', 'RU', 'SY']

results_7 = {}

for lang in LANGUAGES_5:
    print(f"\n  [{lang}]")
    df_l = df_panel[df_panel['language'] == lang].copy()
    n_obs_full = len(df_l)

    # Baseline mean (control group, pre-treatment) — full sample
    cmean = float(df_l.loc[
        (df_l['quarter'] < TREAT_START) & (df_l['gpt_available'] == 0),
        'num_pushers_pc'
    ].mean())

    # --- SDID full (same as Section 5) ---
    att_full, se_full = results_5[lang]['sdid']
    print(f"    SDID (full):       ATT = {att_full:.3f}  SE = {se_full:.3f}  {_stars(att_full, se_full)}")

    # --- SDID restricted (exclude severe censors) ---
    df_r = df_l[~df_l['iso2_code'].isin(SEVERE_CENSORS)].copy()
    n_obs_rest = len(df_r)
    try:
        att_r, se_r, omega_r, lam_r, co_r = _estimate_synthdid(df_r, 'sdid')
    except Exception as exc:
        print(f"    SDID (restricted): FAILED ({exc})")
        att_r, se_r = float('nan'), float('nan')
    print(f"    SDID (restricted): ATT = {att_r:.3f}  SE = {se_r:.3f}  {_stars(att_r, se_r)}")

    results_7[lang] = {
        'att_full': att_full, 'se_full': se_full,
        'att_rest': att_r,    'se_rest': se_r,
        'n_full': n_obs_full, 'n_rest': n_obs_rest,
        'cmean': cmean,
    }

# ── Robustness LaTeX table (Cuadro 5) ─────────────────────────────────────

NOTE_7 = (
    r"Estimaciones del efecto promedio del tratamiento (ATT) del acceso a ChatGPT "
    r"sobre el n\'{u}mero de \textit{unique pushers} por cada 100\,000 habitantes, "
    r"mediante Diferencias en Diferencias Sint\'{e}tica (SDID; Arkhangelsky et al., 2021), "
    r"estimada con la librer\'{i}a \texttt{synthdid} de Python (Clarke et al., 2023). "
    r"La columna ``SDID (completo)'' emplea los 29 pa\'{i}ses de control y corresponde "
    r"a la estimaci\'{o}n principal; la columna ``SDID (restringido)'' excluye del grupo "
    r"de control los seis censores severos de internet: China (CN), Cuba (CU), "
    r"Ir\'{a}n (IR), Bielorrusia (BY), Rusia (RU) y Siria (SY), clasificados como "
    r"``No Libres'' por Freedom House (Freedom on the Net, 2022). "
    r"Los errores est\'{a}ndar se obtienen por \textit{bootstrap} (100 replicaciones). "
    r"El periodo de tratamiento inicia en Q4-2022. "
    r"Fuente: GitHub Innovation Graph "
    r"(\url{https://github.com/github/innovationgraph}). "
    r"Elaboraci\'{o}n propia. Errores est\'{a}ndar entre par\'{e}ntesis. "
    r"* p$<$0.10, ** p$<$0.05, *** p$<$0.01"
)

rob_lines = [
    r"\begin{table}[htbp]\centering",
    r"\caption{Robustez: SDID con grupo de control restringido (excluye censores severos)}",
    r"\label{tab:tabla5}",
    r"\begin{threeparttable}",
    r"{\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Lenguaje & \shortstack{SDID \\ (completo)} & \shortstack{SDID \\ (restringido)}"
    r" & Obs. pleno & Obs. restringido & \shortstack{Baseline \\ Mean} \\",
    r"\midrule",
]

for lang in LANGUAGES_5:
    r7 = results_7[lang]
    tname = LANG_TEX[lang]
    s_f = _stars(r7['att_full'], r7['se_full'])
    s_r = _stars(r7['att_rest'], r7['se_rest'])
    rob_lines.append(
        f"{tname} & {_fmt_es(r7['att_full'])}{s_f} & {_fmt_es(r7['att_rest'])}{s_r}"
        f" & {r7['n_full']} & {r7['n_rest']} & {_fmt_es(r7['cmean'])} \\\\"
    )
    rob_lines.append(
        f"              & ({_fmt_es(r7['se_full'])}) & ({_fmt_es(r7['se_rest'])}) & & & \\\\"
    )
    rob_lines.append(r"\addlinespace")

rob_lines += [
    r"\bottomrule",
    r"\end{tabular}}",
    r"\begin{tablenotes}",
    r"\footnotesize",
    rf"\item \textit{{Nota.}} {NOTE_7}",
    r"\end{tablenotes}",
    r"\end{threeparttable}",
    r"\end{table}",
    "",
]

rob_path = p("output", "tables", "gpt_impact_github_robustez.tex")
with open(rob_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(rob_lines))
print(f"  Robustness table written: {rob_path}")


# ============================================================================
# SECTION 8: Robustness — In-space placebos + LOO donor pool sensitivity
# (Addresses jury comments: Abadie 2021, Arkhangelsky et al. 2021)
# ============================================================================

print("\n" + "=" * 60)
print("SECTION 8  Robustness: in-space placebos + LOO donor pool")
print("=" * 60)


def _sdid_point_only(df_lang, outcome_col='num_pushers_pc'):
    """SDID point estimate using estimator.att directly (skips bootstrap)."""
    df_s = df_lang[['iso2_code', 'quarter', outcome_col, 'gpt_available']].copy()
    df_s['treat'] = (
        (df_s['gpt_available'] == 1) & (df_s['quarter'] >= TREAT_START)
    ).astype(int)
    estimator = Synthdid(df_s, 'iso2_code', 'quarter', 'treat', outcome_col).fit()
    return float(estimator.att)


# ── 8.1 In-space placebos ─────────────────────────────────────────────────────
# For each control country, assign it a placebo treatment and re-run SDID on
# the control group only. The observed ATT is benchmarked against this
# distribution to derive a non-parametric p-value (Abadie, 2010, 2021).

print("\n  [8.1] In-space placebos (Abadie 2010 style)")
placebo_results = {}

for lang in LANGUAGES_5:
    df_l = df_panel[df_panel['language'] == lang].copy()
    control_units = sorted(df_l[df_l['gpt_available'] == 0]['iso2_code'].unique())

    placebo_atts = []
    fails = 0
    for placebo_iso in control_units:
        df_p = df_l[df_l['gpt_available'] == 0].copy()
        df_p['gpt_available'] = (df_p['iso2_code'] == placebo_iso).astype(int)
        try:
            att_p = _sdid_point_only(df_p)
            if np.isfinite(att_p):
                placebo_atts.append((placebo_iso, att_p))
            else:
                fails += 1
        except Exception:
            fails += 1

    att_obs = results_5[lang]['sdid'][0]
    placebo_arr = np.array([a for _, a in placebo_atts])
    p_value = (float(np.mean(np.abs(placebo_arr) >= abs(att_obs)))
               if placebo_arr.size > 0 else float('nan'))

    placebo_results[lang] = {
        'att_obs':       att_obs,
        'placebos':      placebo_atts,
        'placebo_mean':  float(np.mean(placebo_arr)) if placebo_arr.size else float('nan'),
        'placebo_std':   float(np.std(placebo_arr, ddof=1)) if placebo_arr.size > 1 else float('nan'),
        'p_value':       p_value,
        'n_placebos':    int(placebo_arr.size),
        'n_fails':       fails,
    }
    print(f"    {lang:<12s} obs={att_obs:7.3f}  placebo mean={placebo_results[lang]['placebo_mean']:7.3f}"
          f"  p={p_value:.3f}  (n={placebo_arr.size}, fails={fails})")


# ── 8.2 Leave-one-out donor pool sensitivity ─────────────────────────────────
# Drop one control country at a time, re-estimate SDID with the remaining 28
# donors plus the full set of 130 treated countries. Checks robustness of the
# headline ATT against the choice of donor pool.

print("\n  [8.2] Leave-one-out donor pool sensitivity")
loo_results = {}

for lang in LANGUAGES_5:
    df_l = df_panel[df_panel['language'] == lang].copy()
    control_units = sorted(df_l[df_l['gpt_available'] == 0]['iso2_code'].unique())

    loo_atts = []
    fails = 0
    for drop_iso in control_units:
        df_loo = df_l[df_l['iso2_code'] != drop_iso].copy()
        try:
            att_loo = _sdid_point_only(df_loo)
            if np.isfinite(att_loo):
                loo_atts.append((drop_iso, att_loo))
            else:
                fails += 1
        except Exception:
            fails += 1

    att_obs = results_5[lang]['sdid'][0]
    loo_arr = np.array([a for _, a in loo_atts])

    loo_results[lang] = {
        'att_obs':    att_obs,
        'loo':        loo_atts,
        'loo_min':    float(np.min(loo_arr))    if loo_arr.size else float('nan'),
        'loo_max':    float(np.max(loo_arr))    if loo_arr.size else float('nan'),
        'loo_median': float(np.median(loo_arr)) if loo_arr.size else float('nan'),
        'loo_std':    float(np.std(loo_arr, ddof=1)) if loo_arr.size > 1 else float('nan'),
        'n_loo':      int(loo_arr.size),
        'n_fails':    fails,
    }
    print(f"    {lang:<12s} obs={att_obs:7.3f}  LOO[{loo_results[lang]['loo_min']:7.3f},"
          f" {loo_results[lang]['loo_max']:7.3f}]  med={loo_results[lang]['loo_median']:7.3f}"
          f"  (n={loo_arr.size}, fails={fails})")


# ── 8.3 LaTeX summary table (Cuadro 6) ────────────────────────────────────────

NOTE_8 = (
    r"Pruebas de robustez del estimador SDID. \textit{Placebos in-space}: para "
    r"cada uno de los 29 pa\'{i}ses del grupo de control se le asigna un "
    r"tratamiento placebo y se reestima el SDID restringiendo la muestra al "
    r"grupo de control; el p-valor placebo es la fracci\'{o}n de placebos cuyo "
    r"$|ATT|$ es mayor o igual al $|ATT|$ observado, una prueba no param\'{e}trica "
    r"\textit{a la} Abadie (2010, 2021). \textit{Donor pool LOO}: se elimina un "
    r"pa\'{i}s de control a la vez del pool de donantes y se reestima el SDID "
    r"con los 28 pa\'{i}ses restantes m\'{a}s los 130 pa\'{i}ses tratados; las "
    r"columnas reportan el m\'{i}nimo, mediana y m\'{a}ximo de las 29 estimaciones "
    r"\textit{leave-one-out} (Arkhangelsky et al., 2021). Fuente: GitHub "
    r"Innovation Graph (\url{https://github.com/github/innovationgraph}). "
    r"Elaboraci\'{o}n propia."
)

placebo_lines = [
    r"\begin{table}[htbp]\centering",
    r"\caption{Robustez SDID: placebos in-space y sensibilidad del pool de donantes (LOO)}",
    r"\label{tab:tabla6}",
    r"\begin{threeparttable}",
    r"\small",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r" & ATT & Media & p-valor & LOO & LOO & LOO \\",
    r"Lenguaje & observado & placebos & placebo & m\'{i}n. & mediana & m\'{a}x. \\",
    r"\midrule",
]

for lang in LANGUAGES_5:
    tname = LANG_TEX[lang]
    pr = placebo_results[lang]
    lr = loo_results[lang]
    placebo_lines.append(
        f"{tname} & {_fmt_es(pr['att_obs'])} & {_fmt_es(pr['placebo_mean'])}"
        f" & {_fmt_es(pr['p_value'])} & {_fmt_es(lr['loo_min'])}"
        f" & {_fmt_es(lr['loo_median'])} & {_fmt_es(lr['loo_max'])} \\\\"
    )

placebo_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{tablenotes}",
    r"\footnotesize",
    rf"\item \textit{{Nota.}} {NOTE_8}",
    r"\end{tablenotes}",
    r"\end{threeparttable}",
    r"\end{table}",
    "",
]

pl_path = p("output", "tables", "gpt_impact_github_placebo_loo.tex")
with open(pl_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(placebo_lines))
print(f"\n  Placebo/LOO table written: {pl_path}")


# ── 8.4 Forest-plot figure: observed ATT vs LOO range vs placebo IC95 ─────────

print("  [8.4] Robustness forest plot")

fig, ax = plt.subplots(figsize=(11, 7))
y_pos = np.arange(len(LANGUAGES_5))[::-1]

from matplotlib.lines import Line2D as _L2D

for i, lang in enumerate(LANGUAGES_5):
    y = y_pos[i]
    pr = placebo_results[lang]
    lr = loo_results[lang]

    placebo_arr = np.array([a for _, a in pr['placebos']])
    if placebo_arr.size > 0:
        plac_lo, plac_hi = np.percentile(placebo_arr, [2.5, 97.5])
        ax.plot([plac_lo, plac_hi], [y, y], color='lightgrey', linewidth=9,
                solid_capstyle='round', alpha=0.75, zorder=1)

    if not (np.isnan(lr['loo_min']) or np.isnan(lr['loo_max'])):
        ax.plot([lr['loo_min'], lr['loo_max']], [y, y], color='steelblue',
                linewidth=3.5, solid_capstyle='round', zorder=2)
    if not np.isnan(lr['loo_median']):
        ax.plot(lr['loo_median'], y, marker='|', color='steelblue',
                markersize=14, markeredgewidth=2.5, zorder=3)

    ax.plot(pr['att_obs'], y, marker='o', color='red', markersize=10,
            markeredgecolor='black', markeredgewidth=0.8, zorder=4)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(LANGUAGES_5)
ax.set_xlabel('ATT (unique pushers por 100 mil habitantes)', fontsize=11)
ax.set_title('Robustez SDID: ATT observado, rango LOO y banda placebo 95 %',
             fontsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.4)

legend_elements = [
    _L2D([0], [0], marker='o', color='w', markerfacecolor='red',
         markeredgecolor='black', markersize=9, label='ATT observado (SDID)'),
    _L2D([0], [0], color='steelblue', lw=3.5, label='Rango LOO (mín--máx)'),
    _L2D([0], [0], color='lightgrey', lw=9, alpha=0.75, label='Placebo IC 95%'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()
forest_path = p("output", "figures", "robustness_forest_plot.png")
fig.savefig(forest_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Forest plot saved: {forest_path}")


# ── 8.5 Per-language placebo distribution panel ───────────────────────────────
# 5x2 grid: histogram of placebo ATTs + vertical line for observed ATT.

print("  [8.5] Per-language placebo distribution panel")

fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=False)
axes = axes.ravel()
for i, lang in enumerate(LANGUAGES_5):
    ax_i = axes[i]
    pr = placebo_results[lang]
    placebo_arr = np.array([a for _, a in pr['placebos']])
    if placebo_arr.size > 0:
        ax_i.hist(placebo_arr, bins=15, color='lightgrey', edgecolor='dimgrey',
                  alpha=0.85, zorder=1)
    ax_i.axvline(pr['att_obs'], color='red', linewidth=2.2, zorder=3,
                 label=f"ATT obs.: {pr['att_obs']:.3f}")
    ax_i.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
    ax_i.set_title(f"{lang}   p-val = {pr['p_value']:.3f}", fontsize=10)
    ax_i.set_xlabel('ATT placebo', fontsize=8)
    ax_i.set_ylabel('Frecuencia', fontsize=8)
    ax_i.tick_params(labelsize=8)
    ax_i.legend(fontsize=8, loc='upper right')
    ax_i.grid(axis='y', linestyle='--', alpha=0.3)

plt.suptitle('Distribución de ATT placebo — prueba de Abadie por lenguaje',
             fontsize=13, y=1.0)
plt.tight_layout()
panel_path = p("output", "figures", "placebo_distribution_panel.png")
fig.savefig(panel_path, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"  Placebo panel saved: {panel_path}")


print("\n" + "=" * 60)
print("all_code_python.py completed successfully.")
print("=" * 60)
