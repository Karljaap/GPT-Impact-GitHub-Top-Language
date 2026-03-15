# -*- coding: utf-8 -*-
"""
all_code.py

Combined file containing:
  1. clean_language_data_science.py
  2. language_distribution_charts.py
  3. programming_language_trends.py
  4. chatgpt_global_availability_map.py
  5. analysis_per_lang_thesis_sub_esp.do  (executed via subprocess)
"""

import os

# Root of the project (one level above the code/ folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def p(*parts):
    """Build an absolute path from BASE_DIR."""
    return os.path.join(BASE_DIR, *parts)


# ============================================================================
# SECTION 1: clean_language_data_science.py
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pycountry
from countryinfo import CountryInfo as CInfo
import matplotlib.pyplot as plt
import  altair  as  alt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Suprimir todos los FutureWarning
warnings.simplefilter('ignore', FutureWarning)

"""## 2. Data Loading & Exploratory Analysis"""

data = pd.read_csv("https://raw.githubusercontent.com/github/innovationgraph/main/data/languages.csv", delimiter=',')

# Drop colums
data = data.drop(columns=["language_type"])

# Filter EU
data = data[data.iso2_code != "EU"]

# Filter XK = Kosovo
data = data[data.iso2_code != "XK"]

# This will give you the count of rows with NaN values.
nan_rows_count = data.isna().any(axis=1).sum()
print(f"There are {nan_rows_count} rows with NaN values in the dataset.")

data[data["iso2_code"].isnull()] = "NA"

# Keep the 10 most used programming languages
top_program_lang = programming_languages = [
    "C",
    "C#",
    "C++",
    "Go",
    "Java",
    "JavaScript",
    "PHP",
    "Python",
    "Ruby",
    "TypeScript"
]

data_filter = data[data['language'].isin(top_program_lang)]
data_filter = data_filter.reset_index(drop=True)

data_filter['year_quarter'] = data_filter['year'].astype(str) + '-Q' + data_filter['quarter'].astype(str)
# We reset the index
data_filter = data_filter.reset_index(drop=True)
# Creating a unique identifier
data_filter['unique_id'] = data_filter['iso2_code'] + '-' + data_filter['language']
data_filter

"""## 3. Balanced Panel Data"""

# Create a DataFrame of unique identifiers
iso2_code = pd.DataFrame({'iso2_code': data_filter['iso2_code'].unique()})

# Create a DataFrame of unique identifiers for languages
language = pd.DataFrame({'language': data_filter['language'].unique()})

# Create a DataFrame of all time periods
# time_periods = pd.DataFrame({'year_quarter': range(data_filter['year_quarter'].min(), data_filter['year_quarter'].max() + 1)}
year_quarter = pd.DataFrame({'year_quarter': data_filter['year_quarter'].unique()})

# Create the Cartesian product of unique_ids and time_periods
balanced_panel = iso2_code.merge(language, how='cross').merge(year_quarter, how='cross')

balanced_panel["unique_id"] = balanced_panel["iso2_code"] + "-" +balanced_panel["language"]

# # Merge the balanced panel with the original data
balanced_df = balanced_panel.merge(data_filter, on=['unique_id', 'year_quarter'], how='left')

# # Merge the DataFrames with suffixes
balanced_df = balanced_panel.merge(data_filter, on=['unique_id', 'year_quarter'], how='left', suffixes=('', '_y'))

# # Now, drop the columns with '_y' suffix, which are from the right DataFrame
balanced_df = balanced_df.loc[:, ~balanced_df.columns.str.endswith('_y')]
balanced_df

# Function to convert quarter format to integer
def quarter_to_int(quarter_string):
    year, q = quarter_string.split('-')
    year = int(year)
    quarter_number = int(q[1])  # Q1, Q2, Q3, Q4 -> 1, 2, 3, 4
    base_year = 2020  # Adjust based on your balanced_df, or set dynamically
    return 4 * (year - base_year) + quarter_number

# Applying the function
balanced_df['quarter'] = balanced_df['year_quarter'].apply(quarter_to_int)
balanced_df['year'] = balanced_df['year_quarter'].str.split('-').str[0]
balanced_df.loc[balanced_df["num_pushers"].isnull(), "num_pushers"] = 0
balanced_df

"""
## 4. Preprocessing & SDiD Variable Creation"""

# Define a function that converts country names to ISO2 codes
def country_to_iso2(country_name):
    try:
        # Attempt to get the country's ISO2 code using pycountry
        return pycountry.countries.get(name=country_name).alpha_2
    except AttributeError:
        try:
            # Handle special cases where the country name doesn't exactly match pycountry's database
            special_cases = {
                "Czechia (Czech Republic)": "CZ",
                "Congo (Congo-Brazzaville)": "CG",
                "Holy See": "VA",
                "Timor-Leste (East Timor)": "TL",
                "Ukraine (with certain exceptions)": "UA",
                "Taiwan": "TW",
                "Bolivia": "BO",
                "Tanzania": "TZ",
                "South Korea": "KR",
                "Moldova": "MD",
                "Brunei": "BN"
            }
            return special_cases[country_name]
        except KeyError:
            return None

# Create a list of countries and get their ISO2 codes using the country_to_iso2 function
gpt_countries_list = [
    "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bangladesh", "Barbados", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Cabo Verde", "Canada",
    "Chile", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica", "Côte d'Ivoire", "Croatia", "Cyprus",
    "Czechia", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Estonia", "Fiji",
    "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
    "Guinea-Bissau", "Guyana", "Haiti", "Holy See", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait",
    "Kyrgyzstan", "Latvia", "Lebanon", "Lesotho", "Liberia", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway",
    "Oman", "Pakistan", "Palau", "Palestine, State of", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Romania", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Suriname", "Sweden", "Switzerland", "Taiwan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Tuvalu", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Vanuatu", "Zambia"
]

gpt_countries_iso = [country_to_iso2(country) for country in gpt_countries_list]

# Add a new column 'gpt_available' with value 1 if the iso2_code is in gpt_countries_iso, otherwise 0
balanced_df["gpt_available"] = balanced_df["iso2_code"].apply(lambda row: 1 if row in gpt_countries_iso else 0)

# Get unique ISO2 country codes from the dataset
countries = data.iso2_code.unique()

def create_populations_dictionary():
    # Initialize the dictionary and define fallback values for known problematic codes
    country_populations = {}
    special_cases = {"MM": 54688774, "PS": 5483450, "ME": 602445, "AD":79824}
    # special_cases = {}
    for country in countries:
        try:
            # Try retrieving population by ISO2 code directly via CInfo
            country_populations.update({country: CInfo(country).info()["population"]})
        except KeyError:
            try:
                # Fallback: look up the official country name, then fetch population
                fallback_name = pycountry.countries.lookup(country).name
                country_populations.update({country: CInfo(fallback_name).info()["population"]})
            except KeyError:
                # As a last resort, use the hard-coded special case value
                print(country)
                country_populations.update({country: special_cases[country]})

    return country_populations

country_populations = create_populations_dictionary()

# Create necessary variables
balanced_df["population"] = balanced_df["iso2_code"].map(country_populations)
# balanced_df["num_pushers_pc"] = (balanced_df["num_pushers"] / balanced_df["population"])*100000
balanced_df.loc[:, "num_pushers_pc"] = (balanced_df["num_pushers"] / balanced_df["population"] * 100000).replace([np.inf, -np.inf], 0).fillna(0).astype("float64")
# balanced_df['gpt_available_post1'] = (balanced_df['quarter'] >= 12).astype(int)
balanced_df.loc[:, "post1"] = (balanced_df["quarter"] >= 12).astype("int8")
balanced_df.loc[:, "post2"] = (balanced_df["quarter"] >= 13).astype("int8")
balanced_df.loc[:, "gpt_available_post1"] = (balanced_df["gpt_available"] & balanced_df["post1"]).astype("int8")
balanced_df.loc[:, "gpt_available_post2"] = (balanced_df["gpt_available"] & balanced_df["post2"]).astype("int8")
balanced_df["Treatment"] = (balanced_df["gpt_available_post1"] * balanced_df["post1"]).astype("int8")
# balanced_df.loc[:, "Treatment"] = balanced_df["gpt_available_post1"].astype("int8")

# Cange Types
balanced_df.loc[:, "year"]          = balanced_df["year"].astype("int16")
balanced_df.loc[:, "quarter"]       = balanced_df["quarter"].astype("int16")
balanced_df.loc[:, "population"]    = balanced_df["population"].astype("int64")
balanced_df.loc[:, "num_pushers"]   = balanced_df["num_pushers"].astype("float64")
balanced_df.loc[:, "gpt_available"] = balanced_df["gpt_available"].clip(0,1).astype("int8")

# Filter years
balanced_df = balanced_df[(balanced_df["quarter"] >= 1) & (balanced_df["quarter"] <= 16)]

balanced_df

print(balanced_df.isna().sum())

balanced_df.to_csv(p("output", "data", "data_langs_balanced.csv"))


# ============================================================================
# SECTION 2: language_distribution_charts.py
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load data
# =========================
df = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))

# Analysis period
df = df[(df["year"] >= 2020) & (df["year"] <= 2023)]

# =========================
# Plotting function
# =========================
def plot_language_distribution(data, title):

    # Aggregate pushers by language and year
    agg = (
        data.groupby(["language", "year"])["num_pushers"]
        .sum()
        .reset_index()
    )

    # Total pushers per year
    total_by_year = agg.groupby("year")["num_pushers"].sum().reset_index()
    agg = agg.merge(total_by_year, on="year", suffixes=("", "_total"))

    # Percentage share
    agg["pct"] = 100 * agg["num_pushers"] / agg["num_pushers_total"]

    years = sorted(agg["year"].unique())

    # Order languages by the last year percentage
    order = (
        agg[agg["year"] == max(years)]
        .sort_values("pct", ascending=False)["language"]
        .tolist()
    )

    x = np.arange(len(order))
    width = 0.22

    fig, ax = plt.subplots(figsize=(16, 8))

    colors = plt.cm.Pastel1.colors

    # Plot bars per year
    for i, year in enumerate(years):

        vals = (
            agg[agg["year"] == year]
            .set_index("language")
            .loc[order]["pct"]
        )

        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=f"Year {year}",
            color=colors[i],
            edgecolor="gray",
            linewidth=0.8
        )

        # Labels inside bars
        for bar in bars:
            h = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h / 2,
                f"{h:.1f}%",
                ha="center",
                va="center",
                rotation=90,
                fontsize=12,
                color="black"
            )

    # General styling
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


# =========================
# Countries with ChatGPT available
# =========================

df_gpt = df[df["gpt_available"] == 1]

fig = plot_language_distribution(
    df_gpt,
    ""
)

fig.savefig(
    p("output", "figures", "language_distribution_gpt_available_2020_2023.png"),
    dpi=300,
    bbox_inches="tight"
)


# =========================
# Countries without ChatGPT available
# =========================

df_no_gpt = df[df["gpt_available"] == 0]

fig = plot_language_distribution(
    df_no_gpt,
    ""
)

fig.savefig(
    p("output", "figures", "language_distribution_no_gpt_2020_2023.png"),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================================
# SECTION 3: programming_language_trends.py
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Load dataset
# ============================
df = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))

# Rescale to thousands of people
df["num_pushers_thousands"] = df["num_pushers_pc"] * 100

# ============================
# Define quarters
# ============================
quarters = list(range(1, 17))

quarter_labels = [
    "2020-Q1","2020-Q2","2020-Q3","2020-Q4",
    "2021-Q1","2021-Q2","2021-Q3","2021-Q4",
    "2022-Q1","2022-Q2","2022-Q3","2022-Q4",
    "2023-Q1","2023-Q2","2023-Q3","2023-Q4"
]

# Consistent language colors
languages = sorted(df["language"].unique())
color_map = dict(zip(languages, plt.cm.tab10.colors))

# =====================================
# GLOBAL PROGRAMMING LANGUAGE TRENDS
# =====================================

trend_all = (
    df.groupby(["language", "quarter"], as_index=False)
      .agg(num_pushers=("num_pushers_thousands", "mean"))
)

fig, ax = plt.subplots(figsize=(22,10))

for lang in languages:

    sub = (
        trend_all[trend_all["language"] == lang]
        .set_index("quarter")
        .reindex(quarters)
    )

    ax.plot(
        quarters,
        sub["num_pushers"],
        marker="o",
        linewidth=2.8,
        label=lang,
        color=color_map[lang]
    )

ax.set_xlim(1,16)
ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45)

ax.set_ylim(0,6500)
ax.set_yticks(np.arange(0,6501,500))

ax.set_xlabel("Quarter of the year", fontsize=14)
ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=14)

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Programming language", ncol=2)

plt.tight_layout()

fig.savefig(
    p("output", "figures", "language_trend_2020_2023.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================
# COUNTRIES WHERE CHATGPT IS AVAILABLE
# =====================================

df_gpt = df[df["gpt_available"] == 1]

trend_gpt = (
    df_gpt.groupby(["language", "quarter"], as_index=False)
          .agg(num_pushers=("num_pushers_thousands", "mean"))
)

fig, ax = plt.subplots(figsize=(22,10))

for lang in languages:

    sub = (
        trend_gpt[trend_gpt["language"] == lang]
        .set_index("quarter")
        .reindex(quarters)
    )

    ax.plot(
        quarters,
        sub["num_pushers"],
        marker="o",
        linewidth=2.8,
        label=lang,
        color=color_map[lang]
    )

ax.set_xlim(1,16)
ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45)

ax.set_ylim(0,6500)
ax.set_yticks(np.arange(0,6501,500))

ax.set_xlabel("Quarter", fontsize=18)
ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=18)

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Programming language", ncol=2)

plt.tight_layout()

fig.savefig(
    p("output", "figures", "language_trend_gpt_countries_2020_2023.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================
# COUNTRIES WHERE CHATGPT IS NOT AVAILABLE
# =====================================

df_no_gpt = df[df["gpt_available"] == 0]

trend_no_gpt = (
    df_no_gpt.groupby(["language", "quarter"], as_index=False)
             .agg(num_pushers=("num_pushers_thousands", "mean"))
)

fig, ax = plt.subplots(figsize=(22,10))

for lang in languages:

    sub = (
        trend_no_gpt[trend_no_gpt["language"] == lang]
        .set_index("quarter")
        .reindex(quarters)
    )

    ax.plot(
        quarters,
        sub["num_pushers"],
        marker="o",
        linewidth=2.8,
        label=lang,
        color=color_map[lang]
    )

ax.set_xlim(1,16)
ax.set_xticks(quarters)
ax.set_xticklabels(quarter_labels, rotation=45)

ax.set_ylim(0,6500)
ax.set_yticks(np.arange(0,6501,500))

ax.set_xlabel("Quarter", fontsize=18)
ax.set_ylabel("Unique pushers per 100k inhabitants", fontsize=18)

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Programming language", ncol=2, loc="upper left")

plt.tight_layout()

fig.savefig(
    p("output", "figures", "language_trend_no_gpt_countries_2020_2023.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================================
# SECTION 4: chatgpt_global_availability_map.py
# ============================================================================

import warnings
warnings.simplefilter('ignore', FutureWarning)

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import pycountry
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# ==============================
# Load dataset
# ==============================
balanced_df = pd.read_csv(p("output", "data", "data_langs_balanced.csv"))


# ==============================
# ISO2 → ISO3 conversion
# ==============================
def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None


# Convert ISO codes
balanced_df["iso3_code"] = balanced_df["iso2_code"].apply(iso2_to_iso3)
balanced_df = balanced_df.dropna(subset=["iso3_code"])


# ==============================
# Aggregate ChatGPT availability
# ==============================
country_gpt = (
    balanced_df
    .groupby("iso3_code", as_index=False)["gpt_available"]
    .max()
)


# ==============================
# Load world shapefile
# ==============================
shapefile_path = p("external", "ne_110m_admin_0_countries.shp")
world = gpd.read_file(shapefile_path)


# Ensure ISO3 column name
if "ISO_A3" in world.columns:
    world = world.rename(columns={"ISO_A3": "iso_a3"})
elif "iso_a3" not in world.columns:
    raise ValueError("Shapefile does not contain 'ISO_A3' or 'iso_a3' column.")


# ==============================
# Merge data
# ==============================
world = world.merge(
    country_gpt,
    how="left",
    left_on="iso_a3",
    right_on="iso3_code"
)

world["gpt_available"] = world["gpt_available"].fillna(0).astype(int)


# ==============================
# Color palette and legend
# ==============================
colors = ["#ece2f0", "#3b528b"]

cmap = ListedColormap(colors)

legend_patches = [
    mpatches.Patch(color=colors[0], label="Not available"),
    mpatches.Patch(color=colors[1], label="Available")
]


# ==============================
# Plot world map
# ==============================
fig, ax = plt.subplots(figsize=(16, 9))

ax.set_facecolor("#f7f7f7")

world.plot(
    column="gpt_available",
    cmap=cmap,
    edgecolor="white",
    linewidth=0.4,
    ax=ax,
    alpha=0.95
)


# Legend
ax.legend(
    handles=legend_patches,
    loc="lower left",
    frameon=True,
    framealpha=0.8,
    title="ChatGPT Availability",
    title_fontsize=12,
    fontsize=11
)


# Title and styling
ax.set_title(
    "Global ChatGPT Availability",
    fontsize=20,
    fontweight="bold",
    pad=20,
    color="#333333"
)

ax.axis("off")

plt.tight_layout()


# ==============================
# Save figure
# ==============================
output_png = p("output", "figures", "chatgpt_global_availability_map.png")

fig.savefig(
    output_png,
    dpi=600,
    bbox_inches="tight",
    transparent=False
)

print(f"Map saved to: {output_png}")


# ==============================
# Display map
# ==============================
plt.show()


# ==============================
# Print country lists
# ==============================
country_col = "NAME" if "NAME" in world.columns else "ADMIN"


available = (
    world.loc[world["gpt_available"] == 1, country_col]
    .sort_values()
    .tolist()
)

print("\nCountries where ChatGPT is available:")
print(", ".join(available))


not_available = (
    world.loc[world["gpt_available"] == 0, country_col]
    .sort_values()
    .tolist()
)

print("\nCountries where ChatGPT is NOT available:")
print(", ".join(not_available))


# ============================================================================
# SECTION 5: analysis_per_lang_thesis_sub_esp.do  (executed via subprocess)
#
# Requires Stata installed. The script writes the .do content to a temporary
# file and calls Stata in batch mode (/e = no GUI, logs to .log file).
#
# Adjust STATA_EXE if your Stata executable has a different name or path,
# e.g. "StataSE-64", "StataIC-64", "Stata-64", or a full path like
# "C:/Program Files/Stata18/StataMP-64.exe"
# ============================================================================

import subprocess
import tempfile
import os

def _find_stata():
    """Auto-detect Stata executable across Windows and macOS."""
    import platform, glob
    system = platform.system()

    if system == "Windows":
        candidates = []
        for version in range(19, 14, -1):          # Stata 19 → 15
            for edition in ("MP", "SE", "IC", "BE"):
                candidates.append(
                    rf"C:\Program Files\Stata{version}\Stata{edition}-64.exe"
                )
        for path in candidates:
            if os.path.isfile(path):
                return path
        # Last resort: try PATH
        return "StataMP-64"

    elif system == "Darwin":                        # macOS
        for version in range(19, 14, -1):
            for edition in ("MP", "SE", "IC", "BE"):
                path = f"/Applications/Stata/Stata{edition}.app/Contents/MacOS/Stata{edition}"
                if os.path.isfile(path):
                    return path
        return "stata"

    else:                                           # Linux
        return "stata"

STATA_EXE = _find_stata()

STATA_DO_SCRIPT = r"""
//----------------------------------------------------------------------------//
//
// Proyecto: Tesis
// Impacto de ChatGPT en el número de programadores en GitHub
//
//----------------------------------------------------------------------------//

global path "__PROJECT_PATH__"

* Crear carpetas si no existen
cap mkdir "$path/output"
cap mkdir "$path/output/figures"
cap mkdir "$path/output/tables"

import delimited "$path/output/data/data_langs_balanced.csv", clear

sort unique_id year quarter
drop if iso2_code == "HK"
label var num_pushers_pc "Número de pushers por 100k habitantes"
label var gpt_available_post1 "ChatGPT Disponible"

**************************
// Cambio de nombres de etiquetas
**************************
replace language = "C_hashtag" if language == "C#"
replace language = "C_plus"    if language == "C++"

*****************
// Tesis_DataScience
*****************
local DataScience "C C_hashtag C_plus Go Java JavaScript PHP Python Ruby TypeScript"

foreach lang of local DataScience {

        local l`v' : variable label num_pushers_pc

        *-----------------------------------------------------
        * DID
        *-----------------------------------------------------
        eststo `lang'_did: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(did) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'did", .png)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'did_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'did_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'

        *-----------------------------------------------------
        * SC
        *-----------------------------------------------------
        eststo `lang'_sc: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(sc) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'sc", .png)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'sc_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'sc_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'

        *-----------------------------------------------------
        * SDID
        *-----------------------------------------------------
        eststo `lang'_sdid: sdid num_pushers_pc iso2_code quarter gpt_available_post1 if language == "`lang'", ///
                vce(bootstrap) reps(100) seed(1234) method(sdid) graph g1on ///
                g1_opt(xtitle("Trimestre") ytitle("Diferencia") scheme(plotplainblind)) ///
                g2_opt(ytitle("`l`v''-`lang'") scheme(plotplainblind) ///
                        xtitle("Trimestre") ///
                        xlabel(1 "2020-T1" 2 "2020-T2" 3 "2020-T3" 4 "2020-T4"  ///
                        5 "2021-T1" 6 "2021-T2" 7 "2021-T3" 8 "2021-T4" ///
                        9 "2022-T1" 10 "2022-T2" 11 "2022-T3" 12 "2022-T4" ///
                        13 "2023-T1" 14 "2023-T2" 15 "2023-T3" 16 "2023-T4", ///
                        labsize(small) angle(45)) ///
                        legend(order(1 "Control" 2 "Tratado") pos(12) col(2)) ///
                ) graph_export("$path/output/figures/`lang'sdid", .png)

        * Traducir eje Y derecho: Lambda weight -> Peso lambda
        cap graph use "$path/output/figures/`lang'sdid_trends12.gph"
        cap gr_edit .yaxis2.title.text = {}
        cap gr_edit .yaxis2.title.text.Arrpush "Peso lambda"
        cap graph export "$path/output/figures/`lang'sdid_trends12.png", replace

        sum num_pushers_pc if gpt_available_post1==0 & quarter<12 & language == "`lang'"
        estadd scalar control_mean `r(mean)'
}

** Tabla con tres paneles

esttab C_did C_sc C_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                replace label booktabs                                                                   ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                delim("&")  ///
                nomtitle ///
                collabels(none) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                mgroups("\shortstack{DID}" ///
                                        "\shortstack{SC}" ///
                                        "\shortstack{SDID}",  ///
                                        pattern(1 1 1)                           ///
                                        prefix(\multicolumn{@span}{c}{) suffix(}) span                       ///
                                        erepeat(\cmidrule(lr){@span})) ///
                                nomtitles                       ///
                scalars("control_mean Media de referencia") ///
                        refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel A. \textbf{ \textit{C} } } }" , nolabel) ///
                        prefoot("") posthead(\hline) postfoot("")  nonumbers

esttab C_hashtag_did C_hashtag_sc C_hashtag_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel B. \textbf{ \textit{C\#} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab C_plus_did C_plus_sc C_plus_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel C. \textbf{ \textit{C++} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab Go_did Go_sc Go_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel D. \textbf{ \textit{Go} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab Java_did Java_sc Java_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel E. \textbf{ \textit{Java} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab JavaScript_did JavaScript_sc JavaScript_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel F. \textbf{ \textit{JavaScript} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") postfoot("") delim("&") collabels(none) nonumbers nogaps nonote

esttab PHP_did PHP_sc PHP_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel G. \textbf{ \textit{PHP} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

esttab Python_did Python_sc Python_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel H. \textbf{ \textit{Python} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

esttab Ruby_did Ruby_sc Ruby_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel I. \textbf{ \textit{Ruby} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

esttab TypeScript_did TypeScript_sc TypeScript_sdid ///
                using "$path/output/tables/gpt_impact_github_DataScience.tex", ///
                append label booktabs mlabel(,none) ///
                cells(b(star fmt(%9.3f)) se(par fmt(%9.3f)))             ///
                starlevels(* 0.10 * 0.05 ** 0.01) ///
                keep(gpt_available_post1) ///
                order(gpt_available_post1) ///
                scalars("control_mean Media de referencia") ///
                refcat(gpt_available_post1 "\Gape[0.25cm][0.25cm]{ \underline{Panel J. \textbf{ \textit{TypeScript} } } }" , nolabel) ///
                prehead("") prefoot("") posthead("\hline") delim("&") collabels(none) nonumbers nogaps nonote

cd "$path"
""".replace("__PROJECT_PATH__", BASE_DIR.replace(os.sep, "/"))

# Write .do content to a named file inside the project so the log lands there too.
# Stata /e writes <scriptname>.log in the current working directory.
do_path  = p("output", "analysis_per_lang.do")
log_path = p("output", "analysis_per_lang.log")

with open(do_path, "w", encoding="utf-8") as f:
    f.write(STATA_DO_SCRIPT)

try:
    print(f"Running Stata script: {do_path}")
    result = subprocess.run(
        [STATA_EXE, "/e", "do", do_path],
        check=True,
        capture_output=True,
        text=True,
        cwd=p("output")       # log lands in output/
    )
    print("Stata finished successfully.")
    if os.path.exists(log_path):
        with open(log_path, encoding="latin-1") as lf:
            print(lf.read().encode("utf-8", errors="replace").decode("utf-8"))
except subprocess.CalledProcessError as e:
    print(f"Stata exited with error code {e.returncode}.")
    if os.path.exists(log_path):
        with open(log_path, encoding="latin-1") as lf:
            print(lf.read().encode("utf-8", errors="replace").decode("utf-8"))
except FileNotFoundError:
    print(
        f"Stata executable '{STATA_EXE}' not found. "
        "Update STATA_EXE with the correct name or full path."
    )
