import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Load dataset
# ============================
df = pd.read_csv(".../output/data/data_langs_balanced.csv")

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
    ".../output/figures/language_trend_2020_2023.png",
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
    ".../output/figures/language_trend_gpt_countries_2020_2023.png",
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
    ".../output/figures/language_trend_no_gpt_countries_2020_2023.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
