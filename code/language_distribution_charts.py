import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load data
# =========================
df = pd.read_csv(".../output/data/data_langs_balanced.csv")

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
    ".../output/figures/language_distribution_gpt_available_2020_2023.png",
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
    ".../output/figures/language_distribution_no_gpt_2020_2023.png",
    dpi=300,
    bbox_inches="tight"
)
