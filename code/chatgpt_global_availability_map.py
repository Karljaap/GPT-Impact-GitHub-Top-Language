
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
balanced_df = pd.read_csv(".../output/data/data_langs_balanced.csv")


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
shapefile_path = ".../external/ne_110m_admin_0_countries.shp"
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
output_png = ".../output/figures/chatgpt_global_availability_map.png"

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
