"""
src/utils/eda_utils.py
=======================
Reusable EDA helper functions 

Each function takes a DataFrame and returns either:
- A matplotlib Figure (for saving/displaying in notebooks)
- A summary DataFrame (for tabular display)
- A dict (for logging)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# STYLE CONSTANTS

PALETTE = {
    "primary"   : "#2563EB",   # blue
    "secondary" : "#16A34A",   # green
    "accent"    : "#DC2626",   # red  (for anomalies / flags)
    "neutral"   : "#6B7280",   # grey
    "warn"      : "#D97706",   # amber
    "bg"        : "#F9FAFB",   # light grey background
}

plt.rcParams.update({
    "figure.facecolor" : PALETTE["bg"],
    "axes.facecolor"   : "white",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.family"      : "sans-serif",
    "font.size"        : 10,
})


# DATASET OVERVIEW


def dataset_overview(frames: dict) -> pd.DataFrame:
    """
    Build a summary table of all silver datasets.

    frames: dict of {name: DataFrame}

    Returns a DataFrame with columns:
        dataset, rows, columns, memory_kb, null_count, null_pct
    """
    rows = []
    for name, df in frames.items():
        null_count = df.isnull().sum().sum()
        rows.append({
            "dataset"   : name,
            "rows"      : len(df),
            "columns"   : len(df.columns),
            "memory_kb" : round(df.memory_usage(deep=True).sum() / 1024, 1),
            "null_count": int(null_count),
            "null_pct"  : round(100 * null_count / (len(df) * len(df.columns)), 2),
        })
    return pd.DataFrame(rows)

def plot_scatter(df: pd.DataFrame, x_axis: str, y_axis: str, title: str) -> plt.Figure:
    """
    Simple scatter plot of two columns.

    Useful for quick checks of relationships between variables.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x_axis], df[y_axis], alpha=0.5, color=PALETTE["primary"], s=20)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{title}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# VOLUME DISTRIBUTION


def plot_volume_distribution(tx: pd.DataFrame, volume_col: str = "Volume_Liters") -> plt.Figure:
    """
    Plot the distribution of Volume_Liters.

    Two panels:
    - Left  : histogram of all non-zero volumes
    - Right : box plot to surface outliers

    Why we exclude zeros:
    Zero-volume rows are censoring events (no delivery), not sales.
    Including them would distort the distribution analysis.
    """
    nonzero = tx[tx[volume_col] > 0][volume_col]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Volume_Liters Distribution (non-zero rows)", fontsize=13, fontweight="bold")

    # ── Histogram ──
    axes[0].hist(nonzero, bins=60, color=PALETTE["primary"], alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Volume (Liters)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[0].axvline(nonzero.mean(),   color=PALETTE["accent"],    linestyle="--", label=f"Mean  {nonzero.mean():.1f}L")
    axes[0].axvline(nonzero.median(), color=PALETTE["secondary"], linestyle="--", label=f"Median {nonzero.median():.1f}L")
    axes[0].legend()

    # ── Box plot ──
    axes[1].boxplot(nonzero, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE["primary"], alpha=0.5),
                    medianprops=dict(color=PALETTE["accent"], linewidth=2))
    axes[1].set_ylabel("Volume (Liters)")
    axes[1].set_title("Box Plot (outlier detection)")
    axes[1].set_xticks([])

    fig.tight_layout()
    return fig


def plot_volume_by_month(tx: pd.DataFrame,
                          volume_col: str = "Volume_Liters",
                          year_col:   str = "Year",
                          month_col:  str = "Month") -> plt.Figure:
    """
    Plot total volume sold per month, grouped by year.

    Shows seasonality patterns — are certain months consistently higher?
    Useful for understanding whether January 2026 is a peak or trough month.
    """
    monthly = (
        tx.groupby([year_col, month_col])[volume_col]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warn"],
              PALETTE["accent"], PALETTE["neutral"]]

    for i, (year, grp) in enumerate(monthly.groupby(year_col)):
        ax.plot(grp[month_col], grp[volume_col],
                marker="o", linewidth=2,
                color=colors[i % len(colors)],
                label=str(year))

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Volume (Liters)")
    ax.set_title("Monthly Total Volume by Year", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.legend(title="Year")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig



# OUTLET-LEVEL ANALYSIS


def compute_outlet_stats(tx: pd.DataFrame,
                          outlet_col:  str = "Outlet_ID",
                          volume_col:  str = "Volume_Liters",
                          month_col:   str = "Month",
                          year_col:    str = "Year") -> pd.DataFrame:
    """
    Compute per-outlet aggregate statistics from transaction data.

    Returns a DataFrame with one row per outlet containing:
    - total_volume       : sum of all volume across all months
    - mean_volume        : average monthly volume
    - max_volume         : highest single-month volume
    - min_volume         : lowest single-month volume
    - std_volume         : standard deviation of monthly volumes
    - cv                 : coefficient of variation (std/mean) — measure of variability
    - months_active      : number of distinct months with any transaction
    - zero_volume_months : number of months with zero volume
    - ratio_max_to_mean  : max / mean — high ratio signals censoring in low months
    - trend_slope        : linear trend of volume over time (positive = growing)

    Why CV matters:
    CV = std / mean. An outlet with CV near 0 has suspiciously flat sales.
    That is a censoring signal. A natural outlet should show variation.

    Why ratio_max_to_mean matters:
    If an outlet's max is 5x its mean, the low months were probably constrained.
    The max is closer to the true potential.
    """
    def trend_slope(series):
        """Fit a line to volumes over time and return the slope."""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        return float(np.polyfit(x, series, 1)[0])

    grp = tx.groupby(outlet_col)[volume_col]

    stats = pd.DataFrame({
        "total_volume"      : grp.sum(),
        "mean_volume"       : grp.mean(),
        "max_volume"        : grp.max(),
        "min_volume"        : grp.min(),
        "std_volume"        : grp.std().fillna(0),
        "months_active"     : grp.count(),
        "zero_volume_months": (tx.groupby(outlet_col)[volume_col].apply(lambda x: (x == 0).sum())),
    }).reset_index()

    stats.rename(columns={outlet_col: "Outlet_ID"}, inplace=True)

    stats["cv"] = stats["std_volume"] / stats["mean_volume"].replace(0, np.nan)
    stats["cv"] = stats["cv"].fillna(0)

    stats["ratio_max_to_mean"] = stats["max_volume"] / stats["mean_volume"].replace(0, np.nan)
    stats["ratio_max_to_mean"] = stats["ratio_max_to_mean"].fillna(1)

    # Trend slope: sort by year-month before computing
    tx_sorted = tx.sort_values([year_col, month_col])
    slope_map = (
        tx_sorted.groupby(outlet_col)[volume_col]
        .apply(trend_slope)
        .reset_index()
        .rename(columns={volume_col: "trend_slope", outlet_col: "Outlet_ID"})
    )
    stats = stats.merge(slope_map, on="Outlet_ID", how="left")

    return stats


def plot_outlet_cv_distribution(outlet_stats: pd.DataFrame) -> plt.Figure:
    """
    Plot the distribution of Coefficient of Variation (CV) across outlets.

    CV < 0.05 = flat (likely constrained)
    CV 0.05–0.5 = normal variation
    CV > 0.5 = high variation (possibly intermittent supply)
    """
    cv = outlet_stats["cv"].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(cv, bins=50, color=PALETTE["primary"], alpha=0.8, edgecolor="white")
    ax.axvline(0.05, color=PALETTE["accent"],    linestyle="--", linewidth=1.5, label="CV=0.05 (flat threshold)")
    ax.axvline(0.5,  color=PALETTE["warn"],      linestyle="--", linewidth=1.5, label="CV=0.5 (high variation)")
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Number of Outlets")
    ax.set_title("Outlet CV Distribution — Censoring Signal", fontsize=13, fontweight="bold")
    ax.legend()

    flat_count = (cv < 0.05).sum()
    ax.annotate(f"{flat_count} flat outlets\n(possible cap)",
                xy=(0.02, ax.get_ylim()[1] * 0.8),
                color=PALETTE["accent"], fontsize=9)

    fig.tight_layout()
    return fig


def plot_top_bottom_outlets(outlet_stats: pd.DataFrame, n: int = 20) -> plt.Figure:
    """
    Horizontal bar charts of the top-N and bottom-N outlets by total volume.
    """
    top    = outlet_stats.nlargest(n,  "total_volume")
    bottom = outlet_stats.nsmallest(n, "total_volume")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top outlets
    axes[0].barh(top["Outlet_ID"].astype(str), top["total_volume"],
                 color=PALETTE["secondary"], alpha=0.85)
    axes[0].set_title(f"Top {n} Outlets by Total Volume", fontweight="bold")
    axes[0].set_xlabel("Total Volume (Liters)")
    axes[0].invert_yaxis()

    # Bottom outlets
    axes[1].barh(bottom["Outlet_ID"].astype(str), bottom["total_volume"],
                 color=PALETTE["accent"], alpha=0.85)
    axes[1].set_title(f"Bottom {n} Outlets by Total Volume", fontweight="bold")
    axes[1].set_xlabel("Total Volume (Liters)")
    axes[1].invert_yaxis()

    fig.suptitle("Outlet Volume Extremes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig



# OUTLET MASTER ANALYSIS


def plot_outlet_categorical_breakdown(out: pd.DataFrame) -> plt.Figure:
    """
    Bar charts showing distribution of Outlet_Type and Outlet_Size.

    Why this matters:
    Different outlet types (e.g. restaurant vs kiosk) have different
    natural demand ceilings. Knowing the breakdown tells us whether
    our model needs to treat these segments differently.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Outlet Master — Categorical Distributions", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, ["Outlet_Type", "Outlet_Size"]):
        counts = out[col].value_counts()
        ax.barh(counts.index.astype(str), counts.values,
                color=PALETTE["primary"], alpha=0.8)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Count")
        ax.invert_yaxis()
        for i, v in enumerate(counts.values):
            ax.text(v + counts.values.max() * 0.01, i,
                    f"{v:,}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_cooler_vs_volume(tx: pd.DataFrame, out: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot: Cooler_Count vs mean monthly Volume_Liters per outlet.

    Hypothesis: outlets with more coolers can store more product → higher volume.
    If this holds, Cooler_Count is a strong feature for our model.
    """
    outlet_mean = (
        tx.groupby("Outlet_ID")["Volume_Liters"]
        .mean()
        .reset_index()
        .rename(columns={"Volume_Liters": "mean_volume"})
    )
    merged = outlet_mean.merge(out[["Outlet_ID", "Cooler_Count"]], on="Outlet_ID", how="inner")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(merged["Cooler_Count"], merged["mean_volume"],
               alpha=0.4, color=PALETTE["primary"], s=20)

    # Trend line
    z = np.polyfit(merged["Cooler_Count"], merged["mean_volume"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged["Cooler_Count"].min(), merged["Cooler_Count"].max(), 100)
    ax.plot(x_line, p(x_line), color=PALETTE["accent"], linewidth=2, label="Trend line")

    ax.set_xlabel("Cooler Count")
    ax.set_ylabel("Mean Monthly Volume (Liters)")
    ax.set_title("Cooler Count vs Mean Volume per Outlet", fontsize=13, fontweight="bold")
    ax.legend()

    corr = merged["Cooler_Count"].corr(merged["mean_volume"])
    ax.annotate(f"Pearson r = {corr:.3f}",
                xy=(0.7, 0.05), xycoords="axes fraction",
                fontsize=10, color=PALETTE["neutral"])

    fig.tight_layout()
    return fig



# SEASONALITY ANALYSIS


def plot_seasonality_index_distribution(dist: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of Seasonality_Index category counts across all distributor-months.
    """
    counts = dist["Seasonality_Index"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=[PALETTE["secondary"], PALETTE["warn"], PALETTE["accent"]][:len(counts)],
                  alpha=0.85, edgecolor="white")
    ax.set_title("Seasonality_Index Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Seasonality Category")
    ax.set_ylabel("Count")

    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_volume_by_seasonality(tx: pd.DataFrame, dist: pd.DataFrame) -> plt.Figure:
    """
    Box plot of Volume_Liters grouped by the Seasonality_Index of that month.

    This answers: does the distributor's seasonality label actually correspond
    to higher/lower outlet volumes?

    If yes → Seasonality_Index is a useful model feature.
    If no  → the label may not be informative for this problem.
    """
    merged = tx.merge(
        dist[["Distributor_ID", "Year", "Month", "Seasonality_Index"]],
        on=["Distributor_ID", "Year", "Month"],
        how="left"
    )
    merged = merged[merged["Volume_Liters"] > 0]  # exclude zero-volume rows

    groups = merged.groupby("Seasonality_Index")["Volume_Liters"]
    labels = list(groups.groups.keys())
    data   = [groups.get_group(l).values for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color=PALETTE["accent"], linewidth=2))

    colors = [PALETTE["secondary"], PALETTE["warn"], PALETTE["primary"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("Seasonality Index Category")
    ax.set_ylabel("Volume Liters (non-zero)")
    ax.set_title("Volume by Seasonality Category", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig



# HOLIDAY ANALYSIS


def plot_holidays_per_month(hol: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of holiday count per month across all years.

    Why this matters:
    Months with many holidays may have disrupted delivery schedules,
    contributing to censored demand in those months.
    """
    monthly = hol.groupby("Month").size().reset_index(name="holiday_count")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["Month"], monthly["holiday_count"],
           color=PALETTE["primary"], alpha=0.8, edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Holidays")
    ax.set_title("Holiday Count per Month (all years)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig



# CENSORING SIGNAL SUMMARY


def censoring_signal_summary(tx: pd.DataFrame, outlet_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table of censoring signals detected.

    Returns a DataFrame for display in the notebook.
    """
    total_outlets    = outlet_stats["Outlet_ID"].nunique()
    total_rows       = len(tx)
    zero_rows        = tx["flag_zero_volume"].sum() if "flag_zero_volume" in tx.columns else "N/A"
    flat_outlets_n   = (outlet_stats["cv"] < 0.05).sum()
    high_ratio_n     = (outlet_stats["ratio_max_to_mean"] > 3).sum()

    summary = pd.DataFrame([
        {"Signal"                    : "Zero-volume transactions",
         "Count" : int(zero_rows) if isinstance(zero_rows, (int, float)) else "N/A",
         "As % of total rows"        : f"{100*zero_rows/total_rows:.1f}%" if isinstance(zero_rows, (int, float)) else "N/A",
         "Interpretation"            : "Possible stockout / missed visit / credit block"},
        {"Signal"                    : "Flat-volume outlets (CV < 0.05)",
         "Count"                     : int(flat_outlets_n),
         "As % of total rows"        : f"{100*flat_outlets_n/total_outlets:.1f}%",
         "Interpretation"            : "Delivery cap likely — true demand may be higher"},
        {"Signal"                    : "High max/mean ratio (> 3×)",
         "Count"                     : int(high_ratio_n),
         "As % of total rows"        : f"{100*high_ratio_n/total_outlets:.1f}%",
         "Interpretation"            : "Low months were constrained; max is closer to true potential"},
    ])
    return summary
