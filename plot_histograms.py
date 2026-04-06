import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", rc={
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "grid.color": "#E5E5E5",
    "grid.linewidth": 1,
})

data = pd.read_csv("telecom_churn.csv")
clean = data.drop(columns=["DataPlan"])

features = [
    "AccountWeeks",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins"
]

os.makedirs("Histograms", exist_ok=True)
palette = ["#A8DFF3", "#F4BABF"]
churn_labels = sorted(clean["Churn"].unique())

for column in features:
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, churn_value in enumerate(churn_labels):
        subset = clean[clean["Churn"] == churn_value][column]
        weights = (100 / len(subset)) * np.ones(len(subset))  # converts to percentage
        ax.hist(
            subset,
            bins=30,
            weights=weights,
            color=palette[i],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.5,
            label=f"Churn={churn_value}"
        )

    ax.set_ylabel("Percentage (%)", fontsize=11, color="#000000")

    # five-number summary
    summaries = []
    grouped = clean.groupby("Churn")[column]

    for churn_value, group in grouped:
        summaries.append(
            (
                churn_value,
                f"Churn={churn_value}\n"
                f"Min={group.min():.1f}\n"
                f"Q1={group.quantile(0.25):.1f}\n"
                f"Med={group.quantile(0.50):.1f}\n"
                f"Q3={group.quantile(0.75):.1f}\n"
                f"Max={group.max():.1f}"
            )
        )

    # summary cards
    ax.text(
        0.75, 0.97, summaries[0][1],
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            facecolor="#FFFFFF",
            edgecolor="#EBEBEB",
            alpha=1.0,
            boxstyle="round,pad=0.4"
        )
    )

    ax.text(
        0.88, 0.97, summaries[1][1],
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            facecolor="#FFFFFF",
            edgecolor="#EBEBEB",
            alpha=1.0,
            boxstyle="round,pad=0.4"
        )
    )

    ax.legend(loc="upper left", title="Churn", fontsize=9, title_fontsize=9)

    plt.title(f"{column} vs Churn", fontsize=14, fontweight="bold", color="#000000", pad=12)
    plt.xlabel(column, fontsize=11, color="#000000")
    plt.ylabel("Percentage (%)", fontsize=11, color="#000000")
    ax.tick_params(colors="#000000", labelsize=10)

    sns.despine()
    plt.tight_layout()

    plt.savefig(f"Histograms/{column}_vs_Churn_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

print("all histograms saved in Histograms/ folder")