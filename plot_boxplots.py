import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

summary = clean[features].describe()
os.makedirs("Boxplots", exist_ok=True)
palette = ["#A8DFF3", "#F4BABF"]

for column in features:
    plt.figure(figsize=(9, 6))

    ax = sns.boxplot(
        x="Churn",
        y=column,
        data=clean,
        hue="Churn",
        palette=palette,
        width=0.25,
        linewidth=0.8,
        flierprops=dict(
            marker="o",
            markersize=6,
            alpha=0.4,
            markerfacecolor="#000000",
            markeredgewidth=0
        )
    )

    ax.legend(loc="upper left", title="Churn", fontsize=9, title_fontsize=9)

    # ===== 计算 five-number summary =====
    grouped = clean.groupby("Churn")[column]
    summaries = []

    for churn_value, group in grouped:
        min_val = group.min()
        q1 = group.quantile(0.25)
        median = group.quantile(0.50)
        q3 = group.quantile(0.75)
        max_val = group.max()

        summaries.append(
            (
                churn_value,
                f"Churn={churn_value}\n"
                f"Min={min_val:.1f}\n"
                f"Q1={q1:.1f}\n"
                f"Med={median:.1f}\n"
                f"Q3={q3:.1f}\n"
                f"Max={max_val:.1f}"
            )
        )

    # leave space for summary card
    ax.set_xlim(-0.5, 2.2)

    # summary card
    ax.text(
        1.55, 0.8, summaries[0][1],
        transform=ax.get_xaxis_transform(),
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
        1.85, 0.8, summaries[1][1],
        transform=ax.get_xaxis_transform(),
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            facecolor="#FFFFFF",
            edgecolor="#EBEBEB",
            alpha=1.0,
            boxstyle="round,pad=0.4"
        )
    )

    # make it look nicer
    plt.title(f"{column} vs Churn", fontsize=14, fontweight="bold", color="#000000", pad=12)
    plt.xlabel("Churn", fontsize=11, color="#000000")
    plt.ylabel(column, fontsize=11, color="#000000")
    ax.tick_params(colors="#000000", labelsize=10)

    sns.despine()
    plt.tight_layout()

    # save
    plt.savefig(f"Boxplots/{column}_vs_Churn_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

print("all boxplots saved in Boxplots/ folder")