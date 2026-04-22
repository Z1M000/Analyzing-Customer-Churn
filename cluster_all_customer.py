import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------
df = pd.read_csv("cleaned_dataset_new.csv")
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna().copy()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 2. Set up target and feature columns
# --------------------------------------------------
target_col = "Churn"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

# --------------------------------------------------
# 3. Evaluate candidate k values using elbow + silhouette
# --------------------------------------------------
def evaluate_k(X, k_min=2, k_max=10, random_state=42):
    X_values = X.to_numpy()

    k_values = []
    inertias = []
    silhouettes = []

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=random_state
        )
        labels = model.fit_predict(X_values)

        k_values.append(k)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X_values, labels))

    metrics_df = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
        "silhouette": silhouettes
    })

    best_k = metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", linewidth=2)
    axes[0].axvline(best_k, color="green", linestyle="--", linewidth=2, label=f"Selected k = {best_k}")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", linewidth=2)
    axes[1].axvline(best_k, color="green", linestyle="--", linewidth=2, label=f"Selected k = {best_k}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("all_customers_elbow_silhouette.png", dpi=300, bbox_inches="tight")
    plt.close()

    return metrics_df, best_k

metrics_df, best_k = evaluate_k(X, k_min=2, k_max=8)
print("\nEvaluation table:")
print(metrics_df)
print(f"\nSelected k based on silhouette score: {best_k}")

# --------------------------------------------------
# 4. Run final clustering
# --------------------------------------------------
final_k = int(best_k)

kmeans = KMeans(
    n_clusters=final_k,
    init="k-means++",
    n_init=10,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X)

clustered_df = df.copy()
clustered_df["Cluster"] = cluster_labels

print("\nCluster sizes:")
print(clustered_df["Cluster"].value_counts().sort_index())

# --------------------------------------------------
# 5. Compute churn rate for each cluster
# --------------------------------------------------
cluster_summary = clustered_df.groupby("Cluster").agg(
    CustomerCount=("Churn", "size"),
    ChurnRate=("Churn", "mean")
).reset_index()

cluster_summary["ChurnRate"] = (cluster_summary["ChurnRate"] * 100).round(2)

print("\nCluster churn summary:")
print(cluster_summary)

# --------------------------------------------------
# 6. Find cluster characteristics
# --------------------------------------------------
overall_mean = X.mean()
overall_std = X.std(ddof=0).replace(0, np.nan)

cluster_means = clustered_df.groupby("Cluster")[feature_cols].mean()
cluster_zscores = (cluster_means - overall_mean) / overall_std

print("\nCluster means in standardized form:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(cluster_zscores.round(2))

# --------------------------------------------------
# 7. Helper: create persona text
# --------------------------------------------------
def make_persona_text(cluster_id, z_row, churn_rate, customer_count, top_n=3):
    z_sorted = z_row.dropna().sort_values(ascending=False)
    top_high = z_sorted.head(top_n)
    top_low = z_sorted.tail(top_n).sort_values()

    high_features = ", ".join([f"{feat} ({val:.2f})" for feat, val in top_high.items()])
    low_features = ", ".join([f"{feat} ({val:.2f})" for feat, val in top_low.items()])

    title = "general customers"
    if ("AccountWeeks" in z_row.index and z_row["AccountWeeks"] > 0.5) and ("ContractRenewal" in z_row.index and z_row["ContractRenewal"] > 0.5):
        title = "long-term loyal customers"
    elif ("CustServCalls" in z_row.index and z_row["CustServCalls"] > 0.5):
        title = "service-heavy customers"
    elif ("DataUsage" in z_row.index and z_row["DataUsage"] > 0.5) and ("MonthlyCharge" in z_row.index and z_row["MonthlyCharge"] > 0.5):
        title = "high-usage premium customers"
    elif ("OverageFee" in z_row.index and z_row["OverageFee"] > 0.5):
        title = "overage-sensitive customers"

    persona = (
        f"Cluster {cluster_id} represents {title}. "
        f"It contains {customer_count} customers and has a churn rate of {churn_rate:.2f}%. "
        f"The strongest above-average features in this cluster are {high_features}, "
        f"while the strongest below-average features are {low_features}. "
        f"Overall, this group shows a distinct behavioral pattern that helps us understand "
        f"what kind of customers are more likely to stay or leave."
    )
    return persona

print("\n=== Persona paragraphs ===\n")
for _, row in cluster_summary.iterrows():
    cid = int(row["Cluster"])
    churn_rate = float(row["ChurnRate"])
    count = int(row["CustomerCount"])
    z_row = cluster_zscores.loc[cid]

    print(make_persona_text(cid, z_row, churn_rate, count, top_n=3))
    print()

# --------------------------------------------------
# 8. PCA visualization using the same color palette
# --------------------------------------------------
# Use a consistent palette across plots
cmap = plt.cm.Paired
norm = plt.Normalize(vmin=0, vmax=final_k - 1)

# 2D PCA
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=cluster_labels,
    cmap=cmap,
    norm=norm,
    s=50,
    alpha=0.8,
    edgecolors="w",
    linewidth=0.5
)
plt.title("2D PCA View of All Customer Clusters", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("all_customers_pca_2d.png", dpi=300, bbox_inches="tight")
plt.close()

# 3D PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
    c=cluster_labels,
    cmap=cmap,
    norm=norm,
    s=40,
    alpha=0.8,
    edgecolors="w",
    linewidth=0.5
)

ax.set_title("3D PCA View of All Customer Clusters", fontsize=14, pad=15)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.view_init(elev=20, azim=45)

fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label="Cluster")
plt.tight_layout()
plt.savefig("all_customers_pca_3d.png", dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 8b. PCA Loadings Heatmap (add after existing PCA section)
# --------------------------------------------------
# --------------------------------------------------
# PCA Loadings Heatmap — 2D only (PC1 & PC2)
# --------------------------------------------------
import seaborn as sns

pca_for_heatmap = PCA(n_components=2, random_state=42)
pca_for_heatmap.fit(X)

loadings_df = pd.DataFrame(
    pca_for_heatmap.components_,
    columns=feature_cols,
    index=[
        f"PC1 ({pca_for_heatmap.explained_variance_ratio_[0]*100:.1f}%)",
        f"PC2 ({pca_for_heatmap.explained_variance_ratio_[1]*100:.1f}%)"
    ]
)

fig, ax = plt.subplots(figsize=(11, 3.5))
sns.heatmap(
    loadings_df,
    annot=True, fmt=".2f",
    cmap="RdBu_r", center=0,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 12}
)
ax.set_title("PCA Feature Loadings — 2D PCA (Contribution of Each Feature to Each Component)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Features", fontsize=11)
ax.set_ylabel("Principal Components", fontsize=11)
ax.tick_params(axis='x', labelrotation=30, labelsize=10)
ax.tick_params(axis='y', labelrotation=0, labelsize=11)
plt.tight_layout()
plt.savefig("pca_2d_loadings_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 9. Save results
# --------------------------------------------------
clustered_df.to_csv("all_customers_clustered_results.csv", index=False)
cluster_summary.to_csv("cluster_churn_summary.csv", index=False)

print("\nSaved:")
print("- all_customers_elbow_silhouette.png")
print("- all_customers_pca_2d.png")
print("- all_customers_pca_3d.png")
print("- all_customers_clustered_results.csv")
print("- cluster_churn_summary.csv")

# --------------------------------------------------
# Z-Score Table Image
# --------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

feature_labels = {
    "AccountWeeks": "Account Weeks",
    "ContractRenewal": "Contract Renewal",
    "DataUsage": "Data Usage",
    "CustServCalls": "Cust. Service Calls",
    "DayMins": "Day Minutes",
    "DayCalls": "Day Calls",
    "MonthlyCharge": "Monthly Charge",
    "OverageFee": "Overage Fee",
    "RoamMins": "Roaming Minutes"
}

rows = []
for feat in feature_cols:
    z0 = cluster_zscores.loc[0, feat]
    z1 = cluster_zscores.loc[1, feat]
    rows.append([feature_labels.get(feat, feat), f"{z0:+.3f}", f"{z1:+.3f}"])

fig, ax = plt.subplots(figsize=(10.5, 6.3))
ax.axis("off")

col_labels = ["Feature", "Cluster 0\n(High-Value Users)\nz-score", "Cluster 1\n(Standard Users)\nz-score"]
table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    colWidths=[0.34, 0.33, 0.33],
    bbox=[0.0, 0.13, 1.0, 0.73]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.85)

for j in range(3):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold", va="center")
    table[0, j].set_height(table[0, j].get_height() * 1.45)
    table[0, j].PAD = 0.02

for i, (feat, z0_str, z1_str) in enumerate(rows, start=1):
    z0 = float(z0_str); z1 = float(z1_str)
    table[i, 0].set_facecolor("#F8F9FA" if i % 2 != 0 else "#ECEFF1")
    table[i, 0].set_text_props(ha="left", va="center")
    for col_idx, z_val in [(1, z0), (2, z1)]:
        if z_val > 0.5:   color = "#C8E6C9"
        elif z_val < -0.5: color = "#FFCDD2"
        else:              color = "#FFF9C4"
        table[i, col_idx].set_facecolor(color)

green_patch = mpatches.Patch(color="#C8E6C9", label="Above average (z > +0.5)")
yellow_patch = mpatches.Patch(color="#FFF9C4", label="Near average")
red_patch = mpatches.Patch(color="#FFCDD2", label="Below average (z < −0.5)")
ax.legend(handles=[green_patch, yellow_patch, red_patch],
          loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=9)

ax.set_title("Cluster Feature Z-Score Comparison", fontsize=13, fontweight="bold", pad=20)
fig.subplots_adjust(top=0.90, bottom=0.10)
plt.savefig("cluster_zscore_table.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
plt.close()
