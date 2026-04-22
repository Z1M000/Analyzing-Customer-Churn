from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("cleaned_dataset.csv")
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
ROC_FIGURE_PATH = Path("Figures/roc_curves.png")
CONFUSION_MATRIX_PATH = Path("Figures/confusion_matrices.png")
METRIC_TABLE_PATH = Path("Figures/model_comparison_table.png")
DEFAULT_METRIC_TABLE_PATH = Path("Figures/model_comparison_table_default_threshold.png")
TREE_IMPORTANCE_PATH = Path("Figures/decision_tree_feature_importance.png")
FOREST_IMPORTANCE_PATH = Path("Figures/random_forest_feature_importance.png")
LOGISTIC_COEFFICIENT_PATH = Path("Figures/logistic_regression_coefficients.png")
LASSO_COEFFICIENT_PATH = Path("Figures/lasso_logistic_regression_coefficients.png")
PALETTE = ["#A8DFF3", "#F4BABF"]
ROC_PALETTE = ["#1F77B4", "#E15759", "#2CA02C", "#9467BD", "#FF9F1C"]


def load_dataset(path: Path, target_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_scores(model, x_data):
    """Return continuous scores for thresholding and ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_data)
    return model.predict(x_data)


def find_best_threshold(y_true, y_score):
    """Choose the threshold that maximizes F1 on the training split."""
    candidate_thresholds = np.unique(np.append(np.round(y_score, 6), 0.5))
    if candidate_thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def evaluate_model(model_name, model, x_test, y_test, threshold):
    """Compute evaluation metrics on the holdout set with a chosen threshold."""
    y_score = get_scores(model, x_test)
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score),
        "ConfusionMatrix": matrix,
        "Scores": y_score,
    }


def evaluate_model_at_default_threshold(model_name, model, x_test, y_test):
    """Evaluate the model with the standard 0.5 cutoff."""
    return evaluate_model(model_name, model, x_test, y_test, threshold=0.5)


def print_metric_table(results):
    """Print a readable comparison table for all models."""
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["ROC-AUC", "F1-score"], ascending=False
    ).reset_index(drop=True)

    formatted_df = results_df.copy()
    metric_columns = ["Threshold", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    for column in metric_columns:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.4f}")

    print("\nModel Comparison (sorted by ROC-AUC, then F1-score)")
    print(
        formatted_df[
            ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        ].to_string(index=False)
    )
    return results_df


def plot_metric_table(results_df, title, output_path):
    """Save a booktabs-style summary table as an image."""
    display_columns = [
        "Model",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]
    formatted_df = results_df[display_columns].copy()
    metric_columns = display_columns[1:]
    for column in metric_columns:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.4f}")

    row_count = len(formatted_df)
    figure_height = 1.35 + row_count * 0.42
    figure, axis = plt.subplots(figsize=(8.4, figure_height))
    axis.axis("off")

    table = axis.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.28, 0.105, 0.105, 0.105, 0.095, 0.105, 0.105],
        bbox=[0.02, 0.08, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.12)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
        else:
            if column == 0:
                cell.set_text_props(weight="semibold", color="black")

    # Draw booktabs-style horizontal rules only.
    x0, y0, width, height = 0.02, 0.08, 0.96, 0.72
    row_height = height / (row_count + 1)
    axis.hlines(
        y=y0 + height,
        xmin=x0,
        xmax=x0 + width,
        colors="black",
        linewidth=1.5,
        transform=axis.transAxes,
    )
    axis.hlines(
        y=y0 + height - row_height,
        xmin=x0,
        xmax=x0 + width,
        colors="black",
        linewidth=0.8,
        transform=axis.transAxes,
    )
    axis.hlines(
        y=y0,
        xmin=x0,
        xmax=x0 + width,
        colors="black",
        linewidth=1.5,
        transform=axis.transAxes,
    )

    axis.text(
        0.02,
        0.9,
        title,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def print_threshold_comparison(model_name, best_threshold):
    """Explain whether threshold tuning improved over the default cutoff."""
    if np.isclose(best_threshold, 0.5):
        print(f"{model_name}: best threshold was 0.5000 (same as default)")
    else:
        print(
            f"{model_name}: best threshold improved over 0.5000 "
            f"to {best_threshold:.4f}"
        )


def print_coefficients(model, feature_names, title):
    coefficients = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_.ravel(),
        }
    ).sort_values(by="Coefficient", key=np.abs, ascending=False)

    print(f"\n{title}")
    print(coefficients.to_string(index=False))
    return coefficients


def print_selected_features(model, feature_names):
    coefficients = pd.Series(model.coef_.ravel(), index=feature_names)
    selected = coefficients[coefficients != 0].sort_values(
        key=lambda series: np.abs(series), ascending=False
    )

    print("\nLASSO Selected Features (non-zero coefficients)")
    if selected.empty:
        print("No features were selected; all coefficients are zero.")
    else:
        print(selected.rename("Coefficient").reset_index().rename(
            columns={"index": "Feature"}
        ).to_string(index=False))

    return selected


def print_feature_importance(importances, feature_names, title):
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    print(f"\n{title}")
    print(importance_df.to_string(index=False))
    return importance_df


def plot_feature_importance(importance_df, title, output_path, top_n=None):
    """Save a horizontal bar chart for feature importance rankings."""
    plot_df = importance_df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.sort_values(by="Importance", ascending=True)
    figure_height = max(4.5, 0.45 * len(plot_df) + 1.2)
    figure, axis = plt.subplots(figsize=(7, figure_height))

    axis.barh(plot_df["Feature"], plot_df["Importance"], color=PALETTE[0])
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)

    x_max = plot_df["Importance"].max() if not plot_df.empty else 0
    x_offset = max(x_max * 0.02, 0.001)
    axis.set_xlim(0, x_max + x_offset * 8)

    for _, row in plot_df.iterrows():
        axis.text(
            row["Importance"] + x_offset,
            row["Feature"],
            f"{row['Importance']:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            color="black",
        )

    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)

    figure.subplots_adjust(left=0.34, right=0.97)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_coefficients(coefficients_df, title, output_path, top_n=None):
    """Save a horizontal bar chart for signed model coefficients."""
    plot_df = coefficients_df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plot_df = plot_df.sort_values(by="Coefficient", ascending=True)
    colors = [PALETTE[1] if value < 0 else PALETTE[0] for value in plot_df["Coefficient"]]
    figure_height = max(4.5, 0.5 * len(plot_df) + 1.2)
    figure, axis = plt.subplots(figsize=(6, figure_height))

    axis.barh(plot_df["Feature"], plot_df["Coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Coefficient")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)

    min_value = plot_df["Coefficient"].min() if not plot_df.empty else 0
    max_value = plot_df["Coefficient"].max() if not plot_df.empty else 0
    span = max(abs(min_value), abs(max_value))
    x_offset = max(span * 0.03, 0.01)
    left_padding = x_offset * 16
    right_padding = x_offset * 10
    axis.set_xlim(min_value - left_padding, max_value + right_padding)

    for _, row in plot_df.iterrows():
        value = row["Coefficient"]
        if value >= 0:
            x_position = value + x_offset
            alignment = "left"
        else:
            x_position = value - x_offset
            alignment = "right"

        axis.text(
            x_position,
            row["Feature"],
            f"{value:.4f}",
            va="center",
            ha=alignment,
            fontsize=10,
            color="black",
        )

    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)

    figure.subplots_adjust(left=0.34, right=0.97)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def print_confusion_matrices(results):
    print("\nConfusion Matrices")
    for result in results:
        tn, fp, fn, tp = result["ConfusionMatrix"].ravel()
        print(f"\n{result['Model']} (threshold={result['Threshold']:.4f})")
        print(result["ConfusionMatrix"])
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(10, 7))
    for index, result in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, result["Scores"])
        plt.plot(
            fpr,
            tpr,
            color=ROC_PALETTE[index % len(ROC_PALETTE)],
            linewidth=2.6,
            label=f"{result['Model']} (AUC={result['ROC-AUC']:.4f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="#9E9E9E", linewidth=1.2)
    plt.title("ROC Curves for Churn Prediction Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D9D9D9")
    plt.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(ROC_FIGURE_PATH, dpi=300)
    plt.close()


def plot_confusion_matrices(results):
    figure, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()

    for axis, result in zip(axes, results):
        matrix = result["ConfusionMatrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{result['Model']}\nThreshold={result['Threshold']:.3f}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    matrix[row, column],
                    ha="center",
                    va="center",
                    color="black",
                )

    if len(results) < len(axes):
        for axis in axes[len(results):]:
            axis.axis("off")

    figure.colorbar(image, ax=axes.tolist(), shrink=0.8)
    figure.suptitle("Confusion Matrices for Churn Prediction Models", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()


def main():
    try:
        df = load_dataset(DATA_PATH, TARGET_COLUMN)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = x.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    default_threshold_results = []

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(
        "\nNote: The dataset is already z-score normalized, "
        "so no additional normalization is applied."
    )

    logistic_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    logistic_model.fit(x_train, y_train)
    logistic_threshold = find_best_threshold(y_train, get_scores(logistic_model, x_train))
    print_threshold_comparison("Logistic Regression", logistic_threshold)
    default_threshold_results.append(
        evaluate_model_at_default_threshold(
            "Logistic Regression",
            logistic_model,
            x_test,
            y_test,
        )
    )
    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            x_test,
            y_test,
            logistic_threshold,
        )
    )
    logistic_coefficients_df = print_coefficients(
        logistic_model,
        feature_names,
        "Logistic Regression Coefficients",
    )
    plot_coefficients(
        logistic_coefficients_df,
        "Top 5 Logistic Regression Coefficients",
        LOGISTIC_COEFFICIENT_PATH,
        top_n=5,
    )

    lasso_model = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10),
        cv=cv,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        max_iter=3000,
        random_state=RANDOM_STATE,
        refit=True,
        class_weight="balanced",
    )
    lasso_model.fit(x_train, y_train)
    lasso_threshold = find_best_threshold(y_train, get_scores(lasso_model, x_train))
    print_threshold_comparison("LASSO Logistic Regression", lasso_threshold)
    default_threshold_results.append(
        evaluate_model_at_default_threshold(
            "LASSO Logistic Regression",
            lasso_model,
            x_test,
            y_test,
        )
    )
    results.append(
        evaluate_model(
            "LASSO Logistic Regression",
            lasso_model,
            x_test,
            y_test,
            lasso_threshold,
        )
    )
    print(f"\nBest inverse regularization strength (C) for LASSO: {lasso_model.C_[0]:.6f}")
    lasso_selected_df = print_selected_features(lasso_model, feature_names)
    if not lasso_selected_df.empty:
        plot_coefficients(
            lasso_selected_df.rename("Coefficient").reset_index().rename(
                columns={"index": "Feature"}
            ),
            "Top 5 LASSO Logistic Regression Coefficients",
            LASSO_COEFFICIENT_PATH,
            top_n=5,
        )

    tree_grid = GridSearchCV(
        estimator=DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        param_grid={
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    tree_grid.fit(x_train, y_train)
    best_tree = tree_grid.best_estimator_
    tree_threshold = find_best_threshold(y_train, get_scores(best_tree, x_train))
    print_threshold_comparison("Decision Tree", tree_threshold)
    default_threshold_results.append(
        evaluate_model_at_default_threshold(
            "Decision Tree",
            best_tree,
            x_test,
            y_test,
        )
    )
    results.append(
        evaluate_model("Decision Tree", best_tree, x_test, y_test, tree_threshold)
    )
    print(f"\nBest Decision Tree parameters: {tree_grid.best_params_}")
    tree_importance_df = print_feature_importance(
        best_tree.feature_importances_,
        feature_names,
        "Decision Tree Feature Importance",
    )
    plot_feature_importance(
        tree_importance_df,
        "Decision Tree Feature Importance",
        TREE_IMPORTANCE_PATH,
        top_n=5,
    )

    forest_grid = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 5],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    forest_grid.fit(x_train, y_train)
    best_forest = forest_grid.best_estimator_
    forest_threshold = find_best_threshold(y_train, get_scores(best_forest, x_train))
    print_threshold_comparison("Random Forest", forest_threshold)
    default_threshold_results.append(
        evaluate_model_at_default_threshold(
            "Random Forest",
            best_forest,
            x_test,
            y_test,
        )
    )
    results.append(
        evaluate_model("Random Forest", best_forest, x_test, y_test, forest_threshold)
    )
    print(f"\nBest Random Forest parameters: {forest_grid.best_params_}")
    forest_importance_df = print_feature_importance(
        best_forest.feature_importances_,
        feature_names,
        "Random Forest Feature Importance",
    )
    plot_feature_importance(
        forest_importance_df,
        "Random Forest Feature Importance",
        FOREST_IMPORTANCE_PATH,
        top_n=5,
    )

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(best_forest)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_summary = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_summary = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "MeanAbsSHAP": shap_summary,
            }
        ).sort_values(by="MeanAbsSHAP", ascending=False)

        print("\nRandom Forest SHAP Summary (mean absolute SHAP values)")
        print(shap_df.to_string(index=False))
    except ImportError:
        print(
            "\nRandom Forest SHAP Summary skipped because the optional "
            "'shap' package is not installed."
        )
    except Exception as error:
        print(f"\nRandom Forest SHAP Summary skipped due to runtime issue: {error}")

    svm_grid = GridSearchCV(
        estimator=SVC(
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        param_grid={
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    svm_grid.fit(x_train, y_train)
    best_svm = svm_grid.best_estimator_
    svm_threshold = find_best_threshold(y_train, get_scores(best_svm, x_train))
    print_threshold_comparison("SVM", svm_threshold)
    default_threshold_results.append(
        evaluate_model_at_default_threshold(
            "SVM",
            best_svm,
            x_test,
            y_test,
        )
    )
    results.append(evaluate_model("SVM", best_svm, x_test, y_test, svm_threshold))
    print(f"\nBest SVM parameters: {svm_grid.best_params_}")

    print("\nDefault Threshold Results (test set, threshold = 0.5000)")
    default_ranked_results = print_metric_table(default_threshold_results)
    plot_metric_table(
        default_ranked_results,
        "Model Comparison at Default Threshold = 0.5000",
        DEFAULT_METRIC_TABLE_PATH,
    )
    print("\nTrain-Optimized Threshold Results (evaluated on test set)")
    ranked_results = print_metric_table(results)
    plot_metric_table(
        ranked_results,
        "Model Comparison (sorted by ROC-AUC, then F1-score)",
        METRIC_TABLE_PATH,
    )
    print_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results)
    print(
        f"Default-threshold model comparison table saved to: "
        f"{DEFAULT_METRIC_TABLE_PATH.resolve()}"
    )
    print(f"Model comparison table saved to: {METRIC_TABLE_PATH.resolve()}")
    print(f"Logistic Regression coefficients saved to: {LOGISTIC_COEFFICIENT_PATH.resolve()}")
    if not lasso_selected_df.empty:
        print(f"LASSO coefficients saved to: {LASSO_COEFFICIENT_PATH.resolve()}")
    print(f"Decision Tree feature importance saved to: {TREE_IMPORTANCE_PATH.resolve()}")
    print(f"Random Forest feature importance saved to: {FOREST_IMPORTANCE_PATH.resolve()}")
    print(f"\nROC curves saved to: {ROC_FIGURE_PATH.resolve()}")
    print(f"Confusion matrices saved to: {CONFUSION_MATRIX_PATH.resolve()}")
    best_by_roc_auc = ranked_results.iloc[0]
    best_by_f1 = ranked_results.sort_values(by="F1-score", ascending=False).iloc[0]

    print("\nSummary")
    print(
        f"Best model by ROC-AUC: {best_by_roc_auc['Model']} "
        f"({best_by_roc_auc['ROC-AUC']:.4f})"
    )
    print(
        f"Best model by F1-score: {best_by_f1['Model']} "
        f"({best_by_f1['F1-score']:.4f})"
    )


if __name__ == "__main__":
    main()
