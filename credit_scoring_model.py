"""
================================================================================
  CREDIT SCORING MODEL — COMPLETE ML PROJECT
  Predicting Creditworthiness Using Machine Learning
================================================================================
  Author  : Credit Risk Analytics Project
  Purpose : Classify loan applicants as creditworthy or high-risk using
            Logistic Regression, Decision Tree, and Random Forest classifiers.
  Dataset : Synthetically generated (1 000 applicants, 12 features)
  Metrics : Accuracy, Precision, Recall, F1-Score, ROC-AUC
================================================================================
"""

# ─────────────────────────────────────────────
#  1.  IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
#  2.  SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
def generate_credit_dataset(n=1000, seed=42):
    """Generate a realistic credit-risk dataset."""
    rng = np.random.default_rng(seed)

    age             = rng.integers(21, 70, n)
    income          = rng.normal(55_000, 20_000, n).clip(15_000, 200_000).astype(int)
    employment_yrs  = rng.integers(0, 35, n)
    loan_amount     = rng.integers(2_000, 80_000, n)
    loan_term       = rng.choice([12, 24, 36, 48, 60], n)
    num_credit_lines= rng.integers(1, 10, n)
    num_late_payments= rng.integers(0, 10, n)
    credit_util_pct = rng.uniform(0, 100, n).round(2)
    existing_debt   = rng.normal(18_000, 10_000, n).clip(0, 80_000).astype(int)
    num_inquiries   = rng.integers(0, 8, n)
    education_level = rng.choice(["High School", "Bachelor", "Master", "PhD"], n,
                                  p=[0.30, 0.40, 0.20, 0.10])
    home_ownership  = rng.choice(["Rent", "Own", "Mortgage"], n,
                                  p=[0.35, 0.25, 0.40])

    # Derived credit score proxy (300-850)
    credit_score = (
        700
        - num_late_payments * 30
        - credit_util_pct * 0.8
        + income / 5_000
        - existing_debt / 5_000
        - num_inquiries * 10
        + employment_yrs * 2
    ).clip(300, 850).astype(int)

    # Debt-to-income ratio
    dti = (existing_debt + loan_amount) / income

    # Label: 1 = Good credit (creditworthy), 0 = Default risk
    default_prob = 1 / (1 + np.exp(
        -(
          -3
          + num_late_payments * 0.5
          + credit_util_pct * 0.03
          + dti * 1.5
          - credit_score / 300
          + num_inquiries * 0.2
        )
    ))
    target = (rng.uniform(0, 1, n) > default_prob).astype(int)

    edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    own_map = {"Rent": 0, "Own": 1, "Mortgage": 2}

    df = pd.DataFrame({
        "age"               : age,
        "annual_income"     : income,
        "employment_years"  : employment_yrs,
        "loan_amount"       : loan_amount,
        "loan_term_months"  : loan_term,
        "num_credit_lines"  : num_credit_lines,
        "num_late_payments" : num_late_payments,
        "credit_utilization": credit_util_pct,
        "existing_debt"     : existing_debt,
        "num_inquiries"     : num_inquiries,
        "education_level"   : [edu_map[e] for e in education_level],
        "home_ownership"    : [own_map[h] for h in home_ownership],
        "credit_score"      : credit_score,
        "debt_to_income"    : dti.round(4),
        "creditworthy"      : target          # 1 = good, 0 = risky
    })
    return df


# ─────────────────────────────────────────────
#  3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_income_ratio"]        = (df["loan_amount"] / df["annual_income"]).round(4)
    df["monthly_debt_burden"]      = (df["existing_debt"] / (df["loan_term_months"] + 1)).round(2)
    df["late_payment_rate"]        = (df["num_late_payments"] / (df["num_credit_lines"] + 1)).round(4)
    df["credit_score_band"]        = pd.cut(
        df["credit_score"],
        bins=[299, 579, 669, 739, 799, 851],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["high_utilization"]         = (df["credit_utilization"] > 30).astype(int)
    df["has_late_payments"]        = (df["num_late_payments"] > 0).astype(int)
    df["experience_income_ratio"]  = (df["employment_years"] / (df["annual_income"] / 10_000)).round(4)
    return df


# ─────────────────────────────────────────────
#  4.  DATA EXPLORATION (saved as plots)
# ─────────────────────────────────────────────
def explore_and_plot(df: pd.DataFrame, out_dir: str):
    # 4a. Distribution of target
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Credit Dataset — Overview", fontsize=14, fontweight="bold")

    counts = df["creditworthy"].value_counts()
    axes[0].bar(["Default Risk (0)", "Creditworthy (1)"], counts.values,
                color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=0.8)
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    df.groupby("creditworthy")["credit_score"].plot.kde(ax=axes[1], legend=True,
        color=["#e74c3c", "#2ecc71"])
    axes[1].set_title("Credit Score Distribution by Class")
    axes[1].set_xlabel("Credit Score")
    axes[1].legend(["Default Risk", "Creditworthy"])

    corr = df.drop(columns=["creditworthy"]).corr()
    top_corr = corr["credit_score"].abs().nlargest(8).index
    df[top_corr].corr().pipe(lambda c: sns.heatmap(c, ax=axes[2], annot=True,
        fmt=".2f", cmap="coolwarm", linewidths=0.5, annot_kws={"size": 7}))
    axes[2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_data_overview.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 4b. Feature relationships
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    features = ["annual_income", "credit_utilization", "existing_debt",
                "num_late_payments", "debt_to_income", "loan_amount"]
    for ax, feat in zip(axes.flat, features):
        df.boxplot(column=feat, by="creditworthy", ax=ax, patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1", color="steelblue"),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("0 = Default Risk | 1 = Creditworthy")
        ax.set_ylabel("")
    plt.suptitle("Feature Distributions by Creditworthiness", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_feature_distributions.png", dpi=140, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
#  5.  MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=42, C=0.5))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_split=20,
                                           class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                           min_samples_split=15, class_weight="balanced",
                                           random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                               max_depth=4, random_state=42))
        ]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_prob  = pipe.predict_proba(X_test)[:, 1]
        cv_auc  = cross_val_score(pipe, X_train, y_train, cv=cv,
                                  scoring="roc_auc").mean()
        results[name] = {
            "pipeline" : pipe,
            "y_pred"   : y_pred,
            "y_prob"   : y_prob,
            "accuracy" : accuracy_score(y_test, y_pred),
            "roc_auc"  : roc_auc_score(y_test, y_prob),
            "cv_auc"   : cv_auc,
            "f1"       : f1_score(y_test, y_pred, average="weighted"),
            "report"   : classification_report(y_test, y_pred,
                             target_names=["Default Risk", "Creditworthy"])
        }
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        print(f"  Accuracy : {results[name]['accuracy']:.4f}")
        print(f"  ROC-AUC  : {results[name]['roc_auc']:.4f}  (CV: {cv_auc:.4f})")
        print(f"  F1-Score : {results[name]['f1']:.4f}")
        print(f"\n{results[name]['report']}")

    return results


# ─────────────────────────────────────────────
#  6.  VISUALISATIONS — EVALUATION
# ─────────────────────────────────────────────
def plot_evaluation(results: dict, y_test, out_dir: str):
    palette = {
        "Logistic Regression": "#3498db",
        "Decision Tree"      : "#e67e22",
        "Random Forest"      : "#2ecc71",
        "Gradient Boosting"  : "#9b59b6",
    }

    # ── 6a. Confusion matrices ──────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Default", "Creditworthy"],
                    yticklabels=["Default", "Creditworthy"],
                    cbar=False, linewidths=1)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── 6b. ROC curves ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                     color=palette[name], linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves Comparison", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, label=name, color=palette[name], linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_roc_pr_curves.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── 6c. Metrics bar chart ──────────────────
    metrics_df = pd.DataFrame({
        "Model"   : list(results.keys()),
        "Accuracy": [r["accuracy"] for r in results.values()],
        "ROC-AUC" : [r["roc_auc"]  for r in results.values()],
        "F1-Score": [r["f1"]       for r in results.values()],
        "CV AUC"  : [r["cv_auc"]   for r in results.values()],
    }).set_index("Model")

    fig, ax = plt.subplots(figsize=(11, 5))
    x    = np.arange(len(metrics_df))
    w    = 0.2
    cols = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, (col, label) in enumerate(zip(cols, metrics_df.columns)):
        bars = ax.bar(x + i * w, metrics_df[label], w, label=label,
                      color=col, edgecolor="white", alpha=0.9)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(metrics_df.index, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/5_model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    return metrics_df


# ─────────────────────────────────────────────
#  7.  FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
def plot_feature_importance(results: dict, feature_names: list, out_dir: str):
    rf_clf = results["Random Forest"]["pipeline"].named_steps["clf"]
    imp    = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = ["#e74c3c" if v > imp.median() else "#3498db" for v in imp.values]
    bars    = ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.axvline(imp.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {imp.median():.4f}")
    ax.legend()
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/6_feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
#  8.  CREDIT SCORE BAND ANALYSIS
# ─────────────────────────────────────────────
def plot_score_band_analysis(df: pd.DataFrame, out_dir: str):
    bands      = ["Very Poor\n(300-579)", "Fair\n(580-669)", "Good\n(670-739)",
                  "Very Good\n(740-799)", "Excellent\n(800-850)"]
    band_codes = [0, 1, 2, 3, 4]
    df["band_label"] = df["credit_score_band"].map(dict(zip(band_codes, bands)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Credit Score Band Analysis", fontsize=13, fontweight="bold")

    grouped  = df.groupby("credit_score_band")["creditworthy"].mean().reset_index()
    counts   = df.groupby("credit_score_band").size().reset_index(name="n")
    combined = grouped.merge(counts)
    actual_bands = [bands[b] for b in combined["credit_score_band"].tolist()]
    c_palette_all = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    c_palette_used = [c_palette_all[b] for b in combined["credit_score_band"].tolist()]
    axes[0].bar(range(len(combined)), combined["creditworthy"], color=c_palette_used, edgecolor="white")
    axes[0].set_ylabel("Creditworthy Rate")
    axes[0].set_title("Creditworthy Rate per Band")
    axes[0].set_xticks(range(len(combined)))
    axes[0].set_xticklabels(actual_bands, fontsize=8)
    for i, row in combined.iterrows():
        axes[0].text(i - combined.index[0], row["creditworthy"] + 0.01, f'{row["creditworthy"]:.0%}',
                     ha="center", fontweight="bold")

    df.groupby(["credit_score_band", "creditworthy"]).size().unstack().plot(
        kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"], edgecolor="white")
    axes[1].set_xticklabels(actual_bands, rotation=0, fontsize=8)
    axes[1].set_title("Count of Applicants per Band")
    axes[1].set_ylabel("Count")
    axes[1].legend(["Default Risk", "Creditworthy"])

    plt.tight_layout()
    plt.savefig(f"{out_dir}/7_score_band_analysis.png", dpi=140, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
#  9.  SUMMARY REPORT TABLE
# ─────────────────────────────────────────────
def print_summary(metrics_df: pd.DataFrame):
    best = metrics_df["ROC-AUC"].idxmax()
    print("\n" + "═" * 60)
    print("   CREDIT SCORING MODEL — FINAL SUMMARY REPORT")
    print("═" * 60)
    print(metrics_df.round(4).to_string())
    print(f"\n  ✅  Best Model by ROC-AUC : {best}")
    print(f"  ROC-AUC  = {metrics_df.loc[best,'ROC-AUC']:.4f}")
    print(f"  Accuracy = {metrics_df.loc[best,'Accuracy']:.4f}")
    print(f"  F1-Score = {metrics_df.loc[best,'F1-Score']:.4f}")
    print("═" * 60)


# ─────────────────────────────────────────────
#  10.  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    import os
    out_dir = "/mnt/user-data/outputs/credit_scoring_project"
    os.makedirs(out_dir, exist_ok=True)

    print("\n[ 1/7 ] Generating synthetic credit dataset …")
    df = generate_credit_dataset(n=1000, seed=42)
    print(f"        Dataset shape: {df.shape} | Default rate: {1-df['creditworthy'].mean():.1%}")

    print("[ 2/7 ] Engineering features …")
    df = engineer_features(df)
    df.to_csv(f"{out_dir}/credit_dataset.csv", index=False)
    print(f"        Feature count after engineering: {df.shape[1]-1}")

    print("[ 3/7 ] Generating exploratory plots …")
    explore_and_plot(df, out_dir)

    feature_cols = [c for c in df.columns if c != "creditworthy"]
    X = df[feature_cols].values
    y = df["creditworthy"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"        Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("[ 4/7 ] Training & evaluating models …")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("[ 5/7 ] Plotting evaluation charts …")
    metrics_df = plot_evaluation(results, y_test, out_dir)

    print("[ 6/7 ] Plotting feature importance …")
    plot_feature_importance(results, feature_cols, out_dir)

    print("[ 7/7 ] Score-band analysis …")
    plot_score_band_analysis(df, out_dir)

    print_summary(metrics_df)

    metrics_df.reset_index().to_csv(f"{out_dir}/model_metrics.csv", index=False)
    print(f"\n  All outputs saved to: {out_dir}\n")

if __name__ == "__main__":
    main()
