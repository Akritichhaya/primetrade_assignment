#!/usr/bin/env python3
"""
Fear/Greed vs Trader Performance (Hyperliquid) — End-to-End Analysis + Simple ML

Usage:
  python Primetrade.py \
     D:\Interview\Primetrade\fear_greed_index.csv \
     D:\Interview\Primetrade\historical_data.csv\
     D:\Interview\Primetrade\Output
     

What it does:
1) Loads Bitcoin Fear & Greed Index and Hyperliquid historical trades.
2) Cleans/standardizes columns, derives trade DATE from timestamps.
3) Joins trades with same-day sentiment (Fear/Greed).
4) Computes performance metrics (PnL, win rate, etc.) by sentiment, symbol, side, account.
5) Saves CSV tables and charts.
6) Trains simple ML models to predict trade win (PnL>0) from sentiment + trade features.

Author: Akriti 
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json

import os
import argparse
import math
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- ML libs (standard, lightweight) ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Helpers
# ----------------------------

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def to_date_safe(series: pd.Series) -> pd.Series:
    """Try to convert to datetime; if numeric epoch, detect ms vs s; return date."""
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().sum() == 0:
        # Try epoch seconds vs ms
        try:
            med = pd.to_numeric(series, errors="coerce").dropna().astype(float).median()
            if np.isnan(med):
                return pd.NaT
            unit = "ms" if med > 1e12 else "s"
            s = pd.to_datetime(series, unit=unit, errors="coerce")
        except Exception:
            pass
    return s.dt.date

def welch_ttest(x, y):
    """Returns approximate t-stat, df, and p (normal approx if SciPy not available)."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().values.astype(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().values.astype(float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return {"t_stat": np.nan, "df": np.nan, "p_value_approx": np.nan}
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.var(ddof=1), y.var(ddof=1)
    t = (m1 - m2) / np.sqrt(s1/n1 + s2/n2)
    df = (s1/n1 + s2/n2)**2 / ((s1**2)/((n1**2)*(n1-1)) + (s2**2)/((n2**2)*(n2-1)))
    # normal approx for p-value
    from math import erf, sqrt
    z = abs(t)
    p_approx = 2 * (1 - 0.5*(1 + erf(z/np.sqrt(2))))
    return {"t_stat": float(t), "df": float(df), "p_value_approx": float(p_approx)}

def load_and_prepare(fgi_path: str, trades_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Read
    fgi = pd.read_csv(fgi_path, engine="python")
    trades = pd.read_csv(trades_path, engine="python")

    # Normalize cols
    fgi.columns = [c.strip().lower().replace(" ", "_") for c in fgi.columns]
    trades.columns = [c.strip().lower().replace(" ", "_") for c in trades.columns]

    # FGI: expect 'date' + 'classification'
    if "date" not in fgi.columns and "timestamp" in fgi.columns:
        fgi["date"] = pd.to_datetime(fgi["timestamp"], unit="s", errors="coerce").dt.date
    else:
        fgi["date"] = pd.to_datetime(fgi["date"], errors="coerce").dt.date
    fgi["classification"] = fgi["classification"].astype(str).str.strip().str.title()

    # Trades: derive date from timestamp columns
    if "timestamp_ist" in trades.columns:
        trades["trade_datetime"] = pd.to_datetime(trades["timestamp_ist"], errors="coerce")
    elif "timestamp" in trades.columns:
        # detect s vs ms
        med = pd.to_numeric(trades["timestamp"], errors="coerce").dropna().astype(float).median()
        unit = "ms" if med and med > 1e12 else "s"
        trades["trade_datetime"] = pd.to_datetime(trades["timestamp"], unit=unit, errors="coerce")
    else:
        # fallback: already have a 'date'?
        if "date" in trades.columns:
            trades["trade_datetime"] = pd.to_datetime(trades["date"], errors="coerce")
        else:
            trades["trade_datetime"] = pd.NaT

    trades["date"] = trades["trade_datetime"].dt.date

    # Make sure numeric types are numeric
    for col in ["execution_price", "size_tokens", "size_usd", "closed_pnl", "leverage", "start_position"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    # Merge on date
    merged = trades.merge(fgi[["date", "classification"]], on="date", how="left")
    merged_valid = merged.dropna(subset=["classification"]).copy()

    return fgi, trades, merged_valid

def compute_tables(merged_valid: pd.DataFrame, outdir: str) -> dict:
    # Summary by sentiment
    def win_rate(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return (s > 0).mean() if len(s) else np.nan

    group = merged_valid.groupby("classification")
    summary = pd.DataFrame({
        "trades": group.size(),
        "unique_accounts": group["account"].nunique() if "account" in merged_valid.columns else group.size(),
        "unique_symbols": group["coin"].nunique() if "coin" in merged_valid.columns else group.size(),
        "net_pnl": group["closed_pnl"].sum(min_count=1),
        "avg_pnl": group["closed_pnl"].mean(),
        "median_pnl": group["closed_pnl"].median(),
        "std_pnl": group["closed_pnl"].std(),
        "win_rate": group["closed_pnl"].apply(win_rate),
        "avg_size_tokens": group["size_tokens"].mean() if "size_tokens" in merged_valid.columns else np.nan,
        "avg_size_usd": group["size_usd"].mean() if "size_usd" in merged_valid.columns else np.nan,
    }).reset_index()

    summary_path = os.path.join(outdir, "summary_by_sentiment.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")


    # By symbol
    sym_col = "coin" if "coin" in merged_valid.columns else ("symbol" if "symbol" in merged_valid.columns else None)
    by_symbol = merged_valid.groupby(["classification", sym_col]).agg(
        trades=("closed_pnl", "size"),
        net_pnl=("closed_pnl", "sum"),
        avg_pnl=("closed_pnl", "mean"),
        win_rate=("closed_pnl", lambda s: (pd.to_numeric(s, errors="coerce").dropna() > 0).mean())
    ).reset_index().rename(columns={sym_col: "symbol"})
    by_symbol_path = os.path.join(outdir, "by_symbol_sentiment.csv")
    by_symbol.to_csv(by_symbol_path, index=False, encoding="utf-8")

    # Symbol delta table
    symbol_pivot = by_symbol.pivot_table(index="symbol", columns="classification", values="avg_pnl")
    symbol_pivot["delta_avg_pnl_Greed_minus_Fear"] = symbol_pivot.get("Greed") - symbol_pivot.get("Fear")
    symbol_pivot_sorted = symbol_pivot.sort_values("delta_avg_pnl_Greed_minus_Fear", ascending=False)
    symbol_delta_path = os.path.join(outdir, "symbol_pnl_delta_greed_minus_fear.csv")
    symbol_pivot_sorted.to_csv(symbol_delta_path, encoding="utf-8")


    # By account
    by_account = merged_valid.groupby(["classification", "account"]).agg(
        trades=("closed_pnl", "size"),
        net_pnl=("closed_pnl", "sum"),
        avg_pnl=("closed_pnl", "mean"),
        win_rate=("closed_pnl", lambda s: (pd.to_numeric(s, errors="coerce").dropna() > 0).mean())
    ).reset_index()

    acct_pivot = by_account.pivot_table(index="account", columns="classification", values=["avg_pnl", "win_rate"])
    if isinstance(acct_pivot.columns, pd.MultiIndex):
        acct_pivot.columns = ["_".join([str(c) for c in col if c != ""]) for col in acct_pivot.columns]

    def tag_style(row):
        fear = row.get("avg_pnl_Fear", np.nan)
        greed = row.get("avg_pnl_Greed", np.nan)
        if pd.isna(fear) or pd.isna(greed):
            return "insufficient_data"
        return "contrarian" if fear > greed else "momentum"

    acct_pivot["style"] = acct_pivot.apply(tag_style, axis=1)

    top_contrarian = acct_pivot[acct_pivot["style"]=="contrarian"].copy()
    top_contrarian["delta_avg_pnl_Fear_minus_Greed"] = top_contrarian.get("avg_pnl_Fear", np.nan) - top_contrarian.get("avg_pnl_Greed", np.nan)
    top_contrarian = top_contrarian.sort_values("delta_avg_pnl_Fear_minus_Greed", ascending=False).head(20)
    top_contrarian_path = os.path.join(outdir, "top_contrarian_accounts.csv")
    top_contrarian.to_csv(top_contrarian_path, encoding="utf-8")


    top_momentum = acct_pivot[acct_pivot["style"]=="momentum"].copy()
    top_momentum["delta_avg_pnl_Greed_minus_Fear"] = top_momentum.get("avg_pnl_Greed", np.nan) - top_momentum.get("avg_pnl_Fear", np.nan)
    top_momentum = top_momentum.sort_values("delta_avg_pnl_Greed_minus_Fear", ascending=False).head(20)
    top_momentum_path = os.path.join(outdir, "top_momentum_accounts.csv")
    top_momentum.to_csv(top_momentum_path)

    # Side-level
    side_summary_path = None
    if "side" in merged_valid.columns:
        side_summary = merged_valid.groupby(["classification", "side"]).agg(
            trades=("closed_pnl", "size"),
            net_pnl=("closed_pnl", "sum"),
            avg_pnl=("closed_pnl", "mean"),
            win_rate=("closed_pnl", lambda s: (pd.to_numeric(s, errors="coerce").dropna() > 0).mean()),
            avg_size_tokens=("size_tokens", "mean") if "size_tokens" in merged_valid.columns else ("size", "mean")
        ).reset_index()
        side_summary_path = os.path.join(outdir, "side_summary.csv")
        side_summary.to_csv(side_summary_path, index=False, encoding="utf-8")


    # Charts (no explicit colors/styles)
    plt.figure()
    plt.bar(summary["classification"].astype(str), summary["avg_pnl"].astype(float).fillna(0.0))
    plt.title("Average PnL by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Average PnL")
    plt.tight_layout()
    chart1 = os.path.join(outdir, "chart_avg_pnl_by_sentiment.png")
    plt.savefig(chart1); plt.close()

    plt.figure()
    plt.bar(summary["classification"].astype(str), (summary["win_rate"]*100.0).astype(float).fillna(0.0))
    plt.title("Win Rate by Sentiment (%)")
    plt.xlabel("Sentiment")
    plt.ylabel("Win Rate (%)")
    plt.tight_layout()
    chart2 = os.path.join(outdir, "chart_win_rate_by_sentiment.png")
    plt.savefig(chart2); plt.close()

    labels = summary["classification"].astype(str).tolist()
    data = [merged_valid.loc[merged_valid["classification"]==c, "closed_pnl"].dropna().values for c in labels]
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title("Closed PnL Distribution by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Closed PnL")
    plt.tight_layout()
    chart3 = os.path.join(outdir, "chart_boxplot_pnl_by_sentiment.png")
    plt.savefig(chart3); plt.close()

    # Welch t-test
    greed = merged_valid.loc[merged_valid["classification"]=="Greed", "closed_pnl"]
    fear  = merged_valid.loc[merged_valid["classification"]=="Fear", "closed_pnl"]
    ttest = welch_ttest(greed, fear)

    # Save text report
    def fmt_pct(x): return "NA" if pd.isna(x) else f"{x*100:.2f}%"
    lines = ["INSIGHTS SUMMARY", ""]
    for _, row in summary.iterrows():
        lines.append(f"{row['classification']}: trades={int(row['trades'])}, "
                     f"net_pnl={row['net_pnl']:.2f}, avg_pnl={row['avg_pnl']:.4f}, "
                     f"median_pnl={row['median_pnl']:.4f}, win_rate={fmt_pct(row['win_rate'])}")
    lines.append("")
    lines.append(
    f"Welch t-test (Greed vs Fear avg PnL): "
    f"t={ttest['t_stat']:.3f}, df~{ttest['df']:.1f}, p~{ttest['p_value_approx']:.4f}"
)

    report_path = os.path.join(outdir, "insights_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "summary_csv": summary_path,
        "by_symbol_csv": by_symbol_path,
        "symbol_delta_csv": symbol_delta_path,
        "top_contrarian_csv": top_contrarian_path,
        "top_momentum_csv": top_momentum_path,
        "side_summary_csv": side_summary_path,
        "charts": [chart1, chart2, chart3],
        "report": report_path
    }

def build_ml_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Build a supervised dataset to predict whether a trade is a 'win' (closed_pnl > 0).
    Features include: sentiment (classification), side, coin, leverage, size, execution_price, start_position.
    """
    data = df.copy()

    # Target
    y = (pd.to_numeric(data["closed_pnl"], errors="coerce") > 0).astype(int)

    # Candidate feature columns (robust to missing ones)
    num_cols = [c for c in ["execution_price", "size_tokens", "size_usd", "leverage", "start_position"] if c in data.columns]
    cat_cols = [c for c in ["classification", "side", "coin"] if c in data.columns]

    X = data[num_cols + cat_cols].copy()

    # Simple missing handling
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")

    return X, y, num_cols, cat_cols

def run_ml(merged_valid: pd.DataFrame, outdir: str) -> dict:
    X, y, num_cols, cat_cols = build_ml_dataset(merged_valid)

    # Train/Val split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Preprocess
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # 1) Logistic Regression
    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    logit.fit(X_train, y_train)
    y_pred_l = logit.predict(X_test)
    y_prob_l = logit.predict_proba(X_test)[:,1]

    logit_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_l),
        "roc_auc": roc_auc_score(y_test, y_prob_l),
        "report": classification_report(y_test, y_pred_l, output_dict=True)
    }

    # 2) Random Forest (robust nonlinear baseline)
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    rf.fit(X_train, y_train)
    y_pred_r = rf.predict(X_test)
    y_prob_r = rf.predict_proba(X_test)[:,1]

    rf_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_r),
        "roc_auc": roc_auc_score(y_test, y_prob_r),
        "report": classification_report(y_test, y_pred_r, output_dict=True)
    }

    # Save metrics as JSON + human-readable text
    metrics_path = os.path.join(outdir, "ml_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"logistic_regression": logit_metrics, "random_forest": rf_metrics}, f, indent=2)

    # Attempt to extract RF feature importances (approx; through OHE it's after preprocessor)
    # Instead, we can compute permutation importances or report only overall scores.
    # For simplicity, just save the metrics here.

    # Short note explaining the ML approach
    note = [
        "ML approach:",
        "- Target: 1 if closed_pnl > 0 else 0 (trade win).",
        "- Features: sentiment (Fear/Greed), side (Buy/Sell), coin, leverage, size, execution price, start position (when present).",
        "- Models: Logistic Regression (interpretable baseline) and Random Forest (nonlinear baseline).",
        "- Metrics: Accuracy and ROC-AUC on a 25% holdout set."
    ]
    with open(os.path.join(outdir, "ml_readme.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(note))

    return {
        "logit_accuracy": logit_metrics["accuracy"],
        "logit_roc_auc": logit_metrics["roc_auc"],
        "rf_accuracy": rf_metrics["accuracy"],
        "rf_roc_auc": rf_metrics["roc_auc"],
        "metrics_json": metrics_path,
        "ml_readme": os.path.join(outdir, "ml_readme.txt")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgi", default=r".\fear_greed_index.csv",
                    help="Path to fear_greed_index.csv")
    parser.add_argument("--trades", default=r".\historical_data.csv",
                    help="Path to historical_data.csv")
    parser.add_argument("--outdir", default=r".\Output",
                    help="Directory for outputs")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    # 1) Load + join
    fgi, trades, merged_valid = load_and_prepare(args.fgi, args.trades)
    if merged_valid.empty:
        print("No trades matched to Fear/Greed classifications by date. Check your input files.")
        return

    # 2) Tables + charts + test
    artifacts = compute_tables(merged_valid, outdir)

    # 3) Simple ML
    ml = run_ml(merged_valid, outdir)

    # 4) Print concise summary to stdout
    print("=== Artifacts ===")
    for k, v in artifacts.items():
        print(f"{k}: {v}")
    print("\n=== ML Metrics ===")
    for k, v in ml.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
