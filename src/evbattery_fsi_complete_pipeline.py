"""Standalone EVBattery FSI processing and analysis pipeline.

Inputs (final_results/data/):
  evbattery_all_vehicles_fsi.csv
  evbattery_vehicle_fsi.csv

Run:
  python src/evbattery_fsi_complete_pipeline.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent   # final_results/
OUT = ROOT / "output"
GLOBAL_INPUT = ROOT / "data" / "evbattery_all_vehicles_fsi.csv"
COHORT_INPUT = ROOT / "data" / "evbattery_vehicle_fsi.csv"
BASE = ["car_id", "mileage_km", "capacity_ah", "soh_pct", "health_label",
        "anomaly_flag", "ki_mean", "dod_mean", "t_avg_c", "c_rate_mean",
        "snippet_count", "ki_mean_norm", "dod_mean_norm", "t_avg_c_norm",
        "c_rate_mean_norm", "FSI"]
NUMERIC = [c for c in BASE if c != "health_label"]
FEATURES = ["FSI", "ki_mean", "dod_mean", "t_avg_c", "c_rate_mean", "mileage_km"]
COLORS = {"dataset1": "#1f77b4", "dataset2": "#ff7f0e", "dataset3": "#2ca02c"}


def clean(frame, has_dataset, source):
    required = (["dataset"] if has_dataset else []) + BASE
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")
    frame = frame[required].copy()
    if has_dataset:
        frame["dataset"] = frame.dataset.astype(str).str.strip()
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["car_id"] = frame.car_id.astype("Int64")
    frame["anomaly_flag"] = frame.anomaly_flag.fillna(0).astype("Int64")
    frame["health_label"] = frame.health_label.astype(str).str.strip()
    keys = ["dataset", "car_id"] if has_dataset else ["car_id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{source} contains duplicate keys: {keys}")
    return frame


def correlations(data):
    rows = []
    candidates = FEATURES + ["capacity_ah", "ki_mean_norm", "dod_mean_norm",
                              "t_avg_c_norm", "c_rate_mean_norm"]
    for dataset, group in list(data.groupby("dataset")) + [("ALL", data)]:
        for x in candidates:
            for y in ["soh_pct", "capacity_ah"]:
                if x == y:
                    continue
                pair = group[[x, y]].dropna()
                if len(pair) < 5 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
                    continue
                rho, sp = spearmanr(pair[x], pair[y]); r, pp = pearsonr(pair[x], pair[y])
                rows.append({"dataset": dataset, "feature": x, "target": y,
                             "samples": len(pair), "spearman_rho": rho,
                             "spearman_p": sp, "pearson_r": r, "pearson_p": pp})
    return pd.DataFrame(rows)


def importance(data):
    rows = []
    for dataset, group in list(data.groupby("dataset")) + [("ALL", data)]:
        sample = group[FEATURES + ["soh_pct"]].dropna()
        if len(sample) < 10:
            continue
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42)
        model.fit(sample[FEATURES], sample.soh_pct)
        perm = permutation_importance(model, sample[FEATURES], sample.soh_pct,
                                      n_repeats=20, random_state=42)
        for feature, mdi, pi in zip(FEATURES, model.feature_importances_, perm.importances_mean):
            rows.append({"dataset": dataset, "feature": feature,
                         "rf_importance": mdi, "permutation_importance": pi})
    return pd.DataFrame(rows).sort_values(["dataset", "permutation_importance"], ascending=[True, False])


def validate_model(data):
    sample = data[["dataset", "car_id", "soh_pct"] + FEATURES].dropna().copy()
    cv = GroupKFold(n_splits=sample.dataset.nunique())
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42)
    prediction = cross_val_predict(model, sample[FEATURES], sample.soh_pct,
                                    groups=sample.dataset, cv=cv)
    result = sample[["dataset", "car_id", "soh_pct"]].rename(columns={"soh_pct": "soh_actual"})
    result["soh_predicted_leave_one_dataset_out"] = prediction
    result["residual"] = result.soh_actual - prediction
    metrics = {"mae_pct_points": float(mean_absolute_error(result.soh_actual, prediction)),
               "rmse_pct_points": float(np.sqrt(mean_squared_error(result.soh_actual, prediction))),
               "r2": float(r2_score(result.soh_actual, prediction))}
    return result, metrics


def save_plots(data, comparison, ranking, prediction):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [("FSI", "FSI"), ("mileage_km", "Mileage (km)"),
             ("c_rate_mean", "Mean C-rate"), ("t_avg_c", "Mean temperature (°C)")]
    for ax, (column, label) in zip(axes.flat, specs):
        for dataset, group in data.groupby("dataset"):
            ax.scatter(group[column], group.soh_pct, s=28, alpha=.7,
                       label=dataset, color=COLORS.get(dataset))
        ax.set(xlabel=label, ylabel="SoH (%)", title=f"{label} vs measured SoH")
        ax.set_ylim(0, 100); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "Fig_EVBattery_FSI_SoH_Correlations.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(comparison.FSI_global, comparison.FSI_cohort,
                        c=comparison.soh_pct_global, cmap="viridis_r", s=35)
    low = min(comparison.FSI_global.min(), comparison.FSI_cohort.min())
    high = max(comparison.FSI_global.max(), comparison.FSI_cohort.max())
    ax.plot([low, high], [low, high], "k--", label="1:1")
    ax.set(xlabel="FSI normalized across datasets 1–3",
           ylabel="FSI normalized within dataset 1",
           title="Global vs cohort-normalized FSI")
    fig.colorbar(points, ax=ax, label="Measured SoH (%)"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "Fig_EVBattery_Global_vs_Cohort_FSI.png", dpi=180); plt.close(fig)

    overall = ranking[ranking.dataset.eq("ALL")].sort_values("permutation_importance")
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(overall.feature, overall.permutation_importance, color="#4472c4")
    ax.set(xlabel="Permutation importance for measured SoH",
           title="EVBattery degradation-feature importance"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "Fig_EVBattery_Degradation_Importance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for dataset, group in prediction.groupby("dataset"):
        ax.scatter(group.soh_actual, group.soh_predicted_leave_one_dataset_out,
                   label=dataset, alpha=.7, color=COLORS.get(dataset))
    low = min(prediction.soh_actual.min(), prediction.soh_predicted_leave_one_dataset_out.min())
    high = max(prediction.soh_actual.max(), prediction.soh_predicted_leave_one_dataset_out.max())
    ax.plot([low, high], [low, high], "k--", label="Ideal")
    ax.set(xlabel="Measured SoH (%)", ylabel="Predicted SoH (%)",
           title="Leave-one-dataset-out SoH validation"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "Fig_EVBattery_SoH_Validation.png", dpi=180); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = clean(pd.read_csv(GLOBAL_INPUT), True, GLOBAL_INPUT.name)
    cohort = clean(pd.read_csv(COHORT_INPUT), False, COHORT_INPUT.name)
    comparison = data[data.dataset.eq("dataset1")].merge(
        cohort, on="car_id", how="outer", suffixes=("_global", "_cohort"),
        indicator=True, validate="one_to_one")
    comparison["fsi_delta_cohort_minus_global"] = comparison.FSI_cohort - comparison.FSI_global
    comparison["fsi_rank_global"] = comparison.FSI_global.rank(ascending=False, method="min")
    comparison["fsi_rank_cohort"] = comparison.FSI_cohort.rank(ascending=False, method="min")
    comparison["fsi_rank_shift"] = comparison.fsi_rank_cohort - comparison.fsi_rank_global

    quality = []
    for source, frame in [("all_vehicles", data), ("dataset1_cohort", cohort)]:
        for col in frame.columns:
            quality.append({"source": source, "column": col, "rows": len(frame),
                            "missing_count": int(frame[col].isna().sum()),
                            "missing_pct": float(100 * frame[col].isna().mean()),
                            "unique_count": int(frame[col].nunique(dropna=True))})
    corr = correlations(data); rank = importance(data); pred, metrics = validate_model(data)
    summary = data.groupby("dataset", as_index=False).agg(
        vehicles=("car_id", "nunique"), mean_mileage_km=("mileage_km", "mean"),
        mean_capacity_ah=("capacity_ah", "mean"), mean_soh_pct=("soh_pct", "mean"),
        mean_fsi=("FSI", "mean"), p95_fsi=("FSI", lambda x: x.quantile(.95)),
        anomalies=("anomaly_flag", "sum"))

    data.to_csv(OUT / "evbattery_canonical_clean.csv", index=False)
    cohort.to_csv(OUT / "evbattery_dataset1_cohort_clean.csv", index=False)
    comparison.to_csv(OUT / "evbattery_global_vs_vehicle_fsi_comparison.csv", index=False)
    pd.DataFrame(quality).to_csv(OUT / "evbattery_data_quality.csv", index=False)
    corr.to_csv(OUT / "evbattery_fsi_health_correlations.csv", index=False)
    rank.to_csv(OUT / "evbattery_degradation_feature_importance.csv", index=False)
    pred.to_csv(OUT / "evbattery_soh_cross_dataset_validation.csv", index=False)
    summary.to_csv(OUT / "evbattery_dataset_summary.csv", index=False)
    save_plots(data, comparison[comparison._merge.eq("both")], rank, pred)

    fsi = corr[(corr.dataset == "ALL") & (corr.feature == "FSI") & (corr.target == "soh_pct")].iloc[0]
    report = {"canonical_rows": int(len(data)), "cohort_rows": int(len(cohort)),
              "overlap_rows": int((comparison._merge == "both").sum()),
              "duplicate_policy": "all-vehicles is canonical; cohort file is sensitivity-only",
              "fsi_soh_spearman_rho": float(fsi.spearman_rho),
              "fsi_soh_p_value": float(fsi.spearman_p),
              "cross_dataset_validation": metrics,
              "warning": "Associations are observational; dataset normalization can confound transfer."}
    (OUT / "evbattery_complete_pipeline_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Complete EVBattery pipeline finished: {len(data)} canonical vehicles")
    print(f"Results saved in: {OUT}")


if __name__ == "__main__":
    main()
