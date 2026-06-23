"""
CRISP-DM Phase 4: Modelling
Prescriptive DSS for Football Coaches — Morocco AFCON 2023

Two-stage pipeline (fully data-driven, no hardcoded rules):

  Stage A — Game-State Classifier
    KMeans clustering on rolling KPIs → discrete tactical situations
    (e.g. "dominant & efficient", "pressing under threat", "chasing game")
    Cluster labels validated with PCA + silhouette score.

  Stage B — Intervention Recommender
    For each game-state cluster, a DecisionTreeClassifier trained on
    intervention context features to predict whether the outcome of an
    intervention was positive (xG diff improved).
    A RandomForest surfaces feature importances for explainability.

  Outputs (saved to data/models/):
    game_state_kmeans.pkl     — fitted KMeans model
    game_state_scaler.pkl     — StandardScaler for game-state features
    cluster_profiles.csv      — mean KPIs per cluster
    intervention_rf.pkl       — RandomForest intervention outcome predictor
    intervention_scaler.pkl   — scaler for intervention features
    feature_importances.csv   — ranked feature importances
    modelling_report.txt      — full evaluation summary
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import (
    silhouette_score, classification_report, confusion_matrix
)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

GS_FEATURES = [
    "mar_xg_rate",
    "opp_xg_rate",
    "xg_diff",
    "mar_pass_completion_rate",
    "mar_pass_completion_delta",
    "mar_pass_completion_under_pressure",
    "mar_pressure_intensity",
    "opp_pressure_intensity",
    "mar_territory_mean_x",
    "opp_territory_mean_x",
    "territory_diff",
    "momentum_index",
    "mar_progressive_pass_rate",
    "opp_progressive_pass_rate",
    "mar_shot_rate",
    "opp_shot_rate",
    "score_diff",
    "time_remaining_proxy",
]

INTERVENTION_FEATURES = [
    "score_diff",
    "momentum_index",
    "xg_diff",
    "mar_xg_rate",
    "opp_xg_rate",
    "mar_pass_completion_rate",
    "mar_pass_completion_delta",
    "mar_pressure_intensity",
    "opp_pressure_intensity",
    "mar_territory_mean_x",
    "mar_progressive_pass_rate",
    "time_remaining_proxy",
]

CLUSTER_NAMES = {
    # Named after fitting; order may vary — we remap after profiling
    0: "Dominant_Efficient",
    1: "Pressing_Under_Threat",
    2: "Chasing_Game",
    3: "Defensive_Consolidation",
}

N_CLUSTERS = 4


# ---------------------------------------------------------------------------
# Stage A: Game-state clustering
# ---------------------------------------------------------------------------

def build_game_state_clusters(game_states: pd.DataFrame):
    print("\n--- Stage A: Game-State Clustering ---")

    X_raw = game_states[GS_FEATURES].copy()
    X_raw = X_raw.fillna(X_raw.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Silhouette sweep to validate K=4
    print("\n  Silhouette scores (K=2..6):")
    best_k, best_sil = N_CLUSTERS, -1
    for k in range(2, 7):
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels_tmp = km_tmp.fit_predict(X)
        sil = silhouette_score(X, labels_tmp)
        marker = " ← chosen" if k == N_CLUSTERS else ""
        print(f"    K={k}  silhouette={sil:.4f}{marker}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    print(f"  Best silhouette at K={best_k} (using K={N_CLUSTERS} for interpretability)")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)

    game_states = game_states.copy()
    game_states["cluster"] = labels

    # Profile clusters
    profile = game_states.groupby("cluster")[GS_FEATURES + ["cluster"]].mean().round(3)
    profile = profile.drop(columns=["cluster"], errors="ignore")

    print("\n  Cluster profiles (mean feature values):")
    print(profile.T.to_string())

    # Semantic labelling based on highest-signal features
    semantic = _label_clusters(profile)
    game_states["cluster_label"] = game_states["cluster"].map(semantic)
    print("\n  Cluster semantic labels:")
    for k, v in semantic.items():
        n = (game_states["cluster"] == k).sum()
        print(f"    Cluster {k}: {v}  (n={n})")

    # PCA for 2D visualisation coordinates (stored, not plotted)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    game_states["pca_x"] = coords[:, 0]
    game_states["pca_y"] = coords[:, 1]
    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return game_states, kmeans, scaler, profile, semantic


def _label_clusters(profile: pd.DataFrame) -> dict:
    """
    Assign semantic names based on relative feature rankings across clusters.
    Priority order: Pressing_Under_Threat (highest opp xg) → Dominant_Efficient
    (highest mar xg) → Chasing_Game (lowest score_diff) → Balanced_Phase.
    Remaining clusters get generic names.
    """
    labels = {}
    remaining = list(profile.index)

    # Cluster with highest opponent xg rate = under threat
    k_threat = profile.loc[remaining, "opp_xg_rate"].idxmax()
    labels[k_threat] = "Pressing_Under_Threat"
    remaining.remove(k_threat)

    # Cluster with highest Morocco xg rate = dominant
    k_dom = profile.loc[remaining, "mar_xg_rate"].idxmax()
    labels[k_dom] = "Dominant_Efficient"
    remaining.remove(k_dom)

    # Cluster with lowest score diff = chasing game
    k_chase = profile.loc[remaining, "score_diff"].idxmin()
    labels[k_chase] = "Chasing_Game"
    remaining.remove(k_chase)

    # Remainder
    for k in remaining:
        labels[k] = "Balanced_Phase"

    return labels


# ---------------------------------------------------------------------------
# Stage B: Intervention outcome predictor
# ---------------------------------------------------------------------------

def build_intervention_model(interventions: pd.DataFrame, game_states: pd.DataFrame):
    print("\n--- Stage B: Intervention Outcome Predictor ---")

    # Merge cluster label onto interventions
    gs_slim = game_states[["match_id", "minute", "cluster", "cluster_label"]].copy()
    iv = interventions.copy()
    iv = iv.merge(gs_slim, on=["match_id", "minute"], how="left")

    X_raw = iv[INTERVENTION_FEATURES].copy().fillna(iv[INTERVENTION_FEATURES].median())
    y     = iv["outcome_positive"].astype(int)
    groups = iv["match_id"]  # for Leave-One-Match-Out CV

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"\n  Samples: {len(iv)} | Positive outcomes: {y.sum()} ({y.mean()*100:.0f}%)")
    print(f"  Features: {INTERVENTION_FEATURES}")

    # Decision tree — interpretable baseline
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
    dt.fit(X, y)

    print("\n  Decision Tree rules (depth ≤ 3):")
    feature_names = INTERVENTION_FEATURES
    tree_text = export_text(dt, feature_names=feature_names, max_depth=3)
    print(tree_text)

    # Random Forest — better generalisation + feature importances
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=2,
        random_state=42, class_weight="balanced"
    )
    rf.fit(X, y)

    # Leave-One-Match-Out cross-validation (only 4 matches — LOMO is the right CV)
    logo = LeaveOneGroupOut()
    cv_scores = cross_val_score(rf, X, y, cv=logo, groups=groups, scoring="roc_auc")
    print(f"\n  RandomForest LOMO-CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Per-fold AUC: {[round(s,3) for s in cv_scores]}")

    # In-sample evaluation
    y_pred = rf.predict(X)
    print("\n  In-sample classification report:")
    print(classification_report(y, y_pred, target_names=["No Improvement", "Improvement"]))

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=INTERVENTION_FEATURES)
    importances = importances.sort_values(ascending=False)
    print("\n  Feature importances (RandomForest):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<45} {imp:.4f}  {bar}")

    return rf, dt, scaler, importances, iv


# ---------------------------------------------------------------------------
# Cluster-conditional recommendation logic
# ---------------------------------------------------------------------------

def build_cluster_intervention_profiles(iv_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    For each game-state cluster, summarise:
    - How often was an intervention made?
    - What fraction were positive?
    - Mean feature values at intervention time
    Used by the Streamlit app to display contextual recommendations.
    """
    cols = ["cluster_label", "intervention_type", "outcome_positive",
            "score_diff", "momentum_index", "time_remaining_proxy",
            "mar_xg_rate", "opp_xg_rate"]
    present = [c for c in cols if c in iv_with_clusters.columns]
    profile = iv_with_clusters[present].groupby(
        ["cluster_label", "intervention_type"]
    ).agg(
        count=("outcome_positive", "count"),
        positive_rate=("outcome_positive", "mean"),
        mean_minute=("time_remaining_proxy", lambda x: 90 - x.mean()),
        mean_score_diff=("score_diff", "mean"),
        mean_momentum=("momentum_index", "mean"),
    ).round(3).reset_index()
    return profile


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(kmeans, gs_scaler, rf, iv_scaler,
                cluster_profiles, importances,
                iv_with_clusters, cluster_recs,
                game_states_clustered):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "game_state_kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODEL_DIR, "game_state_scaler.pkl"), "wb") as f:
        pickle.dump(gs_scaler, f)
    with open(os.path.join(MODEL_DIR, "intervention_rf.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODEL_DIR, "intervention_scaler.pkl"), "wb") as f:
        pickle.dump(iv_scaler, f)

    cluster_profiles.to_csv(os.path.join(MODEL_DIR, "cluster_profiles.csv"))
    importances.to_frame("importance").to_csv(os.path.join(MODEL_DIR, "feature_importances.csv"))
    iv_with_clusters.to_csv(os.path.join(DATA_DIR, "interventions_clustered.csv"), index=False)
    cluster_recs.to_csv(os.path.join(MODEL_DIR, "cluster_intervention_profiles.csv"), index=False)
    game_states_clustered.to_csv(os.path.join(DATA_DIR, "game_states_clustered.csv"), index=False)

    print(f"\n  Models saved to  : {MODEL_DIR}")
    print(f"  Data   saved to  : {DATA_DIR}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(game_states_clustered, iv_with_clusters, importances,
                 cluster_profiles, cluster_recs, semantic):
    report_path = os.path.join(DATA_DIR, "models", "modelling_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("CRISP-DM PHASE 4 — MODELLING REPORT")
    lines.append("Prescriptive DSS for Football Coaches | Morocco AFCON 2023")
    lines.append("=" * 70)

    lines.append("\n[A] GAME-STATE CLUSTERS")
    lines.append(f"   K = {N_CLUSTERS}  |  Features: {len(GS_FEATURES)}")
    for k, label in semantic.items():
        n = (game_states_clustered["cluster"] == k).sum()
        lines.append(f"   Cluster {k}: {label}  (n_minutes={n})")

    lines.append("\n[B] CLUSTER PROFILES")
    lines.append(cluster_profiles.T.to_string())

    lines.append("\n[C] INTERVENTION OUTCOME PREDICTOR")
    lines.append(f"   Target: outcome_positive (xG diff improved in 15-min post-window)")
    lines.append(f"   Algorithm: RandomForestClassifier (n=200, balanced)")
    lines.append(f"   CV: Leave-One-Match-Out (LOMO)")
    lines.append(f"   Samples: {len(iv_with_clusters)}")

    lines.append("\n[D] TOP PREDICTIVE FEATURES")
    for feat, imp in importances.head(8).items():
        lines.append(f"   {feat:<45} {imp:.4f}")

    lines.append("\n[E] CLUSTER-CONDITIONAL INTERVENTION PROFILES")
    lines.append(cluster_recs.to_string(index=False))

    lines.append("\n[F] MODELLING NOTES & LIMITATIONS")
    lines.append("""
   - Only 4 matches (n=21 interventions) → severe sample constraint.
     Models are directionally valid but should be retrained on the full
     StatsBomb AFCON 2023 dataset (all 52 matches) before production use.
   - LOMO-CV is the correct evaluation strategy (avoids data leakage across
     matches); AUC is an approximation given the tiny fold sizes.
   - The RandomForest is used for feature importance; the DecisionTree
     provides human-readable rules for the coach interface.
   - Game-state clustering is unsupervised — semantic labels are assigned
     post-hoc from relative feature rankings, not from football ontology.
   - Next step: expand training corpus to all AFCON teams + other StatsBomb
     competitions for robust prescriptive recommendations.
""")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_modelling():
    print("=" * 70)
    print("CRISP-DM — MODELLING")
    print("=" * 70)

    print("\n[0] Loading prepared data …")
    game_states   = pd.read_csv(os.path.join(DATA_DIR, "game_states.csv"))
    interventions = pd.read_csv(os.path.join(DATA_DIR, "interventions.csv"))
    print(f"    game_states   : {len(game_states)} rows")
    print(f"    interventions : {len(interventions)} rows")

    # Stage A
    game_states_cl, kmeans, gs_scaler, cluster_profiles, semantic = \
        build_game_state_clusters(game_states)

    # Stage B
    rf, dt, iv_scaler, importances, iv_clustered = \
        build_intervention_model(interventions, game_states_cl)

    # Cluster-level recommendation profiles
    cluster_recs = build_cluster_intervention_profiles(iv_clustered)
    print("\n  Cluster × intervention recommendation profiles:")
    print(cluster_recs.to_string(index=False))

    # Save everything
    print("\n[Saving models & artefacts]")
    save_models(
        kmeans, gs_scaler, rf, iv_scaler,
        cluster_profiles, importances,
        iv_clustered, cluster_recs,
        game_states_cl
    )

    report = write_report(
        game_states_cl, iv_clustered, importances,
        cluster_profiles, cluster_recs, semantic
    )

    print("\n" + "=" * 70)
    print("MODELLING COMPLETE")
    print("=" * 70)
    print(report)

    return {
        "game_states":     game_states_cl,
        "kmeans":          kmeans,
        "gs_scaler":       gs_scaler,
        "rf":              rf,
        "dt":              dt,
        "iv_scaler":       iv_scaler,
        "importances":     importances,
        "interventions":   iv_clustered,
        "cluster_recs":    cluster_recs,
        "cluster_profiles": cluster_profiles,
        "semantic":        semantic,
    }


if __name__ == "__main__":
    artefacts = run_modelling()
