"""
CRISP-DM Phase 3: Data Preparation
Prescriptive DSS for Football Coaches — Morocco AFCON 2023

Outputs:
  data/game_states.csv          — minute-by-minute game-state vectors (all 4 matches)
  data/interventions.csv        — labelled substitution / formation-shift events
  data/player_snapshots.csv     — per-player rolling KPIs at each intervention minute

Features engineered (no hardcoded rules — all derived from event distributions):
  Offensive:  rolling_xg, rolling_shots, rolling_key_passes, rolling_progressive_passes,
              xg_rate, shot_conversion_rate, territory_mean_x
  Defensive:  opp_rolling_xg, opp_rolling_shots, opp_xg_rate, pressure_intensity
  Momentum:   xg_diff, territory_diff, momentum_index (composite)
  Game state: score_diff, minute, time_remaining_proxy
  Fatigue:    pass_completion_rolling, pass_completion_delta (vs first-30 baseline)
  Pressure:   pct_events_under_pressure, pass_completion_under_pressure
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd
import numpy as np
from statsbombpy import sb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPETITION_ID = 1267
SEASON_ID      = 107
TEAM           = "Morocco"
ROLL_WINDOW    = 15   # minutes for rolling aggregations
OUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")

MATCH_META = {
    3920394: {"home": "Morocco", "away": "Tanzania",     "home_score": 3, "away_score": 0, "stage": "Group Stage"},
    3920405: {"home": "Morocco", "away": "Congo DR",     "home_score": 1, "away_score": 1, "stage": "Group Stage"},
    3920419: {"home": "Zambia",  "away": "Morocco",      "home_score": 0, "away_score": 1, "stage": "Group Stage"},
    3922243: {"home": "Morocco", "away": "South Africa", "home_score": 0, "away_score": 2, "stage": "Round of 16"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _x(loc):
    return float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else np.nan

def _y(loc):
    return float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else np.nan

def _to_seconds(minute, second):
    return int(minute) * 60 + int(second if pd.notna(second) else 0)


def _load_events(match_ids):
    frames = [sb.events(match_id=mid) for mid in match_ids]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 1: Enrich raw events with derived atomic columns
# ---------------------------------------------------------------------------

def enrich_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["loc_x"]         = ev["location"].apply(_x)
    ev["loc_y"]         = ev["location"].apply(_y)
    ev["pass_end_x"]    = ev["pass_end_location"].apply(_x)
    ev["pass_end_y"]    = ev["pass_end_location"].apply(_y)

    ev["is_shot"]              = (ev["type"] == "Shot").astype(int)
    ev["is_goal"]              = ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")).astype(int)
    ev["is_pass"]              = (ev["type"] == "Pass").astype(int)
    ev["pass_completed"]       = ((ev["type"] == "Pass") & ev["pass_outcome"].isna()).astype(int)
    ev["pass_under_pressure"]  = ((ev["type"] == "Pass") & (ev["under_pressure"] == True)).astype(int)
    ev["pass_completed_under"] = ((ev["type"] == "Pass") & (ev["under_pressure"] == True) & ev["pass_outcome"].isna()).astype(int)
    ev["is_pressure"]          = (ev["type"] == "Pressure").astype(int)
    ev["is_progressive_pass"]  = (
        (ev["type"] == "Pass") &
        ((ev["pass_end_x"] - ev["loc_x"]) > 10)
    ).astype(int)
    ev["is_key_pass"]          = ((ev["type"] == "Pass") & (ev["pass_shot_assist"] == True)).astype(int)
    ev["is_dribble_complete"]  = ((ev["type"] == "Dribble") & (ev["dribble_outcome"] == "Complete")).astype(int)
    ev["is_dribble"]           = (ev["type"] == "Dribble").astype(int)
    ev["is_carry"]             = (ev["type"] == "Carry").astype(int)
    ev["is_under_pressure"]    = (ev["under_pressure"] == True).astype(int)
    ev["xg"]                   = ev["shot_statsbomb_xg"].fillna(0.0)

    ev["game_second"] = ev.apply(lambda r: _to_seconds(r["minute"], r["second"]), axis=1)
    return ev


# ---------------------------------------------------------------------------
# Step 2: Rolling game-state vector (per team, per match, per minute)
# ---------------------------------------------------------------------------

def _rolling_team_stats(ev_team: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Build a minute-resolution timeseries with rolling features for one team
    in one match. The roll covers the last `window` minutes.
    """
    max_min = int(ev_team["minute"].max()) + 1
    minutes = np.arange(0, max_min + 1)

    # Aggregate raw stats per minute
    per_min = ev_team.groupby("minute").agg(
        xg=("xg", "sum"),
        shots=("is_shot", "sum"),
        goals=("is_goal", "sum"),
        passes=("is_pass", "sum"),
        passes_completed=("pass_completed", "sum"),
        passes_under_pressure=("pass_under_pressure", "sum"),
        passes_completed_under=("pass_completed_under", "sum"),
        progressive_passes=("is_progressive_pass", "sum"),
        key_passes=("is_key_pass", "sum"),
        pressures=("is_pressure", "sum"),
        dribbles=("is_dribble", "sum"),
        dribbles_complete=("is_dribble_complete", "sum"),
        under_pressure_events=("is_under_pressure", "sum"),
        territory_x=("loc_x", "mean"),
    ).reindex(minutes, fill_value=0)

    per_min["territory_x"] = per_min["territory_x"].replace(0, np.nan).ffill()

    # Rolling sums / means
    roll = per_min.rolling(window=window, min_periods=1)

    gs = pd.DataFrame(index=minutes)
    gs["rolling_xg"]                = roll["xg"].sum()
    gs["rolling_shots"]             = roll["shots"].sum()
    gs["rolling_goals"]             = roll["goals"].sum()
    gs["rolling_passes"]            = roll["passes"].sum()
    gs["rolling_passes_completed"]  = roll["passes_completed"].sum()
    gs["rolling_progressive"]       = roll["progressive_passes"].sum()
    gs["rolling_key_passes"]        = roll["key_passes"].sum()
    gs["rolling_pressures"]         = roll["pressures"].sum()
    gs["rolling_under_pressure"]    = roll["under_pressure_events"].sum()
    gs["rolling_dribbles"]          = roll["dribbles"].sum()
    gs["rolling_dribbles_comp"]     = roll["dribbles_complete"].sum()

    # Derived rates (avoid /0)
    gs["pass_completion_rate"] = np.where(
        gs["rolling_passes"] > 0,
        gs["rolling_passes_completed"] / gs["rolling_passes"],
        np.nan
    )
    gs["pass_completion_under_pressure"] = np.where(
        roll["passes_under_pressure"].sum() > 0,
        roll["passes_completed_under"].sum() / roll["passes_under_pressure"].sum(),
        np.nan
    )
    gs["xg_rate"]               = gs["rolling_xg"] / window
    gs["shot_rate"]             = gs["rolling_shots"] / window
    gs["pressure_intensity"]    = gs["rolling_pressures"] / window
    gs["progressive_pass_rate"] = np.where(
        gs["rolling_passes"] > 0,
        gs["rolling_progressive"] / gs["rolling_passes"],
        np.nan
    )
    gs["dribble_success_rate"]  = np.where(
        gs["rolling_dribbles"] > 0,
        gs["rolling_dribbles_comp"] / gs["rolling_dribbles"],
        np.nan
    )
    gs["territory_mean_x"]      = per_min["territory_x"].rolling(window, min_periods=1).mean()

    # Cumulative xG (for score approximation cross-check)
    gs["cumulative_xg"]         = per_min["xg"].cumsum()
    gs["cumulative_shots"]      = per_min["shots"].cumsum()

    # Fatigue: pass completion delta vs first-30-minute baseline
    baseline = gs.loc[gs.index <= 30, "pass_completion_rate"].mean()
    gs["pass_completion_delta"] = gs["pass_completion_rate"] - baseline

    gs["minute"] = minutes
    return gs


def build_game_states(events: pd.DataFrame, match_ids: list) -> pd.DataFrame:
    """
    For every minute in every match, produce a combined game-state vector
    with both Morocco's and the opponent's rolling KPIs.
    """
    all_rows = []

    for mid in match_ids:
        meta    = MATCH_META[mid]
        opponent = meta["away"] if meta["home"] == TEAM else meta["home"]
        ev_match = events[events["match_id"] == mid]

        ev_mar  = ev_match[ev_match["team"] == TEAM]
        ev_opp  = ev_match[ev_match["team"] == opponent]

        mar_gs  = _rolling_team_stats(ev_mar, ROLL_WINDOW).add_prefix("mar_")
        opp_gs  = _rolling_team_stats(ev_opp, ROLL_WINDOW).add_prefix("opp_")

        gs = pd.concat([mar_gs, opp_gs], axis=1)
        gs.rename(columns={"mar_minute": "minute"}, inplace=True)
        gs.drop(columns=["opp_minute"], inplace=True, errors="ignore")

        # Momentum: composite = xg_rate + progressive_pass_rate - opp_xg_rate
        gs["xg_diff"]        = gs["mar_rolling_xg"]  - gs["opp_rolling_xg"]
        gs["territory_diff"] = gs["mar_territory_mean_x"] - gs["opp_territory_mean_x"]
        gs["momentum_index"] = (
            gs["mar_xg_rate"].fillna(0) * 2
            + gs["mar_progressive_pass_rate"].fillna(0)
            + gs["mar_pressure_intensity"].fillna(0) * 0.5
            - gs["opp_xg_rate"].fillna(0) * 2
            - gs["opp_pressure_intensity"].fillna(0) * 0.5
        )

        # Score reconstruction from goals
        mar_goals_cumul = ev_mar[ev_mar["is_goal"] == 1].groupby("minute")["is_goal"].sum().cumsum()
        opp_goals_cumul = ev_opp[ev_opp["is_goal"] == 1].groupby("minute")["is_goal"].sum().cumsum()

        max_min = int(ev_match["minute"].max()) + 1
        minutes = np.arange(0, max_min + 1)
        mar_score = mar_goals_cumul.reindex(minutes, method="ffill").fillna(0).astype(int)
        opp_score = opp_goals_cumul.reindex(minutes, method="ffill").fillna(0).astype(int)

        gs["mar_score"]           = mar_score.values
        gs["opp_score"]           = opp_score.values
        gs["score_diff"]          = gs["mar_score"] - gs["opp_score"]
        gs["time_remaining_proxy"]= np.maximum(0, 90 - gs["minute"])

        gs["match_id"]   = mid
        gs["opponent"]   = opponent
        gs["stage"]      = meta.get("stage", "")
        all_rows.append(gs)

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3: Intervention labels + outcome signal
# ---------------------------------------------------------------------------

def _extract_formations(events: pd.DataFrame, team: str, match_id: int) -> dict[int, int]:
    """Return {minute: formation} for all formation events for team."""
    ev = events[
        (events["match_id"] == match_id) &
        (events["team"] == team) &
        (events["type"].isin(["Starting XI", "Tactical Shift"]))
    ]
    result = {}
    for _, row in ev.iterrows():
        if isinstance(row["tactics"], dict):
            result[int(row["minute"])] = row["tactics"].get("formation")
    return result


def build_interventions(events: pd.DataFrame,
                         game_states: pd.DataFrame,
                         match_ids: list) -> pd.DataFrame:
    """
    Label each substitution and tactical shift with:
    - Context features (game-state vector at that minute)
    - Outcome: xG differential change in the 15 min window after vs before
    - Score context at the time of the intervention
    """
    rows = []

    for mid in match_ids:
        meta     = MATCH_META[mid]
        opponent = meta["away"] if meta["home"] == TEAM else meta["home"]
        ev_match = events[events["match_id"] == mid]
        gs_match = game_states[game_states["match_id"] == mid].set_index("minute")

        formations = _extract_formations(events, TEAM, mid)
        current_formation = formations.get(0, None)

        # --- Substitutions ---
        subs = ev_match[
            (ev_match["team"] == TEAM) & (ev_match["type"] == "Substitution")
        ]
        for _, sub in subs.iterrows():
            minute = int(sub["minute"])
            row = _intervention_row(
                mid, opponent, minute, "substitution", gs_match, current_formation,
                extra={
                    "player_off":  sub["player"],
                    "player_on":   sub["substitution_replacement"],
                    "sub_outcome": sub["substitution_outcome"],
                }
            )
            rows.append(row)

        # --- Tactical shifts ---
        shifts = ev_match[
            (ev_match["team"] == TEAM) & (ev_match["type"] == "Tactical Shift")
        ]
        for _, shift in shifts.iterrows():
            minute = int(shift["minute"])
            new_formation = shift["tactics"].get("formation") if isinstance(shift["tactics"], dict) else None
            row = _intervention_row(
                mid, opponent, minute, "formation_shift", gs_match, current_formation,
                extra={
                    "player_off":     "",
                    "player_on":      "",
                    "sub_outcome":    "",
                    "new_formation":  new_formation,
                }
            )
            rows.append(row)
            current_formation = new_formation

    return pd.DataFrame(rows)


def _intervention_row(match_id, opponent, minute, intervention_type,
                       gs_match, formation, extra: dict) -> dict:
    """
    Pull context (pre-window) and outcome (post-window) xG diff from game_states.
    """
    W = ROLL_WINDOW

    # Context window: state at the intervention minute
    ctx = gs_match.loc[minute] if minute in gs_match.index else pd.Series(dtype=float)

    # Pre-window xG totals (last W minutes before intervention)
    pre_start = max(0, minute - W)
    pre = gs_match.loc[pre_start:minute - 1] if minute > 0 else gs_match.iloc[0:0]
    pre_mar_xg = pre["mar_rolling_xg"].mean() if len(pre) else 0.0
    pre_opp_xg = pre["opp_rolling_xg"].mean() if len(pre) else 0.0

    # Post-window xG totals (W minutes after intervention)
    post_end = min(gs_match.index.max(), minute + W)
    post = gs_match.loc[minute + 1:post_end] if minute + 1 <= gs_match.index.max() else gs_match.iloc[0:0]
    post_mar_xg = post["mar_rolling_xg"].mean() if len(post) else 0.0
    post_opp_xg = post["opp_rolling_xg"].mean() if len(post) else 0.0

    outcome_xg_diff_change = (post_mar_xg - post_opp_xg) - (pre_mar_xg - pre_opp_xg)

    row = {
        "match_id":                match_id,
        "opponent":                opponent,
        "minute":                  minute,
        "intervention_type":       intervention_type,
        "formation_at_time":       formation,
        # Context features
        "score_diff":              ctx.get("score_diff", np.nan),
        "momentum_index":          ctx.get("momentum_index", np.nan),
        "xg_diff":                 ctx.get("xg_diff", np.nan),
        "mar_xg_rate":             ctx.get("mar_xg_rate", np.nan),
        "opp_xg_rate":             ctx.get("opp_xg_rate", np.nan),
        "mar_pass_completion_rate":ctx.get("mar_pass_completion_rate", np.nan),
        "mar_pass_completion_delta":ctx.get("mar_pass_completion_delta", np.nan),
        "mar_pressure_intensity":  ctx.get("mar_pressure_intensity", np.nan),
        "opp_pressure_intensity":  ctx.get("opp_pressure_intensity", np.nan),
        "mar_territory_mean_x":    ctx.get("mar_territory_mean_x", np.nan),
        "mar_progressive_pass_rate":ctx.get("mar_progressive_pass_rate", np.nan),
        "time_remaining_proxy":    ctx.get("time_remaining_proxy", np.nan),
        # Outcome (reward signal)
        "pre_xg_diff":             pre_mar_xg - pre_opp_xg,
        "post_xg_diff":            post_mar_xg - post_opp_xg,
        "outcome_xg_diff_change":  outcome_xg_diff_change,
        "outcome_positive":        int(outcome_xg_diff_change > 0),
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Step 4: Player snapshots at intervention minute
# ---------------------------------------------------------------------------

def build_player_snapshots(events: pd.DataFrame,
                            interventions: pd.DataFrame,
                            match_ids: list) -> pd.DataFrame:
    """
    For each intervention minute, compute per-player rolling KPIs.
    Useful for data-driven player-selection recommendation.
    """
    rows = []
    W = ROLL_WINDOW

    for mid in match_ids:
        ev_match = events[(events["match_id"] == mid) & (events["team"] == TEAM)]
        int_minutes = interventions[interventions["match_id"] == mid]["minute"].tolist()

        for minute in int_minutes:
            window_ev = ev_match[
                (ev_match["minute"] >= max(0, minute - W)) &
                (ev_match["minute"] < minute)
            ]
            for player, pev in window_ev.groupby("player"):
                n_passes  = pev["is_pass"].sum()
                n_press   = pev["pass_under_pressure"].sum()
                rows.append({
                    "match_id":         mid,
                    "intervention_min": minute,
                    "player":           player,
                    "shots":            pev["is_shot"].sum(),
                    "xg":               pev["xg"].sum().round(3),
                    "passes":           n_passes,
                    "pass_completion":  round(pev["pass_completed"].sum() / n_passes * 100, 1) if n_passes else np.nan,
                    "passes_under_pressure": n_press,
                    "pass_compl_under": round(pev["pass_completed_under"].sum() / n_press * 100, 1) if n_press else np.nan,
                    "progressive_passes": pev["is_progressive_pass"].sum(),
                    "key_passes":       pev["is_key_pass"].sum(),
                    "pressures":        pev["is_pressure"].sum(),
                    "dribbles_att":     pev["is_dribble"].sum(),
                    "dribbles_comp":    pev["is_dribble_complete"].sum(),
                    "territory_x":      round(pev["loc_x"].mean(), 1),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: Quality checks
# ---------------------------------------------------------------------------

def quality_report(game_states: pd.DataFrame,
                   interventions: pd.DataFrame,
                   player_snapshots: pd.DataFrame):
    print("\n[QUALITY REPORT]")

    print(f"\n  game_states      : {len(game_states):>5} rows × {len(game_states.columns)} cols")
    null_pct = (game_states.isnull().mean() * 100).round(1)
    high_null = null_pct[null_pct > 10]
    if len(high_null):
        print("  Columns with >10% nulls (game_states):")
        for c, v in high_null.items():
            print(f"    {c:<45} {v:.1f}%")
    else:
        print("  No columns with >10% nulls in game_states.")

    print(f"\n  interventions    : {len(interventions):>5} rows × {len(interventions.columns)} cols")
    print(f"  player_snapshots : {len(player_snapshots):>5} rows × {len(player_snapshots.columns)} cols")

    print("\n  Intervention breakdown:")
    print(interventions["intervention_type"].value_counts().to_string())

    print("\n  Intervention outcomes (positive = improved xG diff after):")
    print(interventions.groupby("intervention_type")["outcome_positive"]
          .agg(["count", "sum", "mean"]).round(2).to_string())

    print("\n  Score context at intervention time:")
    print(interventions.groupby("score_diff")["intervention_type"].value_counts().to_string())

    print("\n  Momentum at intervention time (mean per type):")
    print(interventions.groupby("intervention_type")[
        ["momentum_index","mar_xg_rate","opp_xg_rate",
         "mar_pass_completion_delta","time_remaining_proxy"]
    ].mean().round(3).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_data_preparation():
    print("=" * 70)
    print("CRISP-DM — DATA PREPARATION")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    match_ids = list(MATCH_META.keys())

    print(f"\n[1] Loading & enriching events for {len(match_ids)} matches …")
    raw_events = _load_events(match_ids)
    events = enrich_events(raw_events)
    print(f"    Events loaded : {len(events):,}")

    print(f"\n[2] Building game-state vectors (rolling window = {ROLL_WINDOW} min) …")
    game_states = build_game_states(events, match_ids)
    print(f"    Game-state rows : {len(game_states):,}")
    print("    Feature columns :", [c for c in game_states.columns if c not in ("match_id","opponent","stage","minute")])

    print("\n[3] Labelling interventions with pre/post xG outcome …")
    interventions = build_interventions(events, game_states, match_ids)
    print(f"    Intervention rows : {len(interventions)}")

    print("\n[4] Building player snapshots at intervention minutes …")
    player_snapshots = build_player_snapshots(events, interventions, match_ids)
    print(f"    Player snapshot rows : {len(player_snapshots)}")

    print("\n[5] Quality checks …")
    quality_report(game_states, interventions, player_snapshots)

    print("\n[6] Saving datasets …")
    gs_path  = os.path.join(OUT_DIR, "game_states.csv")
    int_path = os.path.join(OUT_DIR, "interventions.csv")
    ps_path  = os.path.join(OUT_DIR, "player_snapshots.csv")

    game_states.to_csv(gs_path,  index=False)
    interventions.to_csv(int_path, index=False)
    player_snapshots.to_csv(ps_path, index=False)

    print(f"    Saved: {gs_path}")
    print(f"    Saved: {int_path}")
    print(f"    Saved: {ps_path}")

    print("\n[7] Feature preview (first game-state row with all features):")
    sample = game_states[game_states["match_id"] == 3920394].iloc[15]
    for k, v in sample.items():
        if isinstance(v, float):
            print(f"    {k:<45} {v:.4f}")
        else:
            print(f"    {k:<45} {v}")

    print("\n[8] Interventions detail:")
    cols = ["match_id","opponent","minute","intervention_type",
            "score_diff","momentum_index","mar_xg_rate","opp_xg_rate",
            "time_remaining_proxy","outcome_xg_diff_change","outcome_positive",
            "player_off","player_on"]
    print(interventions[[c for c in cols if c in interventions.columns]].to_string(index=False))

    return game_states, interventions, player_snapshots


if __name__ == "__main__":
    gs, iv, ps = run_data_preparation()
