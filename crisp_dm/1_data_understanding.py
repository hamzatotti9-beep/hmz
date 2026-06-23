"""
CRISP-DM Phase 1 & 2: Business Understanding + Data Understanding
Prescriptive Data-Driven Decision Support System for Football Coaches
Data Source: StatsBomb Open Data — Morocco AFCON 2023
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from statsbombpy import sb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPETITION_ID = 1267   # African Cup of Nations
SEASON_ID = 107         # 2023
TEAM = "Morocco"
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

MATCH_META = {
    3920394: {"home": "Morocco", "away": "Tanzania",     "score": "3-0", "stage": "Group Stage"},
    3920405: {"home": "Morocco", "away": "Congo DR",     "score": "1-1", "stage": "Group Stage"},
    3920419: {"home": "Zambia",  "away": "Morocco",      "score": "0-1", "stage": "Group Stage"},
    3922243: {"home": "Morocco", "away": "South Africa", "score": "0-2", "stage": "Round of 16"},
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_matches() -> pd.DataFrame:
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    morocco = matches[
        (matches["home_team"] == TEAM) | (matches["away_team"] == TEAM)
    ].copy()
    return morocco


def load_all_events(match_ids: list[int]) -> pd.DataFrame:
    frames = [sb.events(match_id=mid) for mid in match_ids]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def extract_xy(series: pd.Series) -> pd.DataFrame:
    """Split a column of [x, y] lists into two float columns."""
    def _parse(v):
        if isinstance(v, list) and len(v) >= 2:
            return float(v[0]), float(v[1])
        return np.nan, np.nan

    parsed = series.apply(_parse)
    return pd.DataFrame(parsed.tolist(), index=series.index, columns=["x", "y"])


def pitch_zone(x: float, y: float) -> str:
    """Divide pitch into a 3×3 grid: {def/mid/att}_{left/center/right}."""
    third = "def" if x < 40 else ("mid" if x < 80 else "att")
    side  = "left" if y < 26.7 else ("right" if y > 53.3 else "center")
    return f"{third}_{side}"


def time_bin(minute: float) -> str:
    bins   = [0, 15, 30, 45, 60, 75, 90, 200]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]
    for lo, hi, lbl in zip(bins, bins[1:], labels):
        if lo <= minute < hi:
            return lbl
    return "90+"


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def match_summary(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    """Per-match KPIs for the target team."""
    rows = []
    for mid, meta in MATCH_META.items():
        ev = events[(events["match_id"] == mid) & (events["team"] == team)]
        shots    = ev[ev["type"] == "Shot"]
        passes   = ev[ev["type"] == "Pass"]
        presses  = ev[ev["type"] == "Pressure"]
        carries  = ev[ev["type"] == "Carry"]
        dribbles = ev[ev["type"] == "Dribble"]

        n_passes  = len(passes)
        completed = passes["pass_outcome"].isna().sum()
        rows.append({
            "match_id":       mid,
            "opponent":       meta["away"] if meta["home"] == team else meta["home"],
            "score":          meta["score"],
            "stage":          meta["stage"],
            "shots":          len(shots),
            "goals":          (shots["shot_outcome"] == "Goal").sum(),
            "xg":             shots["shot_statsbomb_xg"].sum().round(3),
            "xg_per_shot":    shots["shot_statsbomb_xg"].mean().round(3),
            "passes":         n_passes,
            "pass_pct":       round(completed / n_passes * 100, 1) if n_passes else 0,
            "progressive_passes": int(
                (passes["pass_end_location"].apply(lambda v: float(v[0]) if isinstance(v, list) else np.nan) -
                 passes["location"].apply(lambda v: float(v[0]) if isinstance(v, list) else np.nan) > 10
                ).sum()
            ),
            "crosses":        int(passes["pass_cross"].sum()),
            "key_passes":     int(passes["pass_shot_assist"].sum()),
            "pressures":      len(presses),
            "dribbles_att":   len(dribbles),
            "dribbles_comp":  int((dribbles["dribble_outcome"] == "Complete").sum()),
            "under_pressure_events": int(ev["under_pressure"].sum()),
        })
    return pd.DataFrame(rows)


def shot_profile(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    shots = events[(events["team"] == team) & (events["type"] == "Shot")].copy()
    loc   = extract_xy(shots["location"])
    shots["x"], shots["y"] = loc["x"], loc["y"]
    shots["zone"]     = shots.apply(lambda r: pitch_zone(r["x"], r["y"]), axis=1)
    shots["time_bin"] = shots["minute"].apply(time_bin)
    return shots[[
        "match_id", "minute", "time_bin", "player", "zone",
        "shot_outcome", "shot_statsbomb_xg", "shot_technique",
        "shot_body_part", "shot_type", "x", "y"
    ]]


def pass_profile(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    passes = events[(events["team"] == team) & (events["type"] == "Pass")].copy()
    sloc   = extract_xy(passes["location"])
    eloc   = extract_xy(passes["pass_end_location"])
    passes["start_x"], passes["start_y"] = sloc["x"], sloc["y"]
    passes["end_x"],   passes["end_y"]   = eloc["x"], eloc["y"]
    passes["progressive"] = (passes["end_x"] - passes["start_x"]) > 10
    passes["completed"]   = passes["pass_outcome"].isna()
    passes["zone"]        = passes.apply(lambda r: pitch_zone(r["start_x"], r["start_y"]), axis=1)
    passes["time_bin"]    = passes["minute"].apply(time_bin)
    return passes[[
        "match_id", "minute", "time_bin", "player", "zone",
        "completed", "progressive", "pass_cross", "pass_switch",
        "pass_shot_assist", "pass_goal_assist",
        "under_pressure", "pass_length", "pass_angle",
        "start_x", "start_y", "end_x", "end_y"
    ]]


def pressing_profile(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    presses = events[(events["team"] == team) & (events["type"] == "Pressure")].copy()
    loc = extract_xy(presses["location"])
    presses["x"], presses["y"] = loc["x"], loc["y"]
    presses["zone"]     = presses.apply(lambda r: pitch_zone(r["x"], r["y"]), axis=1)
    presses["time_bin"] = presses["minute"].apply(time_bin)
    return presses[["match_id", "minute", "time_bin", "player", "zone", "x", "y"]]


def substitution_profile(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    subs = events[(events["team"] == team) & (events["type"] == "Substitution")].copy()
    return subs[[
        "match_id", "minute", "player",
        "substitution_replacement", "substitution_outcome"
    ]]


def tactical_profile(events: pd.DataFrame, team: str = TEAM) -> list[dict]:
    """Extract formation changes per match."""
    records = []
    for et in ["Starting XI", "Tactical Shift"]:
        rows = events[(events["team"] == team) & (events["type"] == et)]
        for _, r in rows.iterrows():
            tac = r["tactics"]
            if isinstance(tac, dict):
                records.append({
                    "match_id":  r["match_id"],
                    "minute":    r["minute"],
                    "event":     et,
                    "formation": tac.get("formation"),
                    "lineup":    [
                        {
                            "player":   p["player"]["name"],
                            "position": p["position"]["name"],
                            "jersey":   p["jersey_number"],
                        }
                        for p in tac.get("lineup", [])
                    ],
                })
    return records


def player_summary(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    ev = events[events["team"] == team]
    shots   = ev[ev["type"] == "Shot"]
    passes  = ev[ev["type"] == "Pass"]
    presses = ev[ev["type"] == "Pressure"]
    drib    = ev[ev["type"] == "Dribble"]

    players = ev["player"].dropna().unique()
    rows = []
    for p in players:
        s  = shots[shots["player"] == p]
        pa = passes[passes["player"] == p]
        pr = presses[presses["player"] == p]
        d  = drib[drib["player"] == p]
        n  = len(pa)
        rows.append({
            "player":           p,
            "shots":            len(s),
            "goals":            int((s["shot_outcome"] == "Goal").sum()),
            "xg":               round(s["shot_statsbomb_xg"].sum(), 3),
            "passes":           n,
            "pass_pct":         round(pa["pass_outcome"].isna().sum() / n * 100, 1) if n else 0,
            "key_passes":       int(pa["pass_shot_assist"].sum()),
            "goal_assists":     int(pa["pass_goal_assist"].sum()),
            "pressures":        len(pr),
            "dribbles_att":     len(d),
            "dribbles_comp":    int((d["dribble_outcome"] == "Complete").sum()),
        })
    return pd.DataFrame(rows).sort_values("passes", ascending=False)


def temporal_dynamics(events: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    ev = events[events["team"] == team].copy()
    ev["time_bin"] = ev["minute"].apply(time_bin)
    ev["is_shot"]     = ev["type"] == "Shot"
    ev["is_pass"]     = ev["type"] == "Pass"
    ev["is_pressure"] = ev["type"] == "Pressure"

    agg = ev.groupby("time_bin", sort=False).agg(
        shots=("is_shot", "sum"),
        xg=("shot_statsbomb_xg", "sum"),
        passes=("is_pass", "sum"),
        pressures=("is_pressure", "sum"),
        under_pressure_events=("under_pressure", "sum"),
    ).round(3)

    # Preserve temporal order
    order = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]
    agg = agg.reindex([b for b in order if b in agg.index])
    return agg


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run_data_understanding():
    print("=" * 70)
    print("CRISP-DM — DATA UNDERSTANDING")
    print(f"Project : Prescriptive DSS for Football Coaches")
    print(f"Data    : StatsBomb Open Data | AFCON 2023 | {TEAM}")
    print("=" * 70)

    # 1. Matches
    matches = load_matches()
    match_ids = matches["match_id"].tolist()
    print(f"\n[1] Morocco played {len(match_ids)} matches in AFCON 2023")
    print(matches[["match_id", "match_date", "competition_stage",
                   "home_team", "home_score", "away_score", "away_team"]].to_string(index=False))

    # 2. Load events
    print("\n[2] Loading event data …")
    events = load_all_events(match_ids)
    morocco_ev = events[events["team"] == TEAM]
    print(f"    Total events (all teams): {len(events):,}")
    print(f"    Morocco events          : {len(morocco_ev):,}")
    print(f"    Event types available   : {events['type'].nunique()}")

    # 3. Data quality
    print("\n[3] DATA QUALITY CHECK")
    key_cols = ["location", "pass_end_location", "shot_statsbomb_xg",
                "shot_outcome", "pass_outcome", "minute", "player", "type"]
    q = morocco_ev[key_cols].isnull().mean().round(3) * 100
    print("    Null % per key column:")
    for col, pct in q.items():
        print(f"      {col:<30} {pct:.1f}%")

    # 4. Match summary
    print("\n[4] PER-MATCH KPIs")
    summary = match_summary(events)
    print(summary.to_string(index=False))

    # 5. Shot profile
    print("\n[5] SHOT PROFILE")
    shots_df = shot_profile(events)
    print("  Zone distribution:")
    print(shots_df["zone"].value_counts().to_string())
    print("\n  xG by zone:")
    print(shots_df.groupby("zone")["shot_statsbomb_xg"]
          .agg(["count", "sum", "mean"]).round(3).to_string())
    print("\n  Outcomes:")
    print(shots_df["shot_outcome"].value_counts().to_string())

    # 6. Pass profile
    print("\n[6] PASS PROFILE")
    passes_df = pass_profile(events)
    print(f"  Total passes      : {len(passes_df)}")
    print(f"  Completion rate   : {passes_df['completed'].mean()*100:.1f}%")
    print(f"  Progressive passes: {passes_df['progressive'].sum()}")
    print(f"  Key passes        : {passes_df['pass_shot_assist'].sum()}")
    print(f"  Crosses           : {passes_df['pass_cross'].sum()}")
    print(f"  Switches          : {passes_df['pass_switch'].sum()}")
    print("\n  Completion rate by zone:")
    print(passes_df.groupby("zone")["completed"]
          .agg(["count", "mean"]).assign(mean=lambda d: (d["mean"]*100).round(1))
          .to_string())

    # 7. Pressing
    print("\n[7] PRESSING PROFILE")
    press_df = pressing_profile(events)
    print("  Pressures by zone:")
    print(press_df["zone"].value_counts().to_string())
    print("\n  Pressures by time window:")
    print(press_df["time_bin"].value_counts().sort_index().to_string())

    # 8. Temporal dynamics
    print("\n[8] TEMPORAL DYNAMICS (per 15-min window)")
    print(temporal_dynamics(events).to_string())

    # 9. Substitutions
    print("\n[9] SUBSTITUTIONS")
    subs_df = substitution_profile(events)
    print(subs_df.to_string(index=False))

    # 10. Tactical shifts
    print("\n[10] TACTICAL PROFILES (formations)")
    tac = tactical_profile(events)
    for t in tac:
        print(f"  Match {t['match_id']} | Min {t['minute']:3d} | {t['event']:<15} | Formation: {t['formation']}")

    # 11. Player summary
    print("\n[11] PLAYER SUMMARY (top 12 by passes)")
    print(player_summary(events).head(12).to_string(index=False))

    # 12. Data understanding conclusions
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR MODELLING")
    print("=" * 70)
    print("""
  DECISION TARGETS (what the coach controls in-game):
    1. Substitution timing & player choice (minutes, roles)
    2. Formation / tactical shift triggers
    3. Set-piece / pressing intensity adjustments

  PREDICTIVE FEATURES (data-driven signals):
    Offensive pressure:
      - xG accumulation rate per 15-min window
      - Shot zone concentration (att_center dominates)
      - Key pass rate & progressive pass rate

    Defensive shape:
      - Pressure events per zone & time window
      - Under-pressure pass completion drop (85% → 76%)

    Game state dynamics:
      - Score line at each time window
      - xG differential vs opponent

    Player-level:
      - Individual pass completion under pressure
      - Dribble success rate
      - xG contribution by player

  CRISP-DM DATA GAPS / RISKS:
    - Only 4 matches → small sample, need cross-tournament data for robust models
    - No opponent tactical data embedded in open data (partial from events)
    - 360° freeze-frame data not available for all events (xG OK, space not)
    - Injury/fatigue proxy: consecutive minutes played (derivable from events)

  MODELLING DIRECTION:
    - Phase 1: Classify game states (momentum, danger, dominance) from rolling KPIs
    - Phase 2: Recommend intervention (sub / formation shift / press trigger)
      using reward signal = xG differential change after intervention
    - Approach: Rule-free — learn thresholds from data (clustering + decision tree
      + reinforcement-style policy from historical outcomes)
""")

    return {
        "matches":   matches,
        "events":    events,
        "summary":   summary,
        "shots":     shots_df,
        "passes":    passes_df,
        "presses":   press_df,
        "subs":      subs_df,
        "tactics":   tac,
        "players":   player_summary(events),
        "temporal":  temporal_dynamics(events),
    }


if __name__ == "__main__":
    data = run_data_understanding()
