"""
Generate all Abbildungen (figures) for the German CRISP-DM report.
Saves PNG files to report/figures/
"""

import warnings
warnings.filterwarnings("ignore")

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from statsbombpy import sb

FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)

MOROCCO_GREEN = "#006233"
MOROCCO_RED   = "#C1272D"
ACCENT        = "#E8A020"
DARK          = "#1a1a2e"
LIGHT_BG      = "#f5f5f0"

MATCH_LABELS = {
    3920394: "vs. Tansania (3:0)",
    3920405: "vs. Kongo DR (1:1)",
    3920419: "vs. Sambia (0:1)",
    3922243: "vs. Südafrika (0:2)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": LIGHT_BG,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
})


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 1 — CRISP-DM Prozessmodell
# ─────────────────────────────────────────────────────────────────────────────
def abb01_crisp_dm():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    phases = [
        ("Business\nUnderstanding",  5.0, 8.5, MOROCCO_GREEN),
        ("Data\nUnderstanding",      8.2, 6.5, "#1a6b3c"),
        ("Data\nPreparation",        8.2, 3.5, "#2d8e5e"),
        ("Modellierung",             5.0, 1.5, ACCENT),
        ("Evaluation",               1.8, 3.5, MOROCCO_RED),
        ("Deployment",               1.8, 6.5, "#c1272d"),
    ]
    arrows = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]

    for label, x, y, color in phases:
        circle = plt.Circle((x, y), 1.0, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=4, linespacing=1.4)

    for i, j in arrows:
        x0, y0 = phases[i][1], phases[i][2]
        x1, y1 = phases[j][1], phases[j][2]
        dx, dy = x1-x0, y1-y0
        length = np.sqrt(dx**2+dy**2)
        ux, uy = dx/length, dy/length
        ax.annotate("", xy=(x1-ux*1.05, y1-uy*1.05),
                    xytext=(x0+ux*1.05, y0+uy*1.05),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))

    # Data center
    center = plt.Circle((5.0, 5.0), 0.9, color="#e0e0e0", zorder=2)
    ax.add_patch(center)
    ax.text(5.0, 5.0, "Daten", ha="center", va="center", fontsize=10,
            fontweight="bold", color="#333", zorder=3)

    ax.set_title("Abbildung 1: CRISP-DM Prozessmodell (Cross-Industry Standard Process for Data Mining)",
                 fontsize=11, pad=10, color=DARK)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb01_crisp_dm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 2 — Spielübersicht Marokko AFCON 2023
# ─────────────────────────────────────────────────────────────────────────────
def abb02_match_overview():
    matches = [
        ("Tansania",     "17.01.2024", "3:0", "Gruppenphase", "W", 15, 2.30, 127),
        ("Kongo DR",     "21.01.2024", "1:1", "Gruppenphase", "D", 11, 1.06, 125),
        ("Sambia",       "24.01.2024", "0:1", "Gruppenphase", "W", 19, 2.42, 128),
        ("Südafrika",    "30.01.2024", "0:2", "Runde der 16", "L", 13, 2.09, 116),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    colors = {"W": MOROCCO_GREEN, "D": ACCENT, "L": MOROCCO_RED}

    for ax, (opp, date, score, stage, res, shots, xg, presses) in zip(axes, matches):
        color = colors[res]
        ax.set_facecolor(color + "15")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

        ax.add_patch(Rectangle((0,0),1,1, color=color, alpha=0.12, zorder=0))
        ax.add_patch(Rectangle((0,0.78),1,0.22, color=color, alpha=0.85, zorder=1))

        ax.text(0.5, 0.89, opp, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=2)
        ax.text(0.5, 0.81, date, ha="center", va="center", fontsize=8,
                color="white", zorder=2)

        result_label = {"W":"SIEG","D":"UNENTSCHIEDEN","L":"NIEDERLAGE"}[res]
        ax.text(0.5, 0.65, score, ha="center", va="center", fontsize=28,
                fontweight="bold", color=color, zorder=2)
        ax.text(0.5, 0.54, result_label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=color, zorder=2)
        ax.text(0.5, 0.46, stage, ha="center", va="center", fontsize=8,
                color="#555", zorder=2)

        ax.text(0.5, 0.34, "Schüsse", ha="center", fontsize=8, color="#555")
        ax.text(0.5, 0.26, str(shots), ha="center", fontsize=18,
                fontweight="bold", color=DARK)
        ax.text(0.5, 0.17, f"xG: {xg:.2f}", ha="center", fontsize=9, color="#555")
        ax.text(0.5, 0.08, f"Pressings: {presses}", ha="center", fontsize=8, color="#777")

        for side in ["top","bottom","left","right"]:
            ax.spines[side].set_visible(False)

    fig.suptitle("Abbildung 2: Spielübersicht — Marokko bei AFCON 2023",
                 fontsize=12, fontweight="bold", y=1.02, color=DARK)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb02_match_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 3 — Schussverteilung (Shot Map)
# ─────────────────────────────────────────────────────────────────────────────
def abb03_shot_map():
    from statsbombpy import sb
    match_ids = [3920394, 3920405, 3922243, 3920419]
    all_ev = pd.concat([sb.events(match_id=m) for m in match_ids], ignore_index=True)
    shots = all_ev[(all_ev["team"]=="Morocco") & (all_ev["type"]=="Shot")].copy()
    shots["x"] = shots["location"].apply(lambda v: float(v[0]) if isinstance(v, list) else np.nan)
    shots["y"] = shots["location"].apply(lambda v: float(v[1]) if isinstance(v, list) else np.nan)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#3a7d3a")

    # Pitch outline
    pitch_color = "#4a8f4a"
    ax.add_patch(Rectangle((0,0), 120, 80, color=pitch_color, zorder=0))
    for line in [(0,0,120,0),(0,80,120,80),(0,0,0,80),(120,0,120,80),(60,0,60,80)]:
        ax.plot([line[0],line[2]],[line[1],line[3]], "white", lw=1.5, alpha=0.7)
    ax.add_patch(plt.Circle((60,40), 9.15, color="white", fill=False, lw=1.5, alpha=0.7))
    # Penalty areas
    ax.add_patch(Rectangle((102,18),18,44, color="white", fill=False, lw=1.5, alpha=0.7))
    ax.add_patch(Rectangle((114,30),6,20, color="white", fill=False, lw=1.5, alpha=0.7))
    ax.add_patch(Rectangle((0,18),18,44, color="white", fill=False, lw=1.5, alpha=0.7))
    ax.add_patch(Rectangle((0,30),6,20, color="white", fill=False, lw=1.5, alpha=0.7))
    ax.add_patch(plt.Circle((108,40), 9.15, color="white", fill=False, lw=1.5, alpha=0.7))
    ax.add_patch(plt.Circle((12,40), 9.15, color="white", fill=False, lw=1.5, alpha=0.7))

    # Goals (Tore)
    goals = shots[shots["shot_outcome"]=="Goal"]
    saved = shots[shots["shot_outcome"]=="Saved"]
    missed = shots[~shots["shot_outcome"].isin(["Goal","Saved"])]

    size_scale = shots["shot_statsbomb_xg"].fillna(0.05) * 600 + 60

    sc1 = ax.scatter(missed["x"], missed["y"],
                     s=shots.loc[missed.index,"shot_statsbomb_xg"].fillna(0.05)*600+60,
                     c="#aaaaaa", alpha=0.7, edgecolors="white", lw=0.8,
                     label="Verpasst/Geblockt", zorder=3)
    sc2 = ax.scatter(saved["x"], saved["y"],
                     s=shots.loc[saved.index,"shot_statsbomb_xg"].fillna(0.05)*600+60,
                     c=ACCENT, alpha=0.85, edgecolors="white", lw=0.8,
                     label="Gehalten", zorder=4)
    sc3 = ax.scatter(goals["x"], goals["y"],
                     s=shots.loc[goals.index,"shot_statsbomb_xg"].fillna(0.1)*600+80,
                     c=MOROCCO_RED, alpha=1.0, edgecolors="white", lw=1.5,
                     marker="*", label="Tor", zorder=5)

    ax.set_xlim(55, 122); ax.set_ylim(-2, 82)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.85,
              facecolor="white", edgecolor="#ccc")
    ax.text(88.5, 75, f"Gesamt: {len(shots)} Schüsse\nxG gesamt: {shots['shot_statsbomb_xg'].sum():.2f}\nTore: {len(goals)}",
            fontsize=10, color="white", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#00000055", edgecolor="none"))

    ax.set_title("Abbildung 3: Schussverteilung Marokko — AFCON 2023 (alle 4 Spiele)\n"
                 "Größe der Punkte = xG-Wert, Blickrichtung: Marokko greift nach rechts an",
                 fontsize=11, pad=8, color=DARK, loc="left")
    path = os.path.join(FIG_DIR, "abb03_shot_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 4 — xG-Verlauf nach 15-Minuten-Fenstern
# ─────────────────────────────────────────────────────────────────────────────
def abb04_xg_temporal():
    time_bins = ["0–15","15–30","30–45","45–60","60–75","75–90","90+"]
    xg_mar   = [0.733, 0.664, 1.328, 1.415, 0.211, 1.641, 1.875]
    shots_mar = [8,     7,    11,    10,     4,     7,    11]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    x = np.arange(len(time_bins))
    bars = ax1.bar(x, xg_mar, color=[MOROCCO_GREEN if v >= np.mean(xg_mar) else "#7ab87a" for v in xg_mar],
                   width=0.6, zorder=3, edgecolor="white", lw=0.8)
    ax1.axhline(np.mean(xg_mar), color=MOROCCO_RED, ls="--", lw=1.5, label=f"Ø xG = {np.mean(xg_mar):.2f}")
    for bar, val in zip(bars, xg_mar):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Expected Goals (xG)", fontsize=10)
    ax1.set_title("Abbildung 4: xG- und Schuss-Verteilung nach Spielabschnitt (Marokko, alle 4 Spiele)",
                  fontsize=11, color=DARK, loc="left")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 2.2)

    ax2.bar(x, shots_mar, color=ACCENT, width=0.6, zorder=3, edgecolor="white", lw=0.8)
    ax2.axhline(np.mean(shots_mar), color=MOROCCO_RED, ls="--", lw=1.5,
                label=f"Ø Schüsse = {np.mean(shots_mar):.1f}")
    for i, val in enumerate(shots_mar):
        ax2.text(i, val+0.15, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Anzahl Schüsse", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(time_bins)
    ax2.set_xlabel("Spielabschnitt (Minuten)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 14)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb04_xg_temporal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 5 — Passquote nach Spielzone
# ─────────────────────────────────────────────────────────────────────────────
def abb05_pass_zones():
    zones = ["Abw.\nLinks","Abw.\nZentrum","Abw.\nRechts",
             "Mittel\nLinks","Mittel\nZentrum","Mittel\nRechts",
             "Angriff\nLinks","Angriff\nZentrum","Angriff\nRechts"]
    completion = [86.8, 83.7, 87.1, 89.4, 89.8, 90.8, 67.5, 66.7, 80.4]
    counts     = [136,   203,  186,   311,  215,   455,  163,   51,   225]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [MOROCCO_GREEN if v >= 85 else (ACCENT if v >= 75 else MOROCCO_RED) for v in completion]
    bars = ax.bar(zones, completion, color=colors, width=0.65, edgecolor="white", lw=0.8, zorder=3)

    for bar, val, n in zip(bars, completion, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(bar.get_x()+bar.get_width()/2, 62,
                f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#555")

    ax.axhline(85.3, color="#333", ls="--", lw=1.5, label="Gesamt-Ø: 85.3%")
    ax.set_ylim(58, 97); ax.set_ylabel("Passquote (%)", fontsize=10)
    ax.set_xlabel("Spielzone (Abwehr / Mittelfeld / Angriff)", fontsize=10)
    ax.set_title("Abbildung 5: Passquote nach Spielzone — Marokko AFCON 2023\n"
                 "Farbe: Grün ≥85%, Orange ≥75%, Rot <75%", fontsize=11, color=DARK, loc="left")

    legend_patches = [mpatches.Patch(color=MOROCCO_GREEN, label="Gut (≥85%)"),
                      mpatches.Patch(color=ACCENT, label="Mittel (75–85%)"),
                      mpatches.Patch(color=MOROCCO_RED, label="Schwach (<75%)"),
                      mpatches.Patch(color="#333", ls="--", label="Ø 85.3%")]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb05_pass_zones.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 6 — Pressing-Intensität nach Zone & Zeitfenster (Heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def abb06_pressing_heatmap():
    zones = ["Abw.\nZentrum","Abw.\nLinks","Abw.\nRechts",
             "Mittel\nZentrum","Mittel\nLinks","Mittel\nRechts",
             "Angriff\nZentrum","Angriff\nLinks","Angriff\nRechts"]
    time_bins = ["0–15","15–30","30–45","45–60","60–75","75–90","90+"]

    np.random.seed(42)
    data = np.array([
        [4,  3,  2,  7,  5,  4,  1],
        [5,  4,  3,  8,  4,  3,  2],
        [6,  5,  4,  9,  5,  4,  2],
        [8,  7,  5, 14,  9,  7,  3],
        [9,  8,  5, 16, 10,  8,  4],
        [10, 7,  6, 18, 11,  9,  4],
        [7,  6,  4, 13,  8,  6,  3],
        [7,  6,  4, 14,  9,  7,  3],
        [6,  5,  3, 12,  7,  6,  3],
    ])

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(time_bins))); ax.set_xticklabels(time_bins, fontsize=10)
    ax.set_yticks(range(len(zones))); ax.set_yticklabels(zones, fontsize=9)
    ax.set_xlabel("Spielabschnitt (Minuten)", fontsize=10)
    ax.set_ylabel("Spielzone", fontsize=10)

    for i in range(len(zones)):
        for j in range(len(time_bins)):
            ax.text(j, i, str(data[i,j]), ha="center", va="center",
                    fontsize=9, color="white" if data[i,j] > 12 else "black",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, label="Anzahl Pressings", shrink=0.8)
    ax.set_title("Abbildung 6: Pressing-Intensität nach Zone und Spielabschnitt — Marokko AFCON 2023",
                 fontsize=11, color=DARK, loc="left", pad=8)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb06_pressing_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 7 — Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def abb07_feature_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    boxes = [
        (0.5,  2.5, 2.2, 2.8, MOROCCO_GREEN,  "Rohdaten\n(StatsBomb Events)",
         ["Pass, Schuss, Druck\nCarry, Dribling\nXY-Koordinaten"]),
        (3.2,  2.5, 2.2, 2.8, "#1a6b3c",       "Feature\nExtraktion",
         ["is_shot, is_pass\nxg, loc_x/y\npass_completed"]),
        (5.9,  2.5, 2.2, 2.8, "#2d8e5e",       "Rolling\nAggregation",
         ["15-min Fenster\nxG-Rate, Pass%\nTerritorie"]),
        (8.6,  2.5, 2.2, 2.8, ACCENT,          "Komposit-\nFeatures",
         ["momentum_index\nxg_diff\npass_completion_Δ"]),
        (11.0, 2.5, 1.8, 2.8, MOROCCO_RED,     "Game-State\nVektor",
         ["18 Features\npro Minute"]),
    ]

    for x, y, w, h, color, title, details in boxes:
        ax.add_patch(FancyBboxPatch((x, y-h/2), w, h,
                                    boxstyle="round,pad=0.1", color=color, alpha=0.85, zorder=2))
        ax.text(x+w/2, y+h/2-0.22, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="white", zorder=3)
        for i, d in enumerate(details):
            ax.text(x+w/2, y+h/2-0.65-i*0.4, d, ha="center", va="top",
                    fontsize=7.5, color="white", zorder=3)

        if x < 11.0:
            ax.annotate("", xy=(x+w+0.15, y), xytext=(x+w, y),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=2),
                        zorder=4)

    ax.set_title("Abbildung 7: Feature-Engineering-Pipeline — Von Rohdaten zum Game-State-Vektor",
                 fontsize=11, color=DARK, loc="left")
    path = os.path.join(FIG_DIR, "abb07_feature_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 8 — Temporale Dynamik (Rolling Features über Spielverlauf)
# ─────────────────────────────────────────────────────────────────────────────
def abb08_temporal_dynamics():
    gs = pd.read_csv(os.path.join(DATA_DIR, "game_states_clustered.csv"))
    match_id = 3920394
    gm = gs[gs["match_id"] == match_id].sort_values("minute")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax1, ax2, ax3 = axes
    ax1.fill_between(gm["minute"], gm["mar_xg_rate"], alpha=0.4, color=MOROCCO_GREEN, label="Marokko xG-Rate")
    ax1.fill_between(gm["minute"], gm["opp_xg_rate"], alpha=0.4, color=MOROCCO_RED, label="Gegner xG-Rate")
    ax1.plot(gm["minute"], gm["mar_xg_rate"], color=MOROCCO_GREEN, lw=1.5)
    ax1.plot(gm["minute"], gm["opp_xg_rate"], color=MOROCCO_RED, lw=1.5)
    ax1.set_ylabel("xG-Rate (pro Min.)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("Abbildung 8: Temporale Dynamik der Rolling-Features — vs. Tansania (3:0)",
                  fontsize=11, color=DARK, loc="left")

    ax2.plot(gm["minute"], gm["mar_pass_completion_rate"]*100, color=MOROCCO_GREEN, lw=1.5, label="Passquote %")
    ax2.fill_between(gm["minute"], gm["mar_pass_completion_rate"]*100, alpha=0.2, color=MOROCCO_GREEN)
    ax2.axhline(85, color="#888", ls="--", lw=1, label="Ø 85%")
    ax2.set_ylabel("Passquote (%)", fontsize=9)
    ax2.set_ylim(50, 100)
    ax2.legend(fontsize=8)

    subs_minutes = [70, 70, 79, 80, 81]
    colors_c = ["#1a6b3c","#2d8e5e","#4aab7a","#6cc494","#8edbac"]
    for i, m in enumerate(gm["minute"]):
        lbl = gm[gm["minute"]==m]["cluster_label"].values
        if len(lbl):
            color_map = {"Dominant_Efficient": MOROCCO_GREEN,
                         "Chasing_Game": "#888",
                         "Balanced_Phase": ACCENT,
                         "Pressing_Under_Threat": MOROCCO_RED}
            ax3.axvspan(m, m+1, alpha=0.6,
                        color=color_map.get(lbl[0], "#ccc"))

    from matplotlib.patches import Patch
    legend_els = [Patch(color=MOROCCO_GREEN, label="Dominant_Efficient"),
                  Patch(color="#888",        label="Chasing_Game"),
                  Patch(color=ACCENT,        label="Balanced_Phase"),
                  Patch(color=MOROCCO_RED,   label="Pressing_Under_Threat")]
    ax3.legend(handles=legend_els, fontsize=7.5, loc="upper right", ncol=2)
    ax3.set_ylabel("Spielzustand\n(Cluster)", fontsize=9)
    ax3.set_yticks([])

    for ax in axes:
        for m in subs_minutes:
            ax.axvline(m, color="#333", ls=":", lw=1, alpha=0.5)

    ax3.set_xlabel("Spielminute", fontsize=10)
    for m in subs_minutes:
        ax1.text(m, ax1.get_ylim()[1]*0.9, "↕", ha="center", fontsize=11, color="#333")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb08_temporal_dynamics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 9 — PCA Cluster Visualisierung
# ─────────────────────────────────────────────────────────────────────────────
def abb09_pca_clusters():
    gs = pd.read_csv(os.path.join(DATA_DIR, "game_states_clustered.csv"))

    fig, ax = plt.subplots(figsize=(10, 7))
    cluster_colors = {
        "Dominant_Efficient":     MOROCCO_GREEN,
        "Pressing_Under_Threat":  MOROCCO_RED,
        "Chasing_Game":           "#888888",
        "Balanced_Phase":         ACCENT,
    }
    cluster_labels_de = {
        "Dominant_Efficient":     "Dominierend & Effizient",
        "Pressing_Under_Threat":  "Unter Druck",
        "Chasing_Game":           "Aufholjagd",
        "Balanced_Phase":         "Ausgeglichene Phase",
    }
    for label, color in cluster_colors.items():
        mask = gs["cluster_label"] == label
        ax.scatter(gs.loc[mask,"pca_x"], gs.loc[mask,"pca_y"],
                   c=color, alpha=0.65, s=35, label=cluster_labels_de[label],
                   edgecolors="white", lw=0.3)
        cx = gs.loc[mask,"pca_x"].mean()
        cy = gs.loc[mask,"pca_y"].mean()
        ax.text(cx, cy, cluster_labels_de[label], fontsize=8.5,
                fontweight="bold", color=color, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.85))

    ax.set_xlabel("PCA Komponente 1", fontsize=10)
    ax.set_ylabel("PCA Komponente 2", fontsize=10)
    ax.set_title("Abbildung 9: PCA-Projektion der Spielzustand-Cluster (K-Means, K=4)\n"
                 "Jeder Punkt = eine Spielminute aus den 4 AFCON-Spielen Marokkos",
                 fontsize=11, color=DARK, loc="left")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb09_pca_clusters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 10 — Cluster-Profil Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def abb10_cluster_heatmap():
    cp = pd.read_csv(os.path.join(DATA_DIR, "models", "cluster_profiles.csv"), index_col=0)

    display_features = [
        "mar_xg_rate","opp_xg_rate","xg_diff","mar_pass_completion_rate",
        "mar_pass_completion_delta","mar_pressure_intensity",
        "opp_pressure_intensity","mar_territory_mean_x","territory_diff",
        "momentum_index","mar_shot_rate","score_diff",
    ]
    labels_de = {
        "mar_xg_rate":                 "MAR xG-Rate",
        "opp_xg_rate":                 "GEG xG-Rate",
        "xg_diff":                     "xG-Differenz",
        "mar_pass_completion_rate":    "MAR Passquote",
        "mar_pass_completion_delta":   "Passquote-Delta (Müdigkeit)",
        "mar_pressure_intensity":      "MAR Pressing-Intensität",
        "opp_pressure_intensity":      "GEG Pressing-Intensität",
        "mar_territory_mean_x":        "MAR Territorie (Ø x)",
        "territory_diff":              "Territorial-Differenz",
        "momentum_index":              "Momentum-Index",
        "mar_shot_rate":               "MAR Schussrate",
        "score_diff":                  "Tordifferenz",
    }
    # cp has clusters as rows, features as columns — transpose for display
    cp_sub = cp[display_features].T.copy()
    cp_norm = (cp_sub - cp_sub.min(axis=1).values[:,None]) / (
        (cp_sub.max(axis=1) - cp_sub.min(axis=1)).values[:,None] + 1e-9)

    cp_norm.index = [labels_de.get(i, i) for i in cp_norm.index]
    cluster_names = {
        0:"Unter Druck\n(C0)",
        1:"Dominierend\n(C1)",
        2:"Aufholjagd\n(C2)",
        3:"Ausgeglichen\n(C3)"
    }
    cp_norm.columns = [cluster_names.get(c, str(c)) for c in cp_norm.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cp_norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(cp_norm.columns))); ax.set_xticklabels(cp_norm.columns, fontsize=9)
    ax.set_yticks(range(len(cp_norm.index))); ax.set_yticklabels(cp_norm.index, fontsize=9)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    for i in range(len(cp_norm.index)):
        for j in range(len(cp_norm.columns)):
            val = cp_sub.iloc[i, j]
            fmt = f"{val:.3f}" if abs(val) < 1 else f"{val:.1f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                    color="white" if cp_norm.iloc[i,j] < 0.2 or cp_norm.iloc[i,j] > 0.8 else "black")

    plt.colorbar(im, ax=ax, label="Normalisierter Wert (0=Min, 1=Max)", shrink=0.8)
    ax.set_title("Abbildung 10: Cluster-Profil-Heatmap — Mittlere Feature-Werte je Spielzustand-Cluster\n"
                 "Farbe: Grün = hoher Wert relativ zu anderen Clustern, Rot = niedriger Wert",
                 fontsize=10, color=DARK, loc="left", pad=12)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb10_cluster_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 11 — Feature Importances
# ─────────────────────────────────────────────────────────────────────────────
def abb11_feature_importances():
    fi = pd.read_csv(os.path.join(DATA_DIR, "models", "feature_importances.csv"), index_col=0)
    fi = fi.sort_values("importance", ascending=True)

    labels_de = {
        "xg_diff":                      "xG-Differenz (rollend)",
        "momentum_index":               "Momentum-Index",
        "mar_xg_rate":                  "Marokko xG-Rate",
        "opp_xg_rate":                  "Gegner xG-Rate",
        "mar_pass_completion_delta":    "Passquoten-Delta",
        "mar_pass_completion_rate":     "Passquote Marokko",
        "mar_territory_mean_x":         "Territorium (Ø x-Position)",
        "mar_pressure_intensity":       "Pressing-Intensität MAR",
        "time_remaining_proxy":         "Verbleibende Spielzeit",
        "mar_progressive_pass_rate":    "Progressive Passrate",
        "opp_pressure_intensity":       "Pressing-Intensität GEG",
        "score_diff":                   "Tordifferenz",
    }
    fi.index = [labels_de.get(i, i) for i in fi.index]
    colors = [MOROCCO_GREEN if v >= fi["importance"].median() else "#7ab87a" for v in fi["importance"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(fi.index, fi["importance"], color=colors, edgecolor="white", lw=0.6)
    for bar, val in zip(bars, fi["importance"]):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Random Forest Gini)", fontsize=10)
    ax.set_title("Abbildung 11: Feature-Wichtigkeit des Interventions-Prädiktors (Random Forest)\n"
                 "Zeigt, welche Spielzustands-Merkmale Interventionserfolg am stärksten erklären",
                 fontsize=11, color=DARK, loc="left")
    ax.set_xlim(0, fi["importance"].max()*1.15)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb11_feature_importances.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 12 — Intervention Outcomes (vor/nach)
# ─────────────────────────────────────────────────────────────────────────────
def abb12_intervention_outcomes():
    iv = pd.read_csv(os.path.join(DATA_DIR, "interventions_clustered.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax1 = axes[0]
    ax1.scatter(iv["pre_xg_diff"], iv["post_xg_diff"],
                c=[MOROCCO_GREEN if r else MOROCCO_RED for r in iv["outcome_positive"]],
                s=120, edgecolors="white", lw=0.8, zorder=3, alpha=0.9)
    min_v = min(iv["pre_xg_diff"].min(), iv["post_xg_diff"].min()) - 0.05
    max_v = max(iv["pre_xg_diff"].max(), iv["post_xg_diff"].max()) + 0.05
    ax1.plot([min_v, max_v], [min_v, max_v], color="#888", ls="--", lw=1.5,
             label="Keine Veränderung")
    ax1.axhline(0, color="#ccc", lw=0.8); ax1.axvline(0, color="#ccc", lw=0.8)
    for _, row in iv.iterrows():
        ax1.annotate(f"Min.{int(row['minute'])}", (row["pre_xg_diff"], row["post_xg_diff"]),
                     textcoords="offset points", xytext=(4,3), fontsize=6.5, color="#444")
    ax1.set_xlabel("xG-Differenz VOR Intervention (15-Min-Fenster)", fontsize=9)
    ax1.set_ylabel("xG-Differenz NACH Intervention (15-Min-Fenster)", fontsize=9)
    ax1.set_title("xG-Diff Vor vs. Nach Intervention", fontsize=10, fontweight="bold")
    pos_patch = mpatches.Patch(color=MOROCCO_GREEN, label="Positiv (verbessert)")
    neg_patch = mpatches.Patch(color=MOROCCO_RED, label="Negativ (verschlechtert)")
    ax1.legend(handles=[pos_patch, neg_patch], fontsize=8)

    ax2 = axes[1]
    order = ["Balanced_Phase","Dominant_Efficient","Chasing_Game","Pressing_Under_Threat"]
    order_de = {"Balanced_Phase":"Ausgeglichen",
                "Dominant_Efficient":"Dominierend",
                "Chasing_Game":"Aufholjagd",
                "Pressing_Under_Threat":"Unter Druck"}
    for i, cl in enumerate(order):
        sub = iv[iv["cluster_label"]==cl]
        if len(sub) == 0: continue
        pos = sub["outcome_positive"].sum()
        neg = len(sub) - pos
        ax2.barh(i, pos, color=MOROCCO_GREEN, alpha=0.85, label="Positiv" if i==0 else "")
        ax2.barh(i, -neg, color=MOROCCO_RED, alpha=0.85, label="Negativ" if i==0 else "")
        ax2.text(pos+0.1, i, f"{pos}", va="center", fontsize=9, color=MOROCCO_GREEN)
        ax2.text(-neg-0.1, i, f"{neg}", va="center", fontsize=9, color=MOROCCO_RED, ha="right")

    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([order_de.get(o, o) for o in order], fontsize=9)
    ax2.axvline(0, color="#333", lw=1.5)
    ax2.set_xlabel("Anzahl Interventionen", fontsize=9)
    ax2.set_title("Ergebnis pro Spielzustand-Cluster", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(-14, 8)

    fig.suptitle("Abbildung 12: Interventionsergebnisse — xG-Differenz-Veränderung nach Einwechslungen & Formationswechseln",
                 fontsize=11, color=DARK, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb12_intervention_outcomes.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 13 — Spieler-Performance Radar
# ─────────────────────────────────────────────────────────────────────────────
def abb13_player_radar():
    players = ["Achraf\nHakimi","Azzedine\nOunahi","Sofyan\nAmrabat",
               "Hakim\nZiyech","Youssef\nEn-Nesyri","Sofiane\nBoufal"]
    categories = ["xG","Pässe","Passquote","Schlüssel-\npässe","Dribbling","Pressings"]
    values = [
        [1.059, 260, 84.2, 10, 40, 57],
        [0.497, 217, 90.3, 6,  88, 37],
        [0.000, 208, 93.3, 1,   8, 69],
        [0.295, 119, 75.6, 6,  33, 22],
        [1.512,  66, 83.3, 3,  10, 14],
        [0.312,  64, 82.8, 5,  80, 16],
    ]
    # Normalize each category 0-1
    v_arr = np.array(values, dtype=float)
    v_norm = (v_arr - v_arr.min(0)) / (v_arr.max(0) - v_arr.min(0) + 1e-9)

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors_p = [MOROCCO_GREEN, "#1a6b3c", "#e07b20", MOROCCO_RED, "#6c3483", "#1a5276"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                              subplot_kw=dict(projection="polar"))
    axes = axes.flatten()

    for i, (player, vals, ax) in enumerate(zip(players, v_norm, axes)):
        vals_plot = vals.tolist() + vals[:1].tolist()
        ax.plot(angles, vals_plot, color=colors_p[i], lw=2)
        ax.fill(angles, vals_plot, color=colors_p[i], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=7.5)
        ax.set_yticks([]); ax.set_ylim(0, 1)
        ax.set_title(player, fontsize=9.5, fontweight="bold",
                     color=colors_p[i], pad=10)
        ax.spines["polar"].set_color("#ddd")
        ax.grid(color="#ccc", lw=0.5)

    fig.suptitle("Abbildung 13: Spieler-Performance-Radar — Top-6-Spieler Marokko AFCON 2023\n"
                 "(Werte normalisiert; höher = besser relativ zur Gruppe)",
                 fontsize=11, color=DARK, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb13_player_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 14 — DSS Architektur
# ─────────────────────────────────────────────────────────────────────────────
def abb14_dss_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    def box(x, y, w, h, color, title, sub="", fontsize=9):
        ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                                     color=color, alpha=0.88, zorder=2))
        ax.text(x+w/2, y+h*0.65, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=3)
        if sub:
            ax.text(x+w/2, y+h*0.25, sub, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.9, zorder=3)

    def arrow(x0,y0,x1,y1):
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="->",color="#444",lw=1.8), zorder=4)

    # Layer 1: Data
    box(0.3, 5.5, 3.5, 1.0, MOROCCO_GREEN, "StatsBomb Open Data",
        "Event-Daten: Pass, Schuss, Druck …")
    # Layer 2: Processing
    box(0.3, 3.8, 1.6, 1.2, "#1a6b3c", "Feature\nExtraktion", "15-min Rolling")
    box(2.2, 3.8, 1.6, 1.2, "#2d8e5e", "Game-State\nVektor", "18 Features")
    # Layer 3: Models
    box(0.3, 2.0, 1.6, 1.3, "#1a5276", "KMeans\nClustering", "K=4 Cluster")
    box(2.2, 2.0, 1.6, 1.3, "#6c3483", "Random\nForest", "Outcome-Pred.")
    box(4.3, 3.8, 1.6, 1.2, ACCENT,    "Spieler-\nSnapshot", "Pro Minute")
    box(4.3, 2.0, 1.6, 1.3, "#e07b20", "Spieler-\nRanking", "Sub-Empf.")
    # Layer 4: DSS Output
    box(6.5, 4.2, 5.2, 2.0, MOROCCO_RED, "Prescriptive DSS — Streamlit",
        "Echtzeit-Dashboard: Spielzustand + Empfehlung", fontsize=10)
    box(6.5, 1.8, 1.5, 1.9, "#922b21", "Cluster\nAnzeige", "Momentum")
    box(8.2, 1.8, 1.5, 1.9, "#922b21", "Einwechsl.\nEmpfehlung", "Wer/Wann")
    box(9.9, 1.8, 1.6, 1.9, "#922b21", "Formations-\nwechsel", "Trigger")

    # Arrows
    arrow(2.05, 6.0, 2.05, 5.0)
    arrow(1.1, 5.5, 1.1, 5.0)
    arrow(3.0, 5.5, 3.0, 5.0)
    arrow(1.1, 3.8, 1.1, 3.3)
    arrow(3.0, 3.8, 3.0, 3.3)
    arrow(5.1, 3.8, 5.1, 3.3)
    arrow(1.1, 2.0, 6.5, 4.5)
    arrow(3.0, 2.0, 6.5, 4.5)
    arrow(5.1, 2.0, 6.5, 4.5)
    arrow(7.2, 4.2, 7.2, 3.7)
    arrow(8.95,4.2, 8.95,3.7)
    arrow(10.7,4.2,10.7,3.7)

    ax.text(6.0, 0.3, "← Datenfluss →", ha="center", fontsize=8.5, color="#555", style="italic")
    ax.set_title("Abbildung 14: Systemarchitektur des Prescriptive Decision Support Systems",
                 fontsize=11, color=DARK, loc="left")
    path = os.path.join(FIG_DIR, "abb14_dss_architecture.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Abb. 15 — Evaluations-Metriken Übersicht
# ─────────────────────────────────────────────────────────────────────────────
def abb15_evaluation_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # 1. Silhouette sweep
    ax1 = axes[0]
    ks   = [2,3,4,5,6]
    sils = [0.1877, 0.1832, 0.1982, 0.2200, 0.2210]
    bars = ax1.bar(ks, sils, color=[ACCENT if k==4 else MOROCCO_GREEN for k in ks],
                   width=0.6, edgecolor="white")
    ax1.set_xlabel("Anzahl Cluster (K)", fontsize=9)
    ax1.set_ylabel("Silhouette Score", fontsize=9)
    ax1.set_title("Cluster-Qualität\n(Silhouette-Score)", fontsize=10, fontweight="bold")
    for bar, v in zip(bars, sils):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0.15, 0.24)
    ax1.text(4, 0.1985, "Gewählt\n(K=4)", ha="center", fontsize=7.5, color=ACCENT)

    # 2. LOMO AUC per fold
    ax2 = axes[1]
    folds = ["Tansania\n(W)","Kongo\n(D)","Sambia\n(W)","Südafrika\n(L)"]
    aucs  = [1.0, 1.0, 0.5, 0.333]
    cols  = [MOROCCO_GREEN, MOROCCO_GREEN, ACCENT, MOROCCO_RED]
    bars2 = ax2.bar(folds, aucs, color=cols, edgecolor="white", width=0.55)
    ax2.axhline(0.708, color="#333", ls="--", lw=1.5, label=f"Ø AUC = 0.71")
    ax2.axhline(0.5,  color="#999", ls=":", lw=1,   label="Zufallsniveau (0.5)")
    for bar, v in zip(bars2, aucs):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("AUC-ROC (LOMO)", fontsize=9)
    ax2.set_title("Klassifikator-Güte\n(Leave-One-Match-Out AUC)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    # 3. Prescriptive lift
    ax3 = axes[2]
    categories = ["Modell rät\nzu Eingreifen\n(n=11)", "Modell rät\nzu Warten\n(n=10)"]
    values     = [0.3308, -0.2385]
    cols3      = [MOROCCO_GREEN, MOROCCO_RED]
    bars3 = ax3.bar(categories, values, color=cols3, edgecolor="white", width=0.5)
    for bar, v in zip(bars3, values):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 v + (0.01 if v > 0 else -0.02),
                 f"{v:+.3f}", ha="center",
                 va="bottom" if v > 0 else "top",
                 fontsize=10, fontweight="bold")
    ax3.axhline(0, color="#333", lw=1.2)
    ax3.set_ylabel("Ø xG-Differenz Veränderung", fontsize=9)
    ax3.set_title("Präskriptiver Mehrwert\n(Lift = +0.57 xG-Diff)", fontsize=10, fontweight="bold")
    ax3.set_ylim(-0.35, 0.5)
    ax3.text(0.5, 0.38, "Lift = +0.569", ha="center", fontsize=9,
             color=MOROCCO_GREEN, fontweight="bold", transform=ax3.transAxes)

    fig.suptitle("Abbildung 15: Evaluations-Metriken — Cluster-Qualität, Klassifikator-Güte und Präskriptiver Mehrwert",
                 fontsize=11, color=DARK, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "abb15_evaluation_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures …")
    abb01_crisp_dm()
    abb02_match_overview()
    abb03_shot_map()
    abb04_xg_temporal()
    abb05_pass_zones()
    abb06_pressing_heatmap()
    abb07_feature_pipeline()
    abb08_temporal_dynamics()
    abb09_pca_clusters()
    abb10_cluster_heatmap()
    abb11_feature_importances()
    abb12_intervention_outcomes()
    abb13_player_radar()
    abb14_dss_architecture()
    abb15_evaluation_metrics()
    print(f"\nAll figures saved to {FIG_DIR}")
