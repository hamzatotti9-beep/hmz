# Prescriptive DSS for Football Coaches — AFCON 2023 (Morocco)

A data-driven decision support system for in-game coaching decisions, built on the **CRISP-DM** framework using **StatsBomb open event data**.

## Project Structure

```
crisp_dm/
  1_data_understanding.py   ← Phase 1+2: exploration & findings (current)
  2_data_preparation.py     ← Phase 3: feature engineering
  3_modelling.py            ← Phase 4: game-state classifier + intervention model
  4_evaluation.py           ← Phase 5: model validation
app/
  streamlit_app.py          ← Phase 6: Streamlit prototype
requirements.txt
```

## CRISP-DM Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Business Understanding | ✅ | In-game coaching DSS: subs, formations, pressing |
| Data Understanding | ✅ | Morocco 4 matches, 6 736 events analysed |
| Data Preparation | 🔲 | Feature engineering — rolling KPIs, game-state vectors |
| Modelling | 🔲 | Game-state clustering + intervention recommender |
| Evaluation | 🔲 | xG-differential reward signal validation |
| Deployment | 🔲 | Streamlit prototype |

## Data Source

StatsBomb open data — AFCON 2023 (competition_id=1267, season_id=107)  
4 Morocco matches: Tanzania (3-0), Congo DR (1-1), Zambia (0-1), South Africa (0-2)

## Key Findings from Data Understanding

- **58 shots** | **7.87 xG** | **5 goals** | avg xG/shot = 0.136
- **85.3% pass completion** overall, drops to **75.6% under pressure**
- **79% of shots** come from the central attacking zone
- **639 progressive passes** | **496 pressures**
- Pass completion degrades in last 15 min (77.3%) — fatigue signal
- Pressing intensity peaks in 45-60 min window (120 pressures)
- Formations used: **4-1-4-1** (main), **4-3-3**, **4-4-2** (reactive shift)
- **19 substitutions** across 4 matches — most subs before 80th minute
