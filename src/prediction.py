"""
RACE PREDICTION (INFERENCE)
===========================
Predict the finishing order of a race from pre-race information only, using the
model saved by the notebook (models/final_model.joblib). Nothing is retrained.

    table = predict_race(
        {"season": 2026, "circuit": "Autodromo Nazionale di Monza", "race_name": "Italian Grand Prix"},
        entries,   # DataFrame: driver, constructor, qualifying_position, grid_position
    )

How the features are built (same definitions as training):
  1. Keep only historical races strictly BEFORE the race being predicted.
  2. Append the new race's entry list with placeholder results.
  3. Run the unchanged features.build_features() and take the new race's rows.
  4. Leakage guard: repeat step 2–3 with completely different placeholder results.
     Every model feature must come out identical; otherwise the race's own
     (unknown) result would be influencing its features and an error is raised.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data_collection
import features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "final_model.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "race_data.csv"

INPUT_COLUMNS = ["driver", "constructor", "qualifying_position", "grid_position"]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_model(path=MODEL_FILE):
    """Load the trained pipeline saved by the notebook (preprocessing + model)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebooks/f1_race_prediction.ipynb once "
            "(Kernel → Restart Kernel and Run All Cells) to create it.")
    return joblib.load(path)


def load_history(path=DATA_FILE):
    """The historical results table the model's history features are built from."""
    return pd.read_csv(path)


def model_feature_names(model):
    """The exact input columns the saved model was trained on, read from the model itself."""
    return list(model.feature_names_in_)


# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------
def prepare_entries(qualifying_results, grid_same_as_qualifying=False, allow_duplicate_qualifying=False):
    """
    Validate the user's entry list. Required columns: driver, constructor,
    qualifying_position, grid_position (grid may be omitted only when
    grid_same_as_qualifying=True). Grid 0 means a pit-lane start.
    Duplicate qualifying positions are rejected (almost always a typo), except when
    replaying a historical race: the dataset contains a few, because a missing
    qualifying time was filled with the grid position during cleaning.
    """
    entries = pd.DataFrame(qualifying_results).copy()
    if grid_same_as_qualifying and "grid_position" not in entries:
        entries["grid_position"] = entries["qualifying_position"]
    elif grid_same_as_qualifying:
        entries["grid_position"] = entries["grid_position"].fillna(entries["qualifying_position"])

    missing = [c for c in INPUT_COLUMNS if c not in entries]
    if missing:
        raise ValueError(f"Missing column(s): {missing}. Needed: {INPUT_COLUMNS}")

    entries = entries[INPUT_COLUMNS].dropna(subset=["driver"])
    entries["driver"] = entries["driver"].astype(str).str.strip()
    entries["constructor"] = entries["constructor"].astype(str).str.strip()
    if entries.empty:
        raise ValueError("No drivers entered.")
    if entries["driver"].duplicated().any():
        raise ValueError(f"Driver listed twice: {sorted(entries.loc[entries['driver'].duplicated(), 'driver'])}")
    if entries[["qualifying_position", "grid_position"]].isna().any().any():
        raise ValueError("Every driver needs a qualifying position and a grid position.")

    entries["qualifying_position"] = entries["qualifying_position"].astype(int)
    entries["grid_position"] = entries["grid_position"].astype(int)
    if entries["qualifying_position"].duplicated().any() and not allow_duplicate_qualifying:
        raise ValueError("Two drivers have the same qualifying position.")
    if (entries["qualifying_position"] < 1).any() or (entries["grid_position"] < 0).any():
        raise ValueError("Qualifying positions start at 1; grid positions at 1 (or 0 for a pit-lane start).")
    return entries.reset_index(drop=True)


def resolve_round(history, season, round_number=None):
    """
    Where the race sits in time. Features only use races strictly before (season, round).
      - round given          → use it (e.g. replaying a historical race)
      - season = latest one  → the race after the last one in the data
      - season > latest one  → first race of that season (no results of that season known)
    """
    latest_season = int(history["season"].max())
    if round_number is not None:
        return int(round_number)
    if season == latest_season:
        return int(history.loc[history["season"] == season, "round"].max()) + 1
    if season > latest_season:
        return 1
    raise ValueError(f"For a past season ({season}), give the race's round number.")


def history_before(history, season, round_number):
    """Only races strictly before the race being predicted."""
    earlier = (history["season"] < season) | ((history["season"] == season) & (history["round"] < round_number))
    return history[earlier]


# ----------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------
def _race_rows(entries, season, round_number, circuit, race_name, variant):
    """The new race as rows in the historical format, with PLACEHOLDER results."""
    rows = entries.copy()
    rows["season"] = season
    rows["round"] = round_number
    rows["race_name"] = race_name
    rows["race_date"] = ""
    rows["circuit"] = circuit
    rows["driver_id"] = rows["driver"]
    # The result is unknown before the race. Two very different placeholders are used,
    # and the leakage guard checks that they make no difference to the features.
    n = len(rows)
    if variant == 0:
        rows["final_position"], rows["points"], rows["status"] = np.arange(1, n + 1), 0.0, "Finished"
    else:
        rows["final_position"], rows["points"], rows["status"] = np.arange(n, 0, -1), 25.0, "Retired"
    return data_collection.clean_results(rows)   # same cleaning as training (pit-lane starts etc.)


def build_race_features(race_info, entries, history, feature_names):
    """
    Construct the model's feature matrix for one race, in the order of `entries`.
    Returns (feature_matrix, info) where info describes the history used.
    """
    season = int(race_info["season"])
    circuit = race_info["circuit"]
    race_name = race_info.get("race_name") or circuit
    round_number = resolve_round(history, season, race_info.get("round"))
    past = history_before(history, season, round_number)

    # Check: nothing at or after the prediction race is in the history used.
    assert not ((past["season"] > season) | ((past["season"] == season) & (past["round"] >= round_number))).any()

    built_versions = []
    for variant in (0, 1):
        combined = pd.concat([past, _race_rows(entries, season, round_number, circuit, race_name, variant)],
                             ignore_index=True)
        built = features.build_features(combined)
        race = built[(built["season"] == season) & (built["round"] == round_number)]
        built_versions.append(race.set_index("driver").loc[entries["driver"]].reset_index())

    # Leakage guard: the race's own (placeholder) result must not affect any model feature.
    pd.testing.assert_frame_equal(built_versions[0][feature_names], built_versions[1][feature_names],
                                  check_dtype=False)

    last = past.sort_values(["season", "round"]).iloc[-1] if len(past) else None
    info = {
        "season": season,
        "round": round_number,
        "history_rows": len(past),
        "last_race_used": f"{int(last['season'])} {last['race_name']}" if last is not None else "none",
        "notes": [],
    }
    if season > history["season"].max():
        info["notes"].append(
            f"The data has no {season} races, so recent form comes from races up to {info['last_race_used']} "
            f"and championship points before the race are 0. Update data/race_data.csv for up-to-date form.")
    known_drivers, known_teams = set(history["driver"]), set(history["constructor"])
    new_drivers = sorted(set(entries["driver"]) - known_drivers)
    new_teams = sorted(set(entries["constructor"]) - known_teams)
    if new_drivers:
        info["notes"].append(f"No history for {', '.join(new_drivers)}: recent-form features fall back to "
                             "their qualifying position (the same rule used in training).")
    if new_teams:
        info["notes"].append(f"No history for team(s) {', '.join(new_teams)}: team form falls back to "
                             "qualifying position.")
    return built_versions[0][feature_names], info


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def predict_race(race_info, qualifying_results, model=None, history=None,
                 grid_same_as_qualifying=False, allow_duplicate_qualifying=False, return_features=False):
    """
    Predict a race from pre-race information.

    race_info: {"season": int, "circuit": str, "race_name": optional str, "round": optional int}
    qualifying_results: table with driver, constructor, qualifying_position, grid_position

    Returns a table sorted into the predicted race order, with:
      Predicted Finish (raw) – the model's output, used for sorting
      Predicted Position     – the same value clipped to [1, number of drivers] for display only
    """
    model = load_model() if model is None else model
    history = load_history() if history is None else history
    feature_names = model_feature_names(model)
    entries = prepare_entries(qualifying_results, grid_same_as_qualifying, allow_duplicate_qualifying)

    feature_matrix, info = build_race_features(race_info, entries, history, feature_names)
    raw_prediction = model.predict(feature_matrix)

    table = pd.DataFrame({
        "Driver": entries["driver"],
        "Constructor": entries["constructor"],
        "Qualifying": entries["qualifying_position"],
        "Grid": entries["grid_position"],
        "Predicted Finish (raw)": raw_prediction,
        # Display only: a finishing position can't be below 1 or beyond the last car.
        "Predicted Position": np.clip(raw_prediction, 1, len(entries)).round(1),
    })
    # Sort by the RAW prediction (not rounded, not clipped); grid breaks exact ties.
    table = table.sort_values(["Predicted Finish (raw)", "Grid"], kind="stable").reset_index(drop=True)
    table.insert(0, "Predicted Order", range(1, len(table) + 1))
    table.attrs["info"] = info

    if return_features:
        return table, feature_matrix.assign(driver=entries["driver"].to_numpy())
    return table
