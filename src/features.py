"""
FEATURE ENGINEERING
===================
Functions that add "history" features to the race table.

The golden rule: a feature for a race may only use races BEFORE it.
Every function here enforces that with shift(1) or by subtracting the
current race's own value.

These functions are imported and demonstrated step by step in the notebook.
"""

import pandas as pd

WINDOW = 5   # "last 5 races"


def average_of_previous_races(values, window=WINDOW):
    """
    For values in time order, the average of the previous `window` values,
    NOT including the current one.

        finishes:  3,   5,   2,   4
        result:  NaN, 3.0, 4.0, 3.33
    """
    return values.shift(1).rolling(window=window, min_periods=1).mean()


def add_driver_form(races):
    """driver_avg_finish_last_5 and driver_avg_qualifying_last_5."""
    races = races.sort_values(["season", "round"]).copy()
    by_driver = races.groupby("driver")
    races["driver_avg_finish_last_5"] = by_driver["final_position"].transform(average_of_previous_races)
    races["driver_avg_qualifying_last_5"] = by_driver["qualifying_position"].transform(average_of_previous_races)
    return races


def add_constructor_form(races):
    """
    constructor_avg_finish_last_5.
    First ONE number per team per race (average of its two cars), then the
    rolling average over races. Rolling over rows would let a teammate's
    result from the same race leak in.
    """
    team_races = (
        races.groupby(["season", "round", "constructor"], as_index=False)["final_position"].mean()
        .rename(columns={"final_position": "team_finish_this_race"})
        .sort_values(["season", "round"])
    )
    team_races["constructor_avg_finish_last_5"] = (
        team_races.groupby("constructor")["team_finish_this_race"].transform(average_of_previous_races)
    )
    return races.merge(
        team_races[["season", "round", "constructor", "constructor_avg_finish_last_5"]],
        on=["season", "round", "constructor"], how="left",
    )


def add_points_before_race(races):
    """
    Championship points scored earlier in the SAME season (resets each year).
    points before race = running total INCLUDING this race - points scored in this race.
    Note: only Grand Prix points are available here, not sprint points.
    """
    races = races.sort_values(["season", "round"]).copy()
    running_total = races.groupby(["driver", "season"])["points"].cumsum()
    races["driver_points_before_race"] = running_total - races["points"]

    team_races = (
        races.groupby(["season", "round", "constructor"], as_index=False)["points"].sum()
        .sort_values(["season", "round"])
    )
    team_running_total = team_races.groupby(["constructor", "season"])["points"].cumsum()
    team_races["constructor_points_before_race"] = team_running_total - team_races["points"]
    return races.merge(
        team_races[["season", "round", "constructor", "constructor_points_before_race"]],
        on=["season", "round", "constructor"], how="left",
    )


def std_of_previous_races(values, window=WINDOW):
    """Standard deviation of the previous `window` values (needs at least 2; otherwise NaN)."""
    return values.shift(1).rolling(window=window, min_periods=2).std()


def add_consistency_and_reliability(races):
    """
    Four extra history features, all from PREVIOUS races only:
      driver_finish_std_last_5       how much the driver's finishes varied (consistency)
      driver_qualifying_std_last_5   how much their qualifying positions varied
      driver_dnf_rate_last_5         share of the driver's last 5 races that were non-finishes
      constructor_dnf_rate_last_5    share of the team's cars that failed to finish, last 5 races
    Early-career rows stay NaN here; the modelling pipeline fills them with the
    TRAINING-set median, so no information from later seasons is used.
    Requires the did_not_finish column (see add_dnf_flag).
    """
    races = races.sort_values(["season", "round"]).copy()
    by_driver = races.groupby("driver")
    races["driver_finish_std_last_5"] = by_driver["final_position"].transform(std_of_previous_races)
    races["driver_qualifying_std_last_5"] = by_driver["qualifying_position"].transform(std_of_previous_races)
    races["driver_dnf_rate_last_5"] = by_driver["did_not_finish"].transform(
        lambda s: average_of_previous_races(s.astype(float)))

    team_races = (
        races.groupby(["season", "round", "constructor"], as_index=False)["did_not_finish"].mean()
        .rename(columns={"did_not_finish": "team_dnf_share_this_race"})
        .sort_values(["season", "round"])
    )
    team_races["constructor_dnf_rate_last_5"] = (
        team_races.groupby("constructor")["team_dnf_share_this_race"].transform(average_of_previous_races)
    )
    return races.merge(
        team_races[["season", "round", "constructor", "constructor_dnf_rate_last_5"]],
        on=["season", "round", "constructor"], how="left",
    )


def fill_missing_history(races):
    """
    A debut race has no history, so the rolling features are NaN.
    Fill them with that race's qualifying position: a sensible pre-race guess.
    """
    for column in ["driver_avg_finish_last_5", "driver_avg_qualifying_last_5", "constructor_avg_finish_last_5"]:
        races[column] = races[column].fillna(races["qualifying_position"]).round(2)
    return races


def add_dnf_flag(races):
    """
    did_not_finish: True for any non-finish: retirement (engine, collision, ...),
    disqualification or did-not-start. "Finished", "+N Lap(s)" and "Lapped" count as finishing.
    POST-RACE information for the race itself: never a model feature directly. Only its
    PREVIOUS-race history is used (driver/constructor DNF rate), plus the error analysis.
    """
    status = races["status"].astype(str)
    finished = (status == "Finished") | status.str.startswith("+") | (status == "Lapped")
    races["did_not_finish"] = ~finished
    return races


def build_features(races):
    """Run every step in order and return the full modelling table."""
    races = add_driver_form(races)
    races = add_constructor_form(races)
    races = add_points_before_race(races)
    races = fill_missing_history(races)
    races = add_dnf_flag(races)
    races = add_consistency_and_reliability(races)
    return races.sort_values(["season", "round", "final_position"]).reset_index(drop=True)
