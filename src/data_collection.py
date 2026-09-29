"""
DATA COLLECTION
===============
Downloads F1 race results + qualifying results (no telemetry) and saves
one tidy table to data/race_data.csv.

One row = one driver in one race.

Run from the project folder:
    python src/data_collection.py
The first run takes ~15–30 minutes (longer if the free API's rate limit is
hit — the script then waits and retries by itself). Everything is cached in
fastf1_cache/, so later runs take seconds.
"""

import time
from pathlib import Path

import pandas as pd
import fastf1
from fastf1.ergast import Ergast

FIRST_SEASON = 2018
LAST_SEASON = 2025

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "race_data.csv"
CACHE_DIR = PROJECT_ROOT / "fastf1_cache"


RATE_LIMIT_WAIT_SECONDS = 60
MAX_WAITS = 70   # the limit resets within an hour, so ~70 minutes is always enough


def ask_api(request_function, **arguments):
    """
    Call the API. If the server says "Too Many Requests" (the free API allows
    only a limited number of requests per hour), wait a minute and try again,
    instead of crashing. Anything already downloaded stays in the cache.
    """
    for attempt in range(1, MAX_WAITS + 1):
        try:
            return request_function(**arguments)
        except Exception as error:
            if "Too Many Requests" not in str(error):
                raise   # a different problem: show it
            print(f"      Rate limit reached. Waiting {RATE_LIMIT_WAIT_SECONDS} s "
                  f"(attempt {attempt}/{MAX_WAITS}) — this is normal, just leave it running...")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
    raise RuntimeError("Still rate-limited after waiting over an hour. Try again later.")


def get_completed_rounds(ergast, season):
    """Round numbers of all races in a season whose date has already passed."""
    schedule = pd.DataFrame(ask_api(ergast.get_race_schedule, season=season))
    today = pd.Timestamp.today().normalize()
    return schedule.loc[pd.to_datetime(schedule["raceDate"]) < today, "round"].tolist()


def get_one_race(ergast, season, round_number):
    """Race result + qualifying result for one Grand Prix, joined on driver."""
    race_response = ask_api(ergast.get_race_results, season=season, round=round_number)
    if len(race_response.content) == 0:
        return None

    race = race_response.content[0]                 # one row per driver
    race_info = race_response.description.iloc[0]   # race name, date, circuit

    race_table = pd.DataFrame({
        "season": season,
        "round": round_number,
        "race_name": race_info["raceName"],
        "race_date": race_info["raceDate"],
        "circuit": race_info["circuitName"],
        "driver_id": race["driverId"],
        "driver": race["givenName"] + " " + race["familyName"],
        "constructor": race["constructorName"],
        "grid_position": race["grid"],
        "final_position": race["position"],   # TARGET (known only after the race)
        "points": race["points"],             # post-race: used ONLY to build "points before race"
        "status": race["status"],             # post-race: used ONLY for error analysis
    })

    quali_response = ask_api(ergast.get_qualifying_results, season=season, round=round_number)
    if len(quali_response.content) > 0:
        quali = quali_response.content[0][["driverId", "position"]].rename(
            columns={"driverId": "driver_id", "position": "qualifying_position"})
        race_table = race_table.merge(quali, on="driver_id", how="left")
    else:
        race_table["qualifying_position"] = pd.NA

    return race_table


def clean_results(results):
    """Fix known quirks in the raw data."""
    results = results.copy()

    # Grid 0 = pit-lane start = the very back. Replace 0 with the number of cars in that race.
    cars_in_race = results.groupby(["season", "round"])["driver"].transform("count")
    results.loc[results["grid_position"] == 0, "grid_position"] = cars_in_race

    # No qualifying time recorded (rare) -> use grid position as the best substitute.
    results["qualifying_position"] = results["qualifying_position"].fillna(results["grid_position"])

    results = results.dropna(subset=["grid_position", "final_position", "qualifying_position"])
    for column in ["grid_position", "final_position", "qualifying_position"]:
        results[column] = results[column].astype(int)

    return results.sort_values(["season", "round", "final_position"]).reset_index(drop=True)


def collect_data(first_season=FIRST_SEASON, last_season=LAST_SEASON):
    """Download every completed race in the season range, clean, save, and return the table."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level("WARNING")

    ergast = Ergast(result_type="pandas", auto_cast=True)
    all_races = []

    for season in range(first_season, last_season + 1):
        rounds = get_completed_rounds(ergast, season)
        print(f"Season {season}: {len(rounds)} races")
        for round_number in rounds:
            try:
                race_table = get_one_race(ergast, season, round_number)
                if race_table is not None:
                    all_races.append(race_table)
                    print(f"   ✓ Round {round_number}: {race_table['race_name'].iloc[0]}")
            except Exception as error:
                print(f"   ✗ Round {round_number} skipped ({error})")
            time.sleep(1)   # go gently: the free API has rate limits

    if not all_races:
        raise RuntimeError("No data downloaded. Check your internet connection and try again.")

    results = clean_results(pd.concat(all_races, ignore_index=True))
    results.to_csv(DATA_FILE, index=False)
    print(f"\nSaved {len(results)} rows ({results.groupby(['season', 'round']).ngroups} races) to {DATA_FILE}")
    return results


if __name__ == "__main__":
    collect_data()
