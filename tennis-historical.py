# %% [markdown]
# ## Import Stuff
#

# %%
from pathlib import Path

import dotenv
import kagglehub
import numpy as np
import pandas as pd

# %% [markdown]
# ## Load Environment Variables
#

# %%
dotenv.load_dotenv()

# %% [markdown]
# ## Load and Read Data
#

# %%
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
print(f"Data Path -> {path}")
dataset_dir = Path(path)
data_file = dataset_dir / "atp_tennis.csv"
df = pd.read_csv(data_file)
print(
    f"data file name -> {data_file.name}\nnumber of matches from 2000-present-> {len(df)}"
)

# %% [markdown]
# ## Preprocess Data
#

# %%
# Convert to datetime objects
df["Date"] = pd.to_datetime(df["Date"])

# sort from most recent matches to earliest matches
df = df.sort_values(by="Date").reset_index(drop=True)

# # Sort by date to ensure historical features are computed correctly later
# df = df.sort_values("Date")

# %%
df.head()

# %% [markdown]
# ## Create Datasets
#

# %%
season_2025_start = pd.Timestamp("2024-12-27")
season_2025_end = pd.Timestamp("2025-11-16")

# %% [markdown]
# ### Training Data
#

# %%
train_df = df[df["Date"] < season_2025_start].reset_index(drop=True)
print(f"Training matches (2000-2024) -> {len(train_df)}")

# %%
train_df.head()

# %% [markdown]
# ### Testing Data
#

# %%
test_df = df[
    (df["Date"] >= season_2025_start) & (df["Date"] <= season_2025_end)
].reset_index(drop=True)
print(f"Testing matches - 2025 season -> {len(test_df)}")

# %%
test_df.head()

# %% [markdown]
# ## Exploratory Data Analysis
#

# %% [markdown]
# ### Column Types
#

# %%
train_df.info()

# %% [markdown]
# ### Min-Max Values - Numerical Feature Columns
#

# %%
numeric_min_max = train_df.select_dtypes(include=[np.number]).agg(["min", "max"])
print(numeric_min_max)

# %% [markdown]
# ### Unique Values in Columns
#

# %%
# For each column, show unique values
for col in train_df.columns:
    unique_vals = train_df[col].unique()
    n_unique = len(unique_vals)
    print(f"\n{col} -> {n_unique} unique values")

    # Only show actual values if there are < 20 unique values
    if n_unique < 20:
        print(f"  Values: {unique_vals}")
    else:
        print("(Too many to display - showing first 10)")
        print(f"  Sample: {unique_vals[:10]}")

# %% [markdown]
# ### Number of Unique Players
#

# %%
total_unique_players = pd.concat([df["Player_1"], train_df["Player_2"]]).nunique()
print(f"Total unique players in the dataset -> {total_unique_players}")

# %% [markdown]
# ### Total Match Stats
#


# %%
def match_stats(df):
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins
    player_stats = pd.DataFrame(
        {"Wins": wins, "Losses": losses, "Total Matches": total_matches}
    ).sort_values(by="Wins", ascending=False)
    print(f"len of list -> {len(player_stats)}")
    print(player_stats.head())


# %%
match_stats(train_df)

# %% [markdown]
# ### Series Stats
#


# %%
def series_stats(df):
    # 1. Overall Stats: Wins, Total Matches, and Losses
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)

    # Reindex to include players with 0 wins
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins

    # 2. Wins by Series
    # Unique series include 'Grand Slam', 'Masters 1000', 'ATP500', etc.
    series_wins = df.groupby(["Winner", "Series"]).size().unstack(fill_value=0)
    series_wins.columns = [f"Wins_{col}" for col in series_wins.columns]

    # 3. Create and Merge DataFrames
    player_stats = pd.DataFrame(
        {"Total Matches": total_matches, "Wins": wins, "Losses": losses}
    )

    # Join series wins to the main stats
    player_stats = player_stats.join(series_wins, how="left").fillna(0)

    # 4. Percentage Logic: Calculate series wins as a % of TOTAL wins
    win_cols = [c for c in player_stats.columns if c.startswith("Wins_")]

    # Vectorized division by total wins
    series_pcts = player_stats[win_cols].div(player_stats["Wins"], axis=0) * 100

    # Rename columns to reflect they are percentages of total wins
    series_pcts.columns = [
        c.replace("Wins_", "WinPct_Total_") for c in series_pcts.columns
    ]

    # Concatenate and handle division by zero
    player_stats = pd.concat([player_stats, series_pcts], axis=1).fillna(0)

    # Final sorting and display
    player_stats = player_stats.sort_values(by="Wins", ascending=False)

    print(f"Number of players analyzed -> {len(player_stats)}")
    print(player_stats.head())


# %%
series_stats(train_df)

# %% [markdown]
# ### Court Stats
#


# %%
def court_stats(df):
    # 1. Existing Logic: Overall Wins, Total Matches, and Losses
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)

    # Reindex to include players with 0 wins
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins

    # 2. New Logic: Wins by Court (Replacing Series/Surface)
    # Group by Winner and Court to get counts for 'Indoor' and 'Outdoor'
    court_wins = df.groupby(["Winner", "Court"]).size().unstack(fill_value=0)
    court_wins.columns = [f"Wins_{col}" for col in court_wins.columns]

    # 3. Create and Merge DataFrames
    player_stats = pd.DataFrame(
        {"Total Matches": total_matches, "Wins": wins, "Losses": losses}
    )

    # Join court wins to the main stats
    player_stats = player_stats.join(court_wins, how="left").fillna(0)

    # 4. Percentage Logic: Calculate court wins as a % of TOTAL wins
    win_cols = [c for c in player_stats.columns if c.startswith("Wins_")]

    # Vectorized division: Divide each court win column by the "Wins" column
    court_pcts = player_stats[win_cols].div(player_stats["Wins"], axis=0) * 100

    # Rename columns to reflect they are percentages of total career wins
    court_pcts.columns = [
        c.replace("Wins_", "WinPct_Total_") for c in court_pcts.columns
    ]

    # Concatenate and handle potential division by zero (for players with 0 wins)
    player_stats = pd.concat([player_stats, court_pcts], axis=1).fillna(0)

    # Final sorting and display
    player_stats = player_stats.sort_values(by="Wins", ascending=False)

    print(f"Number of players analyzed -> {len(player_stats)}")
    print(player_stats.head())

    # return player_stats


# %%
court_stats(train_df)

# %% [markdown]
# ### Surface Stats
#


# %%
def surface_stats(df):
    # 1. Existing Logic: Overall Wins, Total Matches, and Losses
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)

    # Reindex to include players with 0 wins
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins

    # 2. New Logic: Wins by Surface
    # Group by Winner and Surface, then count the occurrences
    surface_wins = df.groupby(["Winner", "Surface"]).size().unstack(fill_value=0)
    surface_wins.columns = [f"Wins_{col}" for col in surface_wins.columns]

    # 3. Create and Merge DataFrames
    player_stats = pd.DataFrame(
        {"Total Matches": total_matches, "Wins": wins, "Losses": losses}
    )

    # Join surface wins first so we have the columns needed for calculation
    player_stats = player_stats.join(surface_wins, how="left").fillna(0)

    # 4. NEW LOGIC: Calculate surface wins as a % of TOTAL wins
    # We select only the columns that start with "Wins_"
    win_cols = [c for c in player_stats.columns if c.startswith("Wins_")]

    # Vectorized division: Divide each surface win column by the "Wins" column
    # .div(..., axis=0) ensures it divides row-by-row
    surface_pcts = player_stats[win_cols].div(player_stats["Wins"], axis=0) * 100

    # Rename columns from "Wins_Clay" to "WinPct_Total_Clay" to clarify the denominator
    surface_pcts.columns = [
        c.replace("Wins_", "WinPct_Total_") for c in surface_pcts.columns
    ]

    # Concatenate the percentage columns and fill NaN (from 0/0 division) with 0
    player_stats = pd.concat([player_stats, surface_pcts], axis=1).fillna(0)

    # Final sorting and display
    player_stats = player_stats.sort_values(by="Wins", ascending=False)

    print(f"len of list -> {len(player_stats)}")
    print(player_stats.head())

    # return player_stats


# %%
surface_stats(train_df)

# %% [markdown]
# ### Round Stats
#


# %%
def round_stats(df):
    # 1. Existing Logic: Overall Wins, Total Matches, and Losses
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)

    # Reindex to include players with 0 wins
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins

    # 2. New Logic: Wins by Round (Replacing Surface)
    # Group by Winner and Round, then count the occurrences
    # Unique rounds in your data include '1st Round', 'Quarterfinals', 'The Final', etc.
    round_wins = df.groupby(["Winner", "Round"]).size().unstack(fill_value=0)
    round_wins.columns = [f"Wins_{col}" for col in round_wins.columns]

    # 3. Create and Merge DataFrames
    player_stats = pd.DataFrame(
        {"Total Matches": total_matches, "Wins": wins, "Losses": losses}
    )

    # Join round wins to the main stats
    player_stats = player_stats.join(round_wins, how="left").fillna(0)

    # 4. Percentage Logic: Calculate round wins as a % of TOTAL wins
    # Select columns that start with "Wins_" (which are now round names)
    win_cols = [c for c in player_stats.columns if c.startswith("Wins_")]

    # Vectorized division: Divide each round win column by the "Wins" column
    # Use .div(..., axis=0) to handle row-wise division
    round_pcts = player_stats[win_cols].div(player_stats["Wins"], axis=0) * 100

    # Rename columns to clarify the denominator (Total Wins)
    round_pcts.columns = [
        c.replace("Wins_", "WinPct_Total_") for c in round_pcts.columns
    ]

    # Concatenate the percentage columns and fill NaN (from 0/0 division) with 0
    player_stats = pd.concat([player_stats, round_pcts], axis=1).fillna(0)

    # Final sorting and display
    player_stats = player_stats.sort_values(by="Wins", ascending=False)

    print(f"Number of players -> {len(player_stats)}")
    print(player_stats.head())


# %%
round_stats(train_df)

# %% [markdown]
# ### Best of Stats
#


# %%
def best_of_stats(df):
    # 1. Overall Stats: Wins, Total Matches, and Losses
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)

    # Reindex to include players with 0 wins
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins

    # 2. Wins by Match Format (Best of 3 vs 5)
    # Group by Winner and the 'Best of' column
    best_of_wins = df.groupby(["Winner", "Best of"]).size().unstack(fill_value=0)

    # Rename columns to 'Wins_3' and 'Wins_5'
    best_of_wins.columns = [f"Wins_{col}" for col in best_of_wins.columns]

    # 3. Create and Merge DataFrames
    player_stats = pd.DataFrame(
        {"Total Matches": total_matches, "Wins": wins, "Losses": losses}
    )

    # Join match format wins to the main stats
    player_stats = player_stats.join(best_of_wins, how="left").fillna(0)

    # 4. Percentage Logic: Calculate format wins as a % of TOTAL wins
    win_cols = [c for c in player_stats.columns if c.startswith("Wins_")]

    # Vectorized division by total wins
    format_pcts = player_stats[win_cols].div(player_stats["Wins"], axis=0) * 100

    # Rename columns to reflect they are percentages of total career wins
    format_pcts.columns = [
        c.replace("Wins_", "WinPct_Total_") for c in format_pcts.columns
    ]

    # Concatenate and handle potential division by zero
    player_stats = pd.concat([player_stats, format_pcts], axis=1).fillna(0)

    # Final sorting and display
    player_stats = player_stats.sort_values(by="Wins", ascending=False)

    print(f"Number of players analyzed: {len(player_stats)}")
    print(player_stats.head())


# %%
best_of_stats(train_df)

# %% [markdown]
# ## Data Preprocessing
#

# %% [markdown]
# ### Player Statistics
#

# %%
# def preprocess_player_stats(df):
#     # 1. base statistics - wins, losses, total matches
#     wins = df["Winner"].value_counts()
#     p1_counts = df["Player_1"].value_counts()
#     p2_counts = df["Player_2"].value_counts()
#     total_matches = p1_counts.add(p2_counts, fill_value=0)
#     wins = wins.reindex(total_matches.index, fill_value=0)
#     losses = total_matches - wins

#     player_stats = pd.DataFrame(
#         {
#             "Total_Matches": total_matches,
#             "Wins": wins,
#             "Losses": losses,
#             "Win_Pct": (wins / total_matches * 100).fillna(0),
#         }
#     )

#     # 2. surface stats
#     surface_wins = df.groupby(["Winner", "Surface"]).size().unstack(fill_value=0)
#     surface_wins.columns = [f"Surface_Wins_{col}" for col in surface_wins.columns]
#     player_stats = player_stats.join(surface_wins, how="left").fillna(0)
#     surface_win_cols = [c for c in surface_wins.columns]
#     surface_pcts = (
#         player_stats[surface_win_cols].div(player_stats["Wins"], axis=0) * 100
#     )
#     surface_pcts.columns = [
#         c.replace("Wins_", "SurfaceWinPct_") for c in surface_pcts.columns
#     ]
#     player_stats = pd.concat([player_stats, surface_pcts], axis=1).fillna(0)

#     # 3. round stats
#     round_wins = df.groupby(["Winner", "Round"]).size().unstack(fill_value=0)
#     round_wins.columns = [f"Round_Wins_{col}" for col in round_wins.columns]
#     player_stats = player_stats.join(round_wins, how="left").fillna(0)
#     round_win_cols = [c for c in round_wins.columns]
#     round_pcts = player_stats[round_win_cols].div(player_stats["Wins"], axis=0) * 100
#     round_pcts.columns = [
#         c.replace("Round_Wins_", "RoundWinPct_") for c in round_pcts.columns
#     ]
#     player_stats = pd.concat([player_stats, round_pcts], axis=1).fillna(0)

#     # 4. series stats
#     series_wins = df.groupby(["Winner", "Series"]).size().unstack(fill_value=0)
#     series_wins.columns = [f"Series_Wins_{col}" for col in series_wins.columns]
#     player_stats = player_stats.join(series_wins, how="left").fillna(0)
#     series_win_cols = [c for c in series_wins.columns]
#     series_pcts = player_stats[series_win_cols].div(player_stats["Wins"], axis=0) * 100
#     series_pcts.columns = [
#         c.replace("Series_Wins_", "SeriesWinPct_") for c in series_pcts.columns
#     ]
#     player_stats = pd.concat([player_stats, series_pcts], axis=1).fillna(0)

#     # 5. court stats
#     court_wins = df.groupby(["Winner", "Court"]).size().unstack(fill_value=0)
#     court_wins.columns = [f"Court_Wins_{col}" for col in court_wins.columns]
#     player_stats = player_stats.join(court_wins, how="left").fillna(0)
#     court_win_cols = [c for c in court_wins.columns]
#     court_pcts = player_stats[court_win_cols].div(player_stats["Wins"], axis=0) * 100
#     court_pcts.columns = [
#         c.replace("Court_Wins_", "CourtWinPct_") for c in court_pcts.columns
#     ]
#     player_stats = pd.concat([player_stats, court_pcts], axis=1).fillna(0)

#     # 6. best of stats
#     best_of_wins = df.groupby(["Winner", "Best of"]).size().unstack(fill_value=0)
#     best_of_wins.columns = [f"BestOf_Wins_{int(col)}" for col in best_of_wins.columns]
#     player_stats = player_stats.join(best_of_wins, how="left").fillna(0)

#     best_of_win_cols = [c for c in best_of_wins.columns]
#     best_of_pcts = (
#         player_stats[best_of_win_cols].div(player_stats["Wins"], axis=0) * 100
#     )
#     best_of_pcts.columns = [
#         c.replace("BestOf_Wins_", "BestOfWinPct_") for c in best_of_pcts.columns
#     ]
#     player_stats = pd.concat([player_stats, best_of_pcts], axis=1).fillna(0)

#     # 7. sort by total wins
#     player_stats = player_stats.sort_values(by="Wins", ascending=False)

#     # 8. Reset index
#     player_stats = player_stats.reset_index()
#     player_stats = player_stats.rename(columns={"index": "Player_Name"})

#     return player_stats

# %%
# player_stats = preprocess_player_stats(train_df)

# %%
# player_stats.info()

# %%
# player_stats.head()

# %% [markdown]
# #### Check Preprocess Player Stats
#

# %%
# print(player_stats[player_stats["Player_Name"] == "Alcaraz C."])


# %%
# print(player_stats[player_stats["Player_Name"] == "Sinner J."])


# %% [markdown]
# #### Find Carlos Alcaraz First Match
#

# %%
# # Filter for any match where Alcaraz played as Player 1 or Player 2
# alcaraz_matches = df[
#     (df["Player_1"] == "Alcaraz C.") | (df["Player_2"] == "Alcaraz C.")
# ]
# alcaraz_first_match = alcaraz_matches.iloc[[0]]
# alcaraz_first_match


# %% [markdown]
# ### Match Statistics
#

# %%
# def init_player_match_history():
#     return {
#         "Total_Matches": 0,
#         "Wins": 0,
#         "Losses": 0,
#         # Surface wins
#         "Wins_Hard": 0,
#         "Wins_Clay": 0,
#         "Wins_Grass": 0,
#         "Wins_Carpet": 0,
#         # Round wins
#         "Wins_1st Round": 0,
#         "Wins_2nd Round": 0,
#         "Wins_3rd Round": 0,
#         "Wins_4th Round": 0,
#         "Wins_Quarterfinals": 0,
#         "Wins_Semifinals": 0,
#         "Wins_The Final": 0,
#         "Wins_Round Robin": 0,
#         # Series wins
#         "Wins_ATP250": 0,
#         "Wins_ATP500": 0,
#         "Wins_Grand Slam": 0,
#         "Wins_Masters": 0,
#         "Wins_Masters 1000": 0,
#         "Wins_Masters Cup": 0,
#         "Wins_International": 0,
#         "Wins_International Gold": 0,
#         # Court wins
#         "Wins_Indoor": 0,
#         "Wins_Outdoor": 0,
#         # Best of wins
#         "Wins_3": 0,
#         "Wins_5": 0,
#     }


# def compute_player_match_history(df):
#     print(f"Processing {len(df)} matches chronologically...")

#     # 1. Sort chronologically
#     df = df.sort_values("Date").reset_index(drop=True)

#     # 2. Initialize player history
#     match_history = {}

#     history_rows = []

#     # Calculate median rank for missing value replacement
#     valid_ranks_1 = df[df["Rank_1"] > 0]["Rank_1"]
#     valid_ranks_2 = df[df["Rank_2"] > 0]["Rank_2"]
#     median_rank = pd.concat([valid_ranks_1, valid_ranks_2]).median()

#     print(f"  Median rank for missing values: {median_rank:.0f}")

#     # 3. Iterate through matches chronologically
#     for idx, row in df.iterrows():
#         if idx % 5000 == 0:
#             print(f"  Processed {idx}/{len(df)} matches...")

#         p1 = row["Player_1"]
#         p2 = row["Player_2"]
#         winner = row["Winner"]

#         if p1 not in match_history:
#             match_history[p1] = init_player_match_history()
#         if p2 not in match_history:
#             match_history[p2] = init_player_match_history()

#         # Handle missing values for this match
#         rank_1 = median_rank if row["Rank_1"] == -1 else row["Rank_1"]
#         rank_2 = median_rank if row["Rank_2"] == -1 else row["Rank_2"]
#         pts_1 = 0 if row["Pts_1"] == -1 else row["Pts_1"]
#         pts_2 = 0 if row["Pts_2"] == -1 else row["Pts_2"]
#         odd_1 = np.nan if row.get("Odd_1", -1) == -1 else row.get("Odd_1", -1)
#         odd_2 = np.nan if row.get("Odd_2", -1) == -1 else row.get("Odd_2", -1)

#         match_info = {
#             # Match identification
#             "Date": row["Date"],
#             "Tournament": row["Tournament"],
#             "Player_1": p1,
#             "Player_2": p2,
#             "Winner": winner,
#             # Match context (for categorical encoding later)
#             "Surface": row["Surface"],
#             "Series": row["Series"],
#             "Round": row["Round"],
#             "Court": row["Court"],
#             "Best_of": row["Best of"],
#             # Rankings (with missing value handling)
#             "Rank_1": rank_1,
#             "Rank_2": rank_2,
#             "Rank_Diff": rank_1 - rank_2,
#             # Points (with missing value handling)
#             "Pts_1": pts_1,
#             "Pts_2": pts_2,
#             "Pts_Diff": pts_1 - pts_2,
#             # Odds (with missing value handling - use NaN)
#             "Odd_1": odd_1,
#             "Odd_2": odd_2,
#             "Odds_Diff": odd_1 - odd_2 if not pd.isna(odd_1) else np.nan,
#         }

#         # Add Player 1's historical stats (before this match)
#         for stat, val in match_history[p1].items():
#             match_info[f"P1_{stat}"] = val

#         # Add Player 2's historical stats (before this match)
#         for stat, val in match_history[p2].items():
#             match_info[f"P2_{stat}"] = val

#         # Create target variable
#         match_info["Player_1_Won"] = 1 if winner == p1 else 0

#         history_rows.append(match_info)

#         # Both players played a match
#         match_history[p1]["Total_Matches"] += 1
#         match_history[p2]["Total_Matches"] += 1

#         # Update winner's stats
#         if winner == p1:
#             match_history[p1]["Wins"] += 1
#             match_history[p2]["Losses"] += 1
#             winner_state = match_history[p1]
#         else:
#             match_history[p2]["Wins"] += 1
#             match_history[p1]["Losses"] += 1
#             winner_state = match_history[p2]

#         # Update winner's context-specific wins
#         surface_key = f"Wins_{row['Surface']}"
#         round_key = f"Wins_{row['Round']}"
#         series_key = f"Wins_{row['Series']}"
#         court_key = f"Wins_{row['Court']}"
#         bestof_key = f"Wins_{row['Best of']}"

#         # Use .get() to safely handle any unexpected categories
#         winner_state[surface_key] = winner_state.get(surface_key, 0) + 1
#         winner_state[round_key] = winner_state.get(round_key, 0) + 1
#         winner_state[series_key] = winner_state.get(series_key, 0) + 1
#         winner_state[court_key] = winner_state.get(court_key, 0) + 1
#         winner_state[bestof_key] = winner_state.get(bestof_key, 0) + 1

#     print(f"Processed all {len(df)} matches")

#     # 4. Convert to DataFrame
#     atp_match_history = pd.DataFrame(history_rows)

#     print(f"match history shape -> {atp_match_history.shape}")

#     return atp_match_history


# %%
# complete_match_history = compute_player_match_history(train_df)

# %%
# complete_match_history.info()

# %%
# complete_match_history.head()

# %%
# print(f"length of train_features() -> {len(complete_match_history)}")

# %% [markdown]
# #### Check Alcaraz Complete Match History - First Match
#

# %%
# alcaraz_debut = complete_match_history[
#     complete_match_history["Player_1"] == "Alcaraz C."
# ].iloc[0]

# alcaraz_debut

# %% [markdown]
# ### Calculate Player Match History Percentages
#

# %%
# def compute_match_percentages(df):
#     """
#     Convert raw win counts to percentages
#     """
#     df1 = df.copy()

#     # Overall win percentage for each player
#     df1["P1_Win_Pct"] = np.where(
#         df1["P1_Total_Matches"] > 0,
#         df1["P1_Wins"] / df1["P1_Total_Matches"],
#         0.5,  # Default for players with no history
#     )
#     df1["P2_Win_Pct"] = np.where(
#         df1["P2_Total_Matches"] > 0, df1["P2_Wins"] / df1["P2_Total_Matches"], 0.5
#     )

#     # Surface-specific win percentages
#     for surface in ["Hard", "Clay", "Grass", "Carpet"]:
#         df1[f"P1_WinPct_{surface}"] = np.where(
#             df1["P1_Total_Matches"] > 0,
#             df1[f"P1_Wins_{surface}"] / df1["P1_Total_Matches"],
#             0,
#         )
#         df1[f"P2_WinPct_{surface}"] = np.where(
#             df1["P2_Total_Matches"] > 0,
#             df1[f"P2_Wins_{surface}"] / df1["P2_Total_Matches"],
#             0,
#         )

#     # Series-specific win percentages
#     for series in ["ATP250", "ATP500", "Grand Slam", "Masters 1000"]:
#         col_name = series.replace(" ", "_")  # Handle spaces
#         df1[f"P1_WinPct_{col_name}"] = np.where(
#             df1["P1_Total_Matches"] > 0,
#             df1[f"P1_Wins_{series}"] / df1["P1_Total_Matches"],
#             0,
#         )
#         df1[f"P2_WinPct_{col_name}"] = np.where(
#             df1["P2_Total_Matches"] > 0,
#             df1[f"P2_Wins_{series}"] / df1["P2_Total_Matches"],
#             0,
#         )

#     # Court-specific win percentages
#     for court in ["Indoor", "Outdoor"]:
#         df1[f"P1_WinPct_{court}"] = np.where(
#             df1["P1_Total_Matches"] > 0,
#             df1[f"P1_Wins_{court}"] / df1["P1_Total_Matches"],
#             0,
#         )
#         df1[f"P2_WinPct_{court}"] = np.where(
#             df1["P2_Total_Matches"] > 0,
#             df1[f"P2_Wins_{court}"] / df1["P2_Total_Matches"],
#             0,
#         )

#     return df1

# %%
# complete_match_history_percentages = compute_match_percentages(complete_match_history)

# %%
# complete_match_history_percentages.info()

# %%
# complete_match_history_percentages.head(10)

# %% [markdown]
# ### Categorical Feature Encoding
#

# %%
# from sklearn.preprocessing import LabelEncoder


# def encode_categorical_features(df):
#     """
#     Label encode categorical features
#     """
#     df1 = df.copy()

#     # 1. Tournament encoding (keep frequent, group rare as 'Other')
#     threshold = 0.01
#     tourney_counts = df1["Tournament"].value_counts(normalize=True)
#     df1["Tournament_Clean"] = df1["Tournament"].apply(
#         lambda x: x if tourney_counts.get(x, 0) > threshold else "Other"
#     )

#     le_tournament = LabelEncoder()
#     df1["Tournament_Encoded"] = le_tournament.fit_transform(df1["Tournament_Clean"])

#     # 2. Surface encoding
#     le_surface = LabelEncoder()
#     df1["Surface_Encoded"] = le_surface.fit_transform(df1["Surface"])

#     # 3. Series encoding
#     le_series = LabelEncoder()
#     df1["Series_Encoded"] = le_series.fit_transform(df1["Series"])

#     # 4. Round encoding
#     le_round = LabelEncoder()
#     df1["Round_Encoded"] = le_round.fit_transform(df1["Round"])

#     # 5. Court encoding
#     le_court = LabelEncoder()
#     df1["Court_Encoded"] = le_court.fit_transform(df1["Court"])

#     return df1


# %%
# complete_match_history_encoded = encode_categorical_features(
#     complete_match_history_percentages
# )

# %%
# complete_match_history_encoded.info()

# %%
# complete_match_history_encoded.head(10)

# %%
# print(f"length of categorical encoding data -> {len(complete_match_history_encoded)}")

# %% [markdown]
# ### Derived Features
#

# %%
# def compute_derived_features(df):
#     df1 = df.copy()

#     # 1. Win percentage differential (who has better record?)
#     df1["Win_Pct_Diff"] = df1["P1_Win_Pct"] - df1["P2_Win_Pct"]

#     # 2. Experience differential (who has played more?)
#     df1["Experience_Diff"] = df1["P1_Total_Matches"] - df1["P2_Total_Matches"]

#     # 3. If playing on Hard, use Hard win % difference
#     df1["Surface_Advantage"] = 0.0

#     df1.loc[df1["Surface"] == "Hard", "Surface_Advantage"] = (
#         df1["P1_WinPct_Hard"] - df1["P2_WinPct_Hard"]
#     )
#     df1.loc[df1["Surface"] == "Clay", "Surface_Advantage"] = (
#         df1["P1_WinPct_Clay"] - df1["P2_WinPct_Clay"]
#     )
#     df1.loc[df1["Surface"] == "Grass", "Surface_Advantage"] = (
#         df1["P1_WinPct_Grass"] - df1["P2_WinPct_Grass"]
#     )
#     df1.loc[df1["Surface"] == "Carpet", "Surface_Advantage"] = (
#         df1["P1_WinPct_Carpet"] - df1["P2_WinPct_Carpet"]
#     )

#     # 4. Series advantage (Grand Slam performance difference)
#     df1.loc[df1["Series"] == "Grand Slam", "Series_Advantage"] = (
#         df1["P1_WinPct_Grand_Slam"] - df1["P2_WinPct_Grand_Slam"]
#     )
#     df1.loc[df1["Series"] == "Masters 1000", "Series_Advantage"] = (
#         df1["P1_WinPct_Masters_1000"] - df1["P2_WinPct_Masters_1000"]
#     )
#     # Add default 0 for other series
#     df1["Series_Advantage"] = df1["Series_Advantage"].fillna(0)

#     # 5. Court advantage
#     df1.loc[df1["Court"] == "Indoor", "Court_Advantage"] = (
#         df1["P1_WinPct_Indoor"] - df1["P2_WinPct_Indoor"]
#     )
#     df1.loc[df1["Court"] == "Outdoor", "Court_Advantage"] = (
#         df1["P1_WinPct_Outdoor"] - df1["P2_WinPct_Outdoor"]
#     )
#     df1["Court_Advantage"] = df1["Court_Advantage"].fillna(0)

#     return df1

# %%
# complete_match_history_derived_features = compute_derived_features(
#     complete_match_history_encoded
# )

# %%
# complete_match_history_derived_features.info()
# print(f"{len(complete_match_history_derived_features)}")

# %% [markdown]
# ### Compute Symmetric Player Statistics
#

# %%
# def create_symmetric_dataset(df):
#     """
#     Create symmetric dataset where each match appears twice:
#     once from Player_1's perspective, once from Player_2's perspective.

#     Example:
#         Original: Federer vs Nadal (Nadal wins)
#         Creates:
#         - Row 1: P1=Federer, P2=Nadal, P1_Won=0
#         - Row 2: P1=Nadal, P2=Federer, P1_Won=1
#     """
#     print("\n=== Creating Symmetric Dataset ===")
#     print(f"Original matches: {len(df):,}")

#     df_p1_info = df.copy()

#     df_p2_info = df.copy()

#     # Swap basic match info
#     df_p2_info["Player_1"] = df["Player_2"]
#     df_p2_info["Player_2"] = df["Player_1"]

#     # Swap rankings (and invert the differential)
#     df_p2_info["Rank_1"] = df["Rank_2"]
#     df_p2_info["Rank_2"] = df["Rank_1"]
#     df_p2_info["Rank_Diff"] = -df["Rank_Diff"]

#     # Swap points (and invert the differential)
#     df_p2_info["Pts_1"] = df["Pts_2"]
#     df_p2_info["Pts_2"] = df["Pts_1"]
#     df_p2_info["Pts_Diff"] = -df["Pts_Diff"]

#     # Swap odds (and invert the differential)
#     df_p2_info["Odd_1"] = df["Odd_2"]
#     df_p2_info["Odd_2"] = df["Odd_1"]
#     df_p2_info["Odds_Diff"] = -df["Odds_Diff"]

#     # Swap ALL Player_1 and Player_2 statistics
#     # Get all P1_* and P2_* columns
#     p1_cols = [col for col in df.columns if col.startswith("P1_")]
#     p2_cols = [col for col in df.columns if col.startswith("P2_")]

#     print(f"Swapping {len(p1_cols)} P1_* columns with {len(p2_cols)} P2_* columns")

#     # Swap the columns
#     for p1_col, p2_col in zip(p1_cols, p2_cols):
#         df_p2_info[p1_col] = df[p2_col]
#         df_p2_info[p2_col] = df[p1_col]

#     diff_features = [
#         "Win_Pct_Diff",
#         "Experience_Diff",
#         "Surface_Advantage",
#         "Court_Advantage",
#     ]

#     # Add Series_Advantage if it exists
#     if "Series_Advantage" in df.columns:
#         diff_features.append("Series_Advantage")

#     for feat in diff_features:
#         if feat in df.columns:
#             df_p2_info[feat] = -df[feat]

#     df_p2_info["Player_1_Won"] = 1 - df["Player_1_Won"]

#     symmetric_df = pd.concat([df_p1_info, df_p2_info], ignore_index=True)

#     # Sort by date to maintain chronological order
#     symmetric_df = symmetric_df.sort_values("Date").reset_index(drop=True)

#     return symmetric_df


# %%
# complete_match_history_symmetric = create_symmetric_dataset(
#     complete_match_history_derived_features
# )


# %%
# complete_match_history_symmetric.info()
# print(f"Total training samples: {len(complete_match_history_symmetric):,}")


# %%
# complete_match_history_symmetric.head(10)

# %% [markdown]
# ## Complete Data Preprocessing
#

# %%
from utils import (
    compute_derived_features,
    compute_match_percentages,
    compute_player_match_history,
    create_symmetric_dataset,
    encode_categorical_features,
)


def preprocess_data(df):
    print("Preprocess all historical data")
    complete_match_history = compute_player_match_history(df)
    complete_match_history_percentages = compute_match_percentages(
        complete_match_history
    )
    complete_encoded_match_history = encode_categorical_features(
        complete_match_history_percentages
    )
    complete_match_history_derived = compute_derived_features(
        complete_encoded_match_history
    )
    final_df = create_symmetric_dataset(complete_match_history_derived)

    return final_df


# %%
clean_data = preprocess_data(df)

# %%
clean_data.columns

# %%
clean_data.head(10)

# %% [markdown]
# ## Final Training Data
#

# %%
from utils import final_train_data

final_train_df = final_train_data(clean_data)
final_train_df["Date"] = final_train_df["Date"].dt.strftime("%Y-%m-%d")

# %%
pk_cols = ["Date", "Player_1", "Player_2"]
duplicate_count = final_train_df.duplicated(subset=pk_cols).sum()
if duplicate_count > 0:
    print(f"Removing {duplicate_count} duplicates...")
    final_train_df = final_train_df.drop_duplicates(subset=pk_cols, keep="first")
    print(f"✓ Final rows: {len(final_train_df):,}")


# %% [markdown]
# ## Hopsworks Feature Store Setup
#

# %%
import hopsworks

project = hopsworks.login(project="ATP_Tennis_Prediction")
fs = project.get_feature_store()
tennis_matches_fg = fs.get_or_create_feature_group(
    # Name of your feature group (what you'll reference later)
    name="tennis_matches",
    # Description of what's inside
    description="ATP tennis match data (2000-2025) with player statistics, encoded categories, win percentages, and symmetric perspectives for match outcome prediction",
    # Version (start with 1, increment if you change features)
    version=2,
    primary_key=["Date", "Player_1", "Player_2"],
    event_time="timestamp",
    # Enable for real-time predictions
    online_enabled=True,
)

# %%
tennis_matches_fg.insert(final_train_df)
print(f"Uploaded {len(final_train_df):,} matches to Hopsworks!")

# %%
