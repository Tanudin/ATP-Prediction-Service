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
# ## Exploratory Data Analysis
#

# %%
df.head()

# %% [markdown]
# ### Column Types
#

# %%
df.info()

# %% [markdown]
# ### Min-Max Values - Numerical Feature Columns
#

# %%
numeric_min_max = df.select_dtypes(include=[np.number]).agg(["min", "max"])
print(numeric_min_max)

# %% [markdown]
# ### Unique Values in Columns
#

# %%
# For each column, show unique values
for col in df.columns:
    unique_vals = df[col].unique()
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
total_unique_players = pd.concat([df["Player_1"], df["Player_2"]]).nunique()
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
match_stats(df)

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
series_stats(df)

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
court_stats(df)

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
surface_stats(df)

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
round_stats(df)

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
best_of_stats(df)

# %% [markdown]
# ## Data Preprocessing
#


# %%
def preprocess_data(df):
    # 1. base statistics - wins, losses, total matches
    wins = df["Winner"].value_counts()
    p1_counts = df["Player_1"].value_counts()
    p2_counts = df["Player_2"].value_counts()
    total_matches = p1_counts.add(p2_counts, fill_value=0)
    wins = wins.reindex(total_matches.index, fill_value=0)
    losses = total_matches - wins

    player_stats = pd.DataFrame(
        {
            "Total_Matches": total_matches,
            "Wins": wins,
            "Losses": losses,
            "Win_Pct": (wins / total_matches * 100).fillna(0),
        }
    )

    # 2. surface stats
    surface_wins = df.groupby(["Winner", "Surface"]).size().unstack(fill_value=0)
    surface_wins.columns = [f"Surface_Wins_{col}" for col in surface_wins.columns]
    player_stats = player_stats.join(surface_wins, how="left").fillna(0)
    surface_win_cols = [c for c in surface_wins.columns]
    surface_pcts = (
        player_stats[surface_win_cols].div(player_stats["Wins"], axis=0) * 100
    )
    surface_pcts.columns = [
        c.replace("Wins_", "SurfaceWinPct_") for c in surface_pcts.columns
    ]
    player_stats = pd.concat([player_stats, surface_pcts], axis=1).fillna(0)

    # 3. round stats
    round_wins = df.groupby(["Winner", "Round"]).size().unstack(fill_value=0)
    round_wins.columns = [f"Round_Wins_{col}" for col in round_wins.columns]
    player_stats = player_stats.join(round_wins, how="left").fillna(0)
    round_win_cols = [c for c in round_wins.columns]
    round_pcts = player_stats[round_win_cols].div(player_stats["Wins"], axis=0) * 100
    round_pcts.columns = [
        c.replace("Round_Wins_", "RoundWinPct_") for c in round_pcts.columns
    ]
    player_stats = pd.concat([player_stats, round_pcts], axis=1).fillna(0)

    # 4. series stats
    series_wins = df.groupby(["Winner", "Series"]).size().unstack(fill_value=0)
    series_wins.columns = [f"Series_Wins_{col}" for col in series_wins.columns]
    player_stats = player_stats.join(series_wins, how="left").fillna(0)
    series_win_cols = [c for c in series_wins.columns]
    series_pcts = player_stats[series_win_cols].div(player_stats["Wins"], axis=0) * 100
    series_pcts.columns = [
        c.replace("Series_Wins_", "SeriesWinPct_") for c in series_pcts.columns
    ]
    player_stats = pd.concat([player_stats, series_pcts], axis=1).fillna(0)

    # 5. court stats
    court_wins = df.groupby(["Winner", "Court"]).size().unstack(fill_value=0)
    court_wins.columns = [f"Court_Wins_{col}" for col in court_wins.columns]
    player_stats = player_stats.join(court_wins, how="left").fillna(0)
    court_win_cols = [c for c in court_wins.columns]
    court_pcts = player_stats[court_win_cols].div(player_stats["Wins"], axis=0) * 100
    court_pcts.columns = [
        c.replace("Court_Wins_", "CourtWinPct_") for c in court_pcts.columns
    ]
    player_stats = pd.concat([player_stats, court_pcts], axis=1).fillna(0)

    # 6. best of stats
    best_of_wins = df.groupby(["Winner", "Best of"]).size().unstack(fill_value=0)
    best_of_wins.columns = [f"BestOf_Wins_{int(col)}" for col in best_of_wins.columns]
    player_stats = player_stats.join(best_of_wins, how="left").fillna(0)

    best_of_win_cols = [c for c in best_of_wins.columns]
    best_of_pcts = (
        player_stats[best_of_win_cols].div(player_stats["Wins"], axis=0) * 100
    )
    best_of_pcts.columns = [
        c.replace("BestOf_Wins_", "BestOfWinPct_") for c in best_of_pcts.columns
    ]
    player_stats = pd.concat([player_stats, best_of_pcts], axis=1).fillna(0)

    # 7. sort by total wins
    player_stats = player_stats.sort_values(by="Wins", ascending=False)

    # 8. Reset index
    player_stats = player_stats.reset_index()
    player_stats = player_stats.rename(columns={"index": "Player_Name"})

    return player_stats


# %%
player_stats = preprocess_data(df)

# %%
player_stats.info()

# %%
player_stats.head()

# %%
result = preprocess_data(df)
print(result[result["Player_Name"] == "Nadal R."])

# %%
