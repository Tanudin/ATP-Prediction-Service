# %% [markdown]
# ## Load Environment Variables
#

# %%
import dotenv

dotenv.load_dotenv()


# %% [markdown]
# ## Import Stuff
#

# %%
from datetime import datetime
from pathlib import Path

import hopsworks
import kagglehub
import pandas as pd

# %% [markdown]
# ## Connect to Hopsworks and Load Recent Data
#

# %%
project = hopsworks.login(project="ATP_Tennis_Prediction")
fs = project.get_feature_store()

# %%
tennis_matches_fg = fs.get_feature_group(
    name="tennis_matches",
    version=2,
)

# %%
existing_df = tennis_matches_fg.read()

# %%
if "Date" in existing_df.columns:
    existing_df["Date"] = pd.to_datetime(existing_df["Date"])
    date_col = "Date"
elif "date" in existing_df.columns:
    existing_df["date"] = pd.to_datetime(existing_df["date"])
    date_col = "date"
else:
    raise ValueError("Cannot find Date or date column in feature store!")

# %%
latest_date_in_fs = existing_df[date_col].max()
print(f"latest date -> {latest_date_in_fs}")

# %% [markdown]
# ## Get Latest Data
#

# %%
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
print(f"Data Path -> {path}")
dataset_dir = Path(path)
data_file = dataset_dir / "atp_tennis.csv"

# %%
df_latest = pd.read_csv(data_file)
df_latest["Date"] = pd.to_datetime(df_latest["Date"])
df_latest = df_latest.sort_values(by="Date").reset_index(drop=True)

# %%
print(f"latest dataset length -> {len(df_latest):,}")
print(f"Date Range -> {df_latest['Date'].min()} to {df_latest['Date'].max()}")

# %%
df_latest.head()

# %% [markdown]
# ## Write Latest Data
#

# %%
df_new = df_latest[df_latest["Date"] > latest_date_in_fs].copy()

# %%
if len(df_new) == 0:
    print("No new matches to process. Feature store is up to date!")
    print(f"latest match in fs -> {latest_date_in_fs}")
else:
    print(
        f"Date range of new matches: {df_new['Date'].min()} to {df_new['Date'].max()}"
    )
    print("First few new matches")
    print(df_new[["Date", "Tournament", "Player_1", "Player_2", "Winner"]].head(10))


# %%
from utils import (
    compute_derived_features,
    compute_match_percentages,
    compute_player_match_history,
    create_symmetric_dataset,
    encode_categorical_features,
    final_train_data,
)

if len(df_new) > 0:
    print("Compute new match history")
    new_match_history = compute_player_match_history(df_latest)
    print("Compute new match percentages")
    new_match_percentages = compute_match_percentages(new_match_history)
    print("Encode new match info")
    new_encoded = encode_categorical_features(new_match_percentages)
    print("Derive features from new matches ")
    new_derived = compute_derived_features(new_encoded)
    print("Create new symmetric match data")
    new_symmetric = create_symmetric_dataset(new_derived)
    print("Final Updated Data")
    new_final = final_train_data(new_symmetric)
    new_final["timestamp"] = datetime.now()
    latest_date_str = latest_date_in_fs.strftime("%Y-%m-%d")
    new_final["Date"] = new_final["Date"].dt.strftime("%Y-%m-%d")
    new_final = new_final[new_final["Date"] > latest_date_str].copy()
    print(f"Processed -> {len(new_final):,} rows (symmetric)")
    print(f"Columns -> {len(new_final.columns)}")


# %% [markdown]
# ## Insert into Hopsworks
#

# %%
if len(df_new) > 0:
    tennis_matches_fg.insert(new_final)
