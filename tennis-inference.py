# %% [markdown]
# ## Environment Setup

# %%
import dotenv

dotenv.load_dotenv()


# %% [markdown]
# ## Imports Stuff

# %%
import hopsworks
import pandas as pd
from xgboost import XGBClassifier

# %% [markdown]
# ## Connect to Hopsworks and Load Data

# %%
project = hopsworks.login(project="ATP_Tennis_Prediction")
fs = project.get_feature_store()

feature_view = fs.get_feature_view(
    name="tennis_match_prediction",
    version=1,
)

# %%
mr = project.get_model_registry()

retrieved_model = mr.get_model(
    name="tennis_match_predictor",  # ← Your model name
    version=1,
)

saved_model_dir = retrieved_model.download()

model = XGBClassifier()
model.load_model(saved_model_dir + "/model.json")


# %% [markdown]
# ### Holdout
# currently no new matches happening yet until January

# %%
# Get ALL matches from main feature group
tennis_fg = fs.get_feature_group(
    name="tennis_matches",
    version=2,
)

all_matches = tennis_fg.read()

# Filter for "upcoming" (pretend 2025 hasn't happened)
all_matches["date"] = pd.to_datetime(all_matches["date"])
df_new = all_matches[all_matches["date"].dt.year == 2025].copy()


# %% [markdown]
# ### Upcoming Matches

# %%
# upcoming_fg = fs.get_feature_group(
#     name="tennis_upcoming_matches",
#     version=1,
# )

# df_new = upcoming_fg.read()


# %% [markdown]
# ## Update Model

# %%
drop_list = [
    # Metadata
    "date",
    "player_1",
    "player_2",
    "winner",
    "timestamp",
    # Categorical text (have encoded versions)
    "tournament",
    "surface",
    "series",
    "round",
    "court",
    "tournament_clean",
]

# Also drop target (we're predicting it!)
drop_list.append("player_1_won")

# Filter for columns that exist
cols_to_drop = [col for col in drop_list if col in df_new.columns]

# Create features
features = df_new.drop(cols_to_drop, axis=1)

print(f"Features shape: {features.shape}")
print(f"Features columns: {features.columns.tolist()}")

# Verify no missing values
if features.isnull().sum().sum() > 0:
    print("Missing values found!")
    features = features.fillna(0)


# %% [markdown]
# ## Prediction

# %%
# Predict match outcomes
predictions = model.predict(features)

# Get probabilities
prediction_probabilities = model.predict_proba(features)
prob_player1_wins = prediction_probabilities[:, 1]  # P(Player 1 wins)

print(f"Predictions made: {len(predictions)}")
print(f"Player 1 wins predicted: {predictions.sum()} ({predictions.mean():.1%})")


# %%
# Create results dataframe with original match info
# Only include columns that exist in df_new
available_cols = ["date", "player_1", "player_2"]
# Add optional columns if they exist
for col in ["tournament", "surface", "round", "odd_1", "odd_2"]:
    if col in df_new.columns:
        available_cols.append(col)

results = df_new[available_cols].copy()

# Add predictions
results["predicted_player_1_wins"] = predictions
results["player_1_win_probability"] = prob_player1_wins

# Determine predicted winner
results["predicted_winner"] = results.apply(
    lambda row: row["player_1"]
    if row["predicted_player_1_wins"] == 1
    else row["player_2"],
    axis=1,
)

# Calculate confidence (distance from 50%)
results["confidence"] = [x if x > 0.5 else 1 - x for x in prob_player1_wins]

# Add actual results if available
if "player_1_won" in df_new.columns:
    results["actual_player_1_won"] = df_new["player_1_won"].values
    results["actual_winner"] = df_new.apply(
        lambda row: row["player_1"] if row["player_1_won"] == 1 else row["player_2"],
        axis=1,
    )
    results["correct"] = results["predicted_winner"] == results["actual_winner"]
    
    # Add odds for betting calculation
    if "odd_1" in df_new.columns and "odd_2" in df_new.columns:
        results["odd_1"] = df_new["odd_1"].values
        results["odd_2"] = df_new["odd_2"].values
        
        # Determine which odds to use based on our prediction
        results["bet_odds"] = results.apply(
            lambda row: row["odd_1"] if row["predicted_player_1_wins"] == 1 else row["odd_2"],
            axis=1
        )

    # Calculate accuracy
    accuracy = results["correct"].mean()
    print(f"\n✅ Validation Accuracy: {accuracy:.2%}")
    
    # DEDUPLICATE MATCHES - Each match appears twice (symmetric dataset)
    # Keep only one row per actual match for betting simulation
    results_unique = results.drop_duplicates(subset=['date', 'player_1', 'player_2'], keep='first').copy()
    results_unique = results_unique.sort_values('date').reset_index(drop=True)
    
    print(f"\n📊 Dataset Info:")
    print(f"Total rows (with duplicates): {len(results)}")
    print(f"Unique matches: {len(results_unique)}")
    
    # BETTING SIMULATION - Only bet on predicted winner
    print(f"\n{'='*50}")
    print("BETTING PERFORMANCE ANALYSIS")
    print(f"{'='*50}")
    
    initial_bankroll = 100.0
    current_bankroll = initial_bankroll
    bet_size = 1.0  # Flat betting: 1 unit per match
    
    # Track each bet
    bet_results = []
    total_bets_with_odds = 0
    total_bets_no_odds = 0
    avg_winning_odds = []
    avg_losing_odds = []
    
    for idx, row in results_unique.iterrows():
        bet_odds = row.get('bet_odds', None)
        
        # Only bet if we have odds for our predicted player
        if pd.notna(bet_odds) and bet_odds > 0:
            total_bets_with_odds += 1
            if row['correct']:
                # Win: profit based on actual odds of predicted player
                profit = bet_size * (bet_odds - 1)
                current_bankroll += profit
                bet_results.append({'profit': profit, 'outcome': 'WIN'})
                avg_winning_odds.append(bet_odds)
            else:
                # Loss: lose bet amount
                profit = -bet_size
                current_bankroll += profit
                bet_results.append({'profit': profit, 'outcome': 'LOSS'})
                avg_losing_odds.append(bet_odds)
        else:
            # Skip this bet if no odds available
            total_bets_no_odds += 1
            bet_results.append({'profit': 0, 'outcome': 'SKIP'})
    
    total_profit = current_bankroll - initial_bankroll
    roi = (total_profit / initial_bankroll) * 100
    
    wins = results_unique['correct'].sum()
    losses = len(results_unique) - wins
    
    print(f"Total Matches: {len(results_unique)}")
    print(f"Bets Placed: {total_bets_with_odds} (with odds)")
    print(f"Bets Skipped: {total_bets_no_odds} (no odds available)")
    print(f"Wins: {wins} ({wins/total_bets_with_odds*100:.1f}%)" if total_bets_with_odds > 0 else "Wins: 0")
    print(f"Losses: {losses - total_bets_no_odds} ({(losses-total_bets_no_odds)/total_bets_with_odds*100:.1f}%)" if total_bets_with_odds > 0 else "Losses: 0")
    
    if avg_winning_odds:
        print(f"\nAverage Odds on Wins: {sum(avg_winning_odds)/len(avg_winning_odds):.2f}")
    if avg_losing_odds:
        print(f"Average Odds on Losses: {sum(avg_losing_odds)/len(avg_losing_odds):.2f}")
    
    print(f"\nInitial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    print(f"Total Profit/Loss: ${total_profit:+.2f}")
    print(f"ROI: {roi:+.2f}%")
    print(f"{'='*50}\n")
    
    # Add betting metrics to results_unique
    results_unique['bet_profit'] = [br['profit'] for br in bet_results]
    results_unique['cumulative_profit'] = results_unique['bet_profit'].cumsum() + initial_bankroll
    
    # Merge back to full results to keep all rows
    results = results.merge(
        results_unique[['date', 'player_1', 'player_2', 'bet_profit', 'cumulative_profit']],
        on=['date', 'player_1', 'player_2'],
        how='left'
    )

results.head(20)


# %%
# Save predictions to CSV for Streamlit dashboard
output_file = "tennis_predictions.csv"
results.to_csv(output_file, index=False)
print(f"✅ Predictions saved to {output_file}")

# Also save to Hopsworks for tracking
try:
    predictions_fg = fs.get_or_create_feature_group(
        name="tennis_predictions",
        description="Tennis match predictions with actual results and betting performance",
        version=1,
        primary_key=["date", "player_1", "player_2"],
        event_time="date",
    )
    
    predictions_fg.insert(results)
    print(f"✅ Predictions uploaded to Hopsworks feature group 'tennis_predictions'")
except Exception as e:
    print(f"⚠️ Could not save to Hopsworks: {e}")

# %%
