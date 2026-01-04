# %% [markdown]
# ## Smart Betting Strategy - Variable Bet Sizing
# This version uses Kelly Criterion and confidence-based betting

# %%
import dotenv

dotenv.load_dotenv()


# %% [markdown]
# ## Imports

# %%
import hopsworks
import pandas as pd
import numpy as np
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
    name="tennis_match_predictor",
    version=1,
)

saved_model_dir = retrieved_model.download()

model = XGBClassifier()
model.load_model(saved_model_dir + "/model.json")


# %% [markdown]
# ### Load 2025 Matches

# %%
# Get ALL matches from main feature group
tennis_fg = fs.get_feature_group(
    name="tennis_matches",
    version=2,
)

all_matches = tennis_fg.read()

# Filter for 2025 matches
all_matches["date"] = pd.to_datetime(all_matches["date"])
df_new = all_matches[all_matches["date"].dt.year == 2025].copy()


# %% [markdown]
# ## Make Predictions

# %%
drop_list = [
    # Metadata
    "date",
    "player_1",
    "player_2",
    "timestamp",
    # Target
    "player_1_won",
]

# Only drop columns that actually exist in the dataframe
drop_list = [col for col in drop_list if col in df_new.columns]

features = df_new.drop(columns=drop_list)

print(f"Features shape: {features.shape}")
print(f"Features columns: {features.columns.tolist()}")

# %%
# Make predictions
predictions = model.predict(features)
prediction_probabilities = model.predict_proba(features)
prob_player1_wins = prediction_probabilities[:, 1]  # P(Player 1 wins)

print(f"Predictions made: {len(predictions)}")
print(f"Player 1 wins predicted: {predictions.sum()} ({predictions.mean():.1%})")


# %%
# Create results dataframe
available_cols = ["date", "player_1", "player_2"]
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
    
    
    # %% [markdown]
    # ## SMART BETTING STRATEGY - Kelly Criterion with Confidence Threshold
    
    # %%
    # DEDUPLICATE MATCHES - Each match appears twice (symmetric dataset)
    results_unique = results.drop_duplicates(subset=['date', 'player_1', 'player_2'], keep='first').copy()
    results_unique = results_unique.sort_values('date').reset_index(drop=True)
    
    print(f"\n📊 Dataset Info:")
    print(f"Total rows (with duplicates): {len(results)}")
    print(f"Unique matches: {len(results_unique)}")
    
    print(f"\n{'='*60}")
    print("SMART BETTING STRATEGY - Kelly Criterion")
    print(f"{'='*60}")
    
    initial_bankroll = 100.0
    current_bankroll = initial_bankroll
    min_confidence = 0.60  # Only bet when confidence > 60%
    kelly_fraction = 0.25  # Use 25% of Kelly (fractional Kelly for safety)
    min_bet = 0.50  # Minimum bet size
    max_bet = 5.0   # Maximum bet size (risk management)
    
    # Track each bet
    bet_results = []
    total_bets_placed = 0
    total_bets_skipped_no_odds = 0
    total_bets_skipped_low_confidence = 0
    avg_winning_odds = []
    avg_losing_odds = []
    avg_bet_sizes = []
    high_confidence_bets = 0
    
    for idx, row in results_unique.iterrows():
        bet_odds = row.get('bet_odds', None)
        confidence = row['confidence']
        
        # Skip if no odds available
        if pd.isna(bet_odds) or bet_odds <= 0:
            total_bets_skipped_no_odds += 1
            bet_results.append({'bet_size': 0, 'profit': 0, 'outcome': 'SKIP_NO_ODDS'})
            continue
        
        # Skip if confidence too low
        if confidence < min_confidence:
            total_bets_skipped_low_confidence += 1
            bet_results.append({'bet_size': 0, 'profit': 0, 'outcome': 'SKIP_LOW_CONF'})
            continue
        
        # Calculate Kelly Criterion bet size
        # Kelly formula: f = (bp - q) / b
        # where: b = odds - 1 (profit multiplier)
        #        p = probability of winning (our model confidence)
        #        q = probability of losing (1 - p)
        
        b = bet_odds - 1  # Profit multiplier
        p = confidence      # Our win probability
        q = 1 - p           # Loss probability
        
        # Kelly fraction
        kelly_bet_fraction = (b * p - q) / b if b > 0 else 0
        
        # Apply fractional Kelly for safety and cap bet size
        if kelly_bet_fraction > 0:
            bet_size = current_bankroll * kelly_bet_fraction * kelly_fraction
            bet_size = max(min_bet, min(bet_size, max_bet))  # Enforce min/max
            bet_size = min(bet_size, current_bankroll * 0.10)  # Never bet more than 10% of bankroll
        else:
            # Negative Kelly means don't bet (odds are unfavorable)
            total_bets_skipped_low_confidence += 1
            bet_results.append({'bet_size': 0, 'profit': 0, 'outcome': 'SKIP_NEG_KELLY'})
            continue
        
        # Place the bet
        total_bets_placed += 1
        avg_bet_sizes.append(bet_size)
        
        if confidence > 0.70:
            high_confidence_bets += 1
        
        if row['correct']:
            # Win: profit based on actual odds
            profit = bet_size * (bet_odds - 1)
            current_bankroll += profit
            bet_results.append({'bet_size': bet_size, 'profit': profit, 'outcome': 'WIN'})
            avg_winning_odds.append(bet_odds)
        else:
            # Loss: lose bet amount
            profit = -bet_size
            current_bankroll += profit
            bet_results.append({'bet_size': bet_size, 'profit': profit, 'outcome': 'LOSS'})
            avg_losing_odds.append(bet_odds)
        
        # Prevent bankruptcy
        if current_bankroll <= 0:
            print(f"\n⚠️ BANKRUPT at match {idx}! Stopping simulation.")
            # Fill remaining with skips
            for _ in range(len(results) - idx - 1):
                bet_results.append({'bet_size': 0, 'profit': 0, 'outcome': 'SKIP_BANKRUPT'})
            break
    
    # Calculate final statistics
    total_profit = current_bankroll - initial_bankroll
    roi = (total_profit / initial_bankroll) * 100
    
    wins = sum(1 for br in bet_results if br['outcome'] == 'WIN')
    losses = sum(1 for br in bet_results if br['outcome'] == 'LOSS')
    
    print(f"\n📊 BETTING SUMMARY")
    print(f"{'─'*60}")
    print(f"Total Matches: {len(results_unique)}")
    print(f"Bets Placed: {total_bets_placed}")
    print(f"  - High Confidence (>70%): {high_confidence_bets}")
    print(f"Bets Skipped: {len(results_unique) - total_bets_placed}")
    print(f"  - No Odds: {total_bets_skipped_no_odds}")
    print(f"  - Low Confidence (<{min_confidence*100:.0f}%): {total_bets_skipped_low_confidence}")
    
    print(f"\n🎯 PERFORMANCE")
    print(f"{'─'*60}")
    if total_bets_placed > 0:
        print(f"Wins: {wins} ({wins/total_bets_placed*100:.1f}%)")
        print(f"Losses: {losses} ({losses/total_bets_placed*100:.1f}%)")
        print(f"Average Bet Size: ${sum(avg_bet_sizes)/len(avg_bet_sizes):.2f}")
        print(f"Largest Bet: ${max(avg_bet_sizes):.2f}")
        print(f"Smallest Bet: ${min(avg_bet_sizes):.2f}")
    
    if avg_winning_odds:
        print(f"\nAverage Odds on Wins: {sum(avg_winning_odds)/len(avg_winning_odds):.2f}")
    if avg_losing_odds:
        print(f"Average Odds on Losses: {sum(avg_losing_odds)/len(avg_losing_odds):.2f}")
    
    print(f"\n💰 FINANCIAL RESULTS")
    print(f"{'─'*60}")
    print(f"Initial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll: ${current_bankroll:.2f}")
    print(f"Total Profit/Loss: ${total_profit:+.2f}")
    print(f"ROI: {roi:+.2f}%")
    print(f"{'='*60}\n")
    
    # Add betting metrics to results_unique
    results_unique['bet_size'] = [br['bet_size'] for br in bet_results]
    results_unique['bet_profit'] = [br['profit'] for br in bet_results]
    results_unique['bet_outcome'] = [br['outcome'] for br in bet_results]
    results_unique['cumulative_profit'] = results_unique['bet_profit'].cumsum()
    results_unique['cumulative_bankroll'] = initial_bankroll + results_unique['cumulative_profit']
    
    # Merge back to full results to keep all rows
    results = results.merge(
        results_unique[['date', 'player_1', 'player_2', 'bet_size', 'bet_profit', 'bet_outcome', 'cumulative_profit', 'cumulative_bankroll']],
        on=['date', 'player_1', 'player_2'],
        how='left'
    )

results.head(20)


# %%
# Save predictions to CSV for comparison
output_file = "tennis_predictions_smart.csv"
results.to_csv(output_file, index=False)
print(f"✅ Smart betting predictions saved to {output_file}")

# Also save to Hopsworks with different feature group name
try:
    predictions_fg = fs.get_or_create_feature_group(
        name="tennis_predictions_smart",
        description="Tennis match predictions with Kelly Criterion betting strategy",
        version=1,
        primary_key=["date", "player_1", "player_2"],
        event_time="date",
    )
    
    predictions_fg.insert(results)
    print(f"✅ Smart predictions uploaded to Hopsworks feature group 'tennis_predictions_smart'")
except Exception as e:
    print(f"⚠️ Could not save to Hopsworks: {e}")

# %%
