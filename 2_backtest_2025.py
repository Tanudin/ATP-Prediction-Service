from pathlib import Path
import dotenv
import hopsworks
import kagglehub
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from utils import preprocess_data, final_train_data

dotenv.load_dotenv()

print("=" * 60)
print("ATP TENNIS PREDICTION - 2025 BACKTEST")
print("=" * 60)

path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
dataset_dir = Path(path)
data_file = dataset_dir / "atp_tennis.csv"
df = pd.read_csv(data_file)
df["Date"] = pd.to_datetime(df["Date"])

season_2025_start = pd.Timestamp("2024-12-27")
test_df = df[df["Date"] >= season_2025_start].copy()
print(f"\n2025 matches for backtesting: {len(test_df):,}")

print("Processing features...")
clean_data = preprocess_data(test_df)
final_test_df = final_train_data(clean_data)

# Save match info before dropping columns (including odds)
match_info = clean_data[["Date", "Player_1", "Player_2", "Winner", "Odd_1", "Odd_2"]].copy()

print("Loading model from Hopsworks...")
project = hopsworks.login(project="ATP_Tennis_Prediction")
mr = project.get_model_registry()

all_models = mr.get_models("tennis_match_predictor")
latest_version = max([m.version for m in all_models])
print(f"Using model version: {latest_version}")

model_meta = mr.get_model("tennis_match_predictor", version=latest_version)
model_dir = model_meta.download()

booster = xgb.Booster()
booster.load_model(f"{model_dir}/model.json")

drop_cols = [
    "Date", "Player_1", "Player_2", "Winner", "timestamp",
    "Tournament", "Surface", "Series", "Round", "Court", "Tournament_Clean",
    "Player_1_Won"
]
odds_cols = [col for col in final_test_df.columns if "odd" in col.lower()]
drop_cols.extend(odds_cols)

X_test = final_test_df.drop([col for col in drop_cols if col in final_test_df.columns], axis=1)
y_test = final_test_df["Player_1_Won"]

dmatrix = xgb.DMatrix(X_test)
prob_player1_wins = booster.predict(dmatrix)
predictions = (prob_player1_wins > 0.5).astype(int)

# Calculate ML evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, prob_player1_wins)
conf_matrix = confusion_matrix(y_test, predictions)

# Get feature importance
feature_importance = booster.get_score(importance_type='gain')
feature_importance_df = pd.DataFrame([
    {"feature": k, "importance": v} 
    for k, v in feature_importance.items()
]).sort_values("importance", ascending=False)

# Save metrics to JSON
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": classification_report(y_test, predictions, output_dict=True)
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

feature_importance_df.to_csv("feature_importance.csv", index=False)

print(f"\nModel Evaluation Metrics:")
print(f"  Accuracy:  {accuracy:.2%}")
print(f"  Precision: {precision:.2%}")
print(f"  Recall:    {recall:.2%}")
print(f"  F1 Score:  {f1:.2%}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TN: {conf_matrix[0][0]:,}  FP: {conf_matrix[0][1]:,}")
print(f"    FN: {conf_matrix[1][0]:,}  TP: {conf_matrix[1][1]:,}")

# Use match_info for results
results = match_info.copy()
results["predicted_player_1_wins"] = predictions
results["player_1_win_probability"] = prob_player1_wins

predicted_winners = []
for i in range(len(predictions)):
    if predictions[i] == 1:
        predicted_winners.append(results.iloc[i]["Player_1"])
    else:
        predicted_winners.append(results.iloc[i]["Player_2"])

results["predicted_winner"] = predicted_winners
results["confidence"] = np.where(prob_player1_wins > 0.5, prob_player1_wins, 1 - prob_player1_wins)
results["correct"] = results["predicted_winner"] == results["Winner"]

accuracy = results["correct"].mean()
print(f"\n2025 Prediction Accuracy: {accuracy:.2%}")

BET_AMOUNT = 1

# Odds-based betting (use actual betting odds)
def calculate_odds_profit(row):
    if row["correct"]:
        # Win: get back (odds × stake) - stake
        if row["predicted_winner"] == row["Player_1"]:
            odds = row["Odd_1"]
        else:
            odds = row["Odd_2"]
        return (odds * BET_AMOUNT) - BET_AMOUNT
    else:
        # Lose: lose the stake
        return -BET_AMOUNT

results["bet_amount"] = BET_AMOUNT
results["profit"] = results.apply(calculate_odds_profit, axis=1)
results["cumulative_profit"] = results["profit"].cumsum()

# Smart betting strategy: only bet when confidence > threshold
CONFIDENCE_THRESHOLD = 0.65

def calculate_smart_profit(row):
    if row["confidence"] < CONFIDENCE_THRESHOLD:
        return 0  # Don't bet on low confidence predictions
    
    if row["correct"]:
        # Win: get back (odds × stake) - stake
        if row["predicted_winner"] == row["Player_1"]:
            odds = row["Odd_1"]
        else:
            odds = row["Odd_2"]
        return (odds * BET_AMOUNT) - BET_AMOUNT
    else:
        # Lose: lose the stake
        return -BET_AMOUNT

results["smart_bet_amount"] = results.apply(
    lambda row: BET_AMOUNT if row["confidence"] >= CONFIDENCE_THRESHOLD else 0,
    axis=1
)
results["smart_profit"] = results.apply(calculate_smart_profit, axis=1)
results["smart_cumulative_profit"] = results["smart_profit"].cumsum()

# Calculate statistics
total_wagered = len(results) * BET_AMOUNT
final_profit = results["cumulative_profit"].iloc[-1]
roi = (final_profit / total_wagered) * 100

smart_bets = (results["confidence"] >= CONFIDENCE_THRESHOLD).sum()
smart_total_wagered = smart_bets * BET_AMOUNT
smart_final_profit = results["smart_cumulative_profit"].iloc[-1]
smart_roi = (smart_final_profit / smart_total_wagered) * 100 if smart_total_wagered > 0 else 0
smart_accuracy = results[results["confidence"] >= CONFIDENCE_THRESHOLD]["correct"].mean()

print(f"\nBetting Performance:")
print(f"\nAll Bets (Odds-Based):")
print(f"  Total Bets:    {len(results):,}")
print(f"  Total Wagered: ${total_wagered:,}")
print(f"  Final Profit:  ${final_profit:+.2f}")
print(f"  ROI:           {roi:+.2f}%")

print(f"\nSmart Betting (Confidence >= {CONFIDENCE_THRESHOLD:.0%}):")
print(f"  Total Bets:    {smart_bets:,} ({smart_bets/len(results)*100:.1f}% of matches)")
print(f"  Accuracy:      {smart_accuracy:.2%}")
print(f"  Total Wagered: ${smart_total_wagered:,}")
print(f"  Final Profit:  ${smart_final_profit:+.2f}")
print(f"  ROI:           {smart_roi:+.2f}%")

results.to_csv("backtest_2025.csv", index=False)
print(f"\nSaved results to backtest_2025.csv")
print("=" * 60)
