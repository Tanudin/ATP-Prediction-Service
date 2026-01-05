from pathlib import Path
from datetime import datetime
import os
import dotenv
import hopsworks
import kagglehub
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from utils import preprocess_data, final_train_data

# Load .env file if it exists (for local development)
dotenv.load_dotenv()

print("=" * 60)
print(f"DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Hopsworks login - API key from environment variable
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("HOPSWORKS_API_KEY environment variable not set")

project = hopsworks.login(
    project="ATP_Tennis_Prediction",
    api_key_value=api_key
)
fs = project.get_feature_store()
mr = project.get_model_registry()

tennis_fg = fs.get_feature_group("tennis_matches", version=2)
existing_data = tennis_fg.read()
existing_data["Date"] = pd.to_datetime(existing_data["Date"])
latest_date_in_fs = existing_data["Date"].max()

print(f"\nLatest data in Hopsworks: {latest_date_in_fs.strftime('%Y-%m-%d')}")

print("Checking Kaggle for new matches...")
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
dataset_dir = Path(path)
data_file = dataset_dir / "atp_tennis.csv"
df = pd.read_csv(data_file)
df["Date"] = pd.to_datetime(df["Date"])

latest_date_kaggle = df["Date"].max()
print(f"Latest data in Kaggle: {latest_date_kaggle.strftime('%Y-%m-%d')}")

new_matches = df[df["Date"] > latest_date_in_fs].copy()

if len(new_matches) == 0:
    print("\nNo new matches found. Exiting.")
    exit(0)

print(f"\n✓ Found {len(new_matches)} new matches!")

print("Processing new matches...")
clean_new = preprocess_data(new_matches)
final_new = final_train_data(clean_new)
final_new["Date"] = final_new["Date"].dt.strftime("%Y-%m-%d")
final_new = final_new.drop_duplicates(subset=["Date", "Player_1", "Player_2"], keep="first")

tennis_fg.insert(final_new)
print(f"✓ Uploaded {len(final_new)} new matches to Hopsworks")

print("\nLoading latest model...")
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
odds_cols = [col for col in final_new.columns if "odd" in col.lower()]
drop_cols.extend(odds_cols)

X_new = final_new.drop([col for col in drop_cols if col in final_new.columns], axis=1)
y_new = final_new["Player_1_Won"]

dmatrix = xgb.DMatrix(X_new)
prob_player1_wins = booster.predict(dmatrix)
predictions = (prob_player1_wins > 0.5).astype(int)

results = final_new[["Date", "Player_1", "Player_2", "Winner"]].copy()
results["predicted_winner"] = results.apply(
    lambda row: row["Player_1"] if predictions[row.name] == 1 else row["Player_2"],
    axis=1
)
results["player_1_win_probability"] = prob_player1_wins
results["confidence"] = np.where(prob_player1_wins > 0.5, prob_player1_wins, 1 - prob_player1_wins)
results["correct"] = results["predicted_winner"] == results["Winner"]

accuracy = results["correct"].mean()
print(f"\nPrediction Accuracy: {accuracy:.2%}")

results.to_csv("latest_predictions.csv", index=False)
print(f"✓ Saved {len(results)} predictions")

print("\nRetraining model with updated data...")
all_data = tennis_fg.read()
all_data["Date"] = pd.to_datetime(all_data["Date"])
all_data = all_data.sort_values("Date")

X_all = all_data.drop([col for col in drop_cols if col in all_data.columns], axis=1)
y_all = all_data["Player_1_Won"]

new_model = XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100)
new_model.fit(X_all, y_all)

y_pred = new_model.predict(X_all)
y_pred_proba = new_model.predict_proba(X_all)[:, 1]

accuracy = accuracy_score(y_all, y_pred)
roc_auc = roc_auc_score(y_all, y_pred_proba)

print(f"Updated Model Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  ROC-AUC:  {roc_auc:.4f}")

import os
model_dir = "tennis_model"
os.makedirs(model_dir, exist_ok=True)
new_model.get_booster().save_model(f"{model_dir}/model.json")

tennis_model = mr.python.create_model(
    name="tennis_match_predictor",
    metrics={
        "Accuracy": str(round(accuracy, 4)),
        "ROC-AUC": str(round(roc_auc, 4)),
        "Updated": datetime.now().strftime("%Y-%m-%d"),
        "Total_Matches": str(len(all_data))
    },
    description=f"Retrained with {len(all_data):,} matches"
)
tennis_model.save(model_dir)

print(f"\n✓ Model retrained (version {tennis_model.version})")
print("=" * 60)
