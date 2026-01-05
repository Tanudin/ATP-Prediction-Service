from pathlib import Path
from datetime import datetime
import dotenv
import hopsworks
import kagglehub
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from utils import preprocess_data, final_train_data

dotenv.load_dotenv()

print("=" * 60)
print("ATP TENNIS PREDICTION - HISTORICAL TRAINING")
print("=" * 60)

path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
dataset_dir = Path(path)
data_file = dataset_dir / "atp_tennis.csv"
df = pd.read_csv(data_file)
df["Date"] = pd.to_datetime(df["Date"])

print(f"\nTotal matches in dataset: {len(df):,}")

train_cutoff = pd.Timestamp("2024-12-27")
train_df = df[df["Date"] < train_cutoff].copy()
print(f"Training period: 2000-2024")
print(f"Training matches: {len(train_df):,}")

print("\nProcessing features...")
clean_data = preprocess_data(train_df)
final_train_df = final_train_data(clean_data)
final_train_df["Date"] = final_train_df["Date"].dt.strftime("%Y-%m-%d")

# Convert encoded columns to int64 for Hopsworks compatibility
encoded_cols = ['Tournament_Encoded', 'Surface_Encoded', 'Series_Encoded', 'Round_Encoded', 'Court_Encoded']
for col in encoded_cols:
    if col in final_train_df.columns:
        final_train_df[col] = final_train_df[col].astype('int64')

pk_cols = ["Date", "Player_1", "Player_2"]
final_train_df = final_train_df.drop_duplicates(subset=pk_cols, keep="first")
print(f"Final training samples: {len(final_train_df):,}")

print("\nConnecting to Hopsworks...")
project = hopsworks.login(project="ATP_Tennis_Prediction")
fs = project.get_feature_store()

tennis_fg = fs.get_or_create_feature_group(
    name="tennis_matches",
    version=2,
    primary_key=["Date", "Player_1", "Player_2"],
    event_time="timestamp",
    online_enabled=False,
)

print("Inserting data to Hopsworks...")
tennis_fg.insert(final_train_df)
print(f"✓ Uploaded to Hopsworks")

print("\nTraining XGBoost model...")
drop_cols = [
    "Date", "Player_1", "Player_2", "Winner", "timestamp",
    "Tournament", "Surface", "Series", "Round", "Court", "Tournament_Clean",
    "Player_1_Won"
]

odds_cols = [col for col in final_train_df.columns if "odd" in col.lower()]
drop_cols.extend(odds_cols)

X = final_train_df.drop([col for col in drop_cols if col in final_train_df.columns], axis=1)
y = final_train_df["Player_1_Won"]

print(f"Features: {len(X.columns)}")
print(f"Removed odds columns: {len(odds_cols)}")

model = XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100)
model.fit(X, y)

y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_proba)

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  ROC-AUC:  {roc_auc:.4f}")

import os
model_dir = "tennis_model"
os.makedirs(model_dir, exist_ok=True)
model.get_booster().save_model(f"{model_dir}/model.json")

mr = project.get_model_registry()
tennis_model = mr.python.create_model(
    name="tennis_match_predictor",
    metrics={
        "Accuracy": str(round(accuracy, 4)),
        "ROC-AUC": str(round(roc_auc, 4)),
        "Training_Samples": str(len(X)),
        "Features": str(len(X.columns))
    },
    description="XGBoost trained on 2000-2024 ATP matches without odds"
)
tennis_model.save(model_dir)

print(f"\n✓ Model saved (version {tennis_model.version})")
print("=" * 60)
