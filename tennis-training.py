# %% [markdown]
# ## Import Stuff
#

# %%
import warnings

import hopsworks
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# %% [markdown]
# ## Connect to Hopsworks and Load Data
#

# %%
project = hopsworks.login(project="ATP_Tennis_Prediction")
fs = project.get_feature_store()
tennis_fg = fs.get_feature_group(
    name="tennis_matches",
    version=2,
)

# %% [markdown]
# ## EDA
#

# %%
tennis_df = tennis_fg.read()

print(f"Total records: {len(tennis_df):,}")
print(f"Total columns: {len(tennis_df.columns)}")
print("\nColumn names:")
print(tennis_df.columns.tolist())

print("\nTarget distribution:")
print(tennis_df["player_1_won"].value_counts())
print(tennis_df["player_1_won"].value_counts(normalize=True))

tennis_df.head()


# %% [markdown]
# ## Feature View
#

# %%
selected_features = tennis_fg.select_all()
selected_features.show(10)


# %%
feature_view = fs.get_or_create_feature_view(
    name="tennis_match_prediction",
    description="Tennis match features for predicting Player 1 win/loss",
    version=1,
    labels=["player_1_won"],
    query=selected_features,
)

# %%
# Run this first to verify:
sample = tennis_fg.read()
target_cols = [col for col in sample.columns if "won" in col.lower()]
print(f"Columns with 'won': {target_cols}")


# %% [markdown]
# ## Model Data Preprocesing
#

# %%
X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# %%
drop_list = [
    # Metadata columns
    "date",  # When match occurred (not predictive)
    "player_1",  # Player name (text)
    "player_2",  # Player name (text)
    "winner",  # This IS the target in different form!
    "timestamp",  # When data was inserted
    # Categorical columns (we have encoded versions)
    "tournament",  # Have Tournament_Encoded
    "surface",  # Have Surface_Encoded
    "series",  # Have Series_Encoded
    "round",  # Have Round_Encoded
    "court",  # Have Court_Encoded
    "tournament_clean",  # Intermediate processing column
]

# Filter for columns that actually exist
cols_to_drop = [col for col in drop_list if col in X_train.columns]

print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")

train_features = X_train.drop(cols_to_drop, axis=1)
test_features = X_test.drop(cols_to_drop, axis=1)

print(f"Features used: {len(train_features.columns)}")
print(f"Column names: {train_features.columns.tolist()}")

# %% [markdown]
# ## Train Models
#

# %%
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42)

models = {
    "XGBoost": xgb_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
}

model_scores = {
    "XGBoost": {},
    "Random Forest": {},
    "Logistic Regression": {},
    "Decision Tree": {},
}

# %%
for name, model in models.items():
    print(f"Training {name}...")

    model.fit(train_features, y_train.values.ravel())

    y_pred = model.predict(test_features)
    y_pred_proba = model.predict_proba(test_features)[:, 1]

    # CLASSIFICATION METRICS (not MSE/R²!)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print()

    # Store results
    model_scores[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    }


# %% [markdown]
# ## Feature Engineering
#

# %%
odds_cols = [col for col in train_features.columns if "odd" in col or "odds" in col]
if len(odds_cols) > 0:
    train_no_odds = train_features.drop(odds_cols, axis=1)
    test_no_odds = test_features.drop(odds_cols, axis=1)

    xgb_no_odds = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_no_odds.fit(train_no_odds, y_train.values.ravel())
    y_pred_no_odds = xgb_no_odds.predict(test_no_odds)

    accuracy_with = model_scores["XGBoost"]["Accuracy"]
    accuracy_without = accuracy_score(y_test, y_pred_no_odds)

    print(f"With odds:    {accuracy_with:.4f}")
    print(f"Without odds: {accuracy_without:.4f}")


# %% [markdown]
# ## Choose Best Model
#

# %%
final_model = XGBClassifier(random_state=42, eval_metric="logloss")
final_model.fit(train_features, y_train.values.ravel())

y_pred_final = final_model.predict(test_features)
y_pred_proba_final = final_model.predict_proba(test_features)[:, 1]

# Final metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_pred_proba_final)

print("Final Performance:")
print(f"  Accuracy:  {final_accuracy:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall:    {final_recall:.4f}")
print(f"  F1 Score:  {final_f1:.4f}")
print(f"  ROC-AUC:   {final_roc_auc:.4f}")


# %% [markdown]
# ## Feature Analysis
#

# %%
from xgboost import plot_importance

# Get feature importance as DataFrame
feature_importance = pd.DataFrame(
    {"Feature": train_features.columns, "Importance": final_model.feature_importances_}
).sort_values("Importance", ascending=False)

plot_importance(final_model, max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# %%
n_samples = 10

for i in range(n_samples):
    pred = y_pred_final[i]
    actual = y_test.iloc[i, 0]
    prob = y_pred_proba_final[i]

    pred_label = "P1 Wins" if pred == 1 else "P2 Wins"
    actual_label = "P1 Wins" if actual == 1 else "P2 Wins"
    correct = "C" if pred == actual else "W"

    print(
        f"{i + 1}. Predicted: {pred_label} ({prob:.2%}) | Actual: {actual_label} {correct}"
    )


# %%
from hsml.model_schema import ModelSchema
from hsml.schema import Schema

input_schema = Schema(train_features)
output_schema = Schema(y_test)

model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

print("Model schema created")


# %%
print("Retraining final model from scratch...")

# Create fresh XGBoost model
final_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
)

# Train
final_model.fit(train_features, y_train.values.ravel())

# Verify it works
y_pred_final = final_model.predict(test_features)
y_pred_proba_final = final_model.predict_proba(test_features)[:, 1]

final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_pred_proba_final)

print("Final Model Retrained:")
print(f"   Type: {type(final_model)}")
print(f"   Accuracy: {final_accuracy:.4f}")
print(f"   F1 Score: {final_f1:.4f}")
print(f"   ROC-AUC: {final_roc_auc:.4f}")

# Update metrics dictionary
metrics_dict = {
    "Accuracy": str(round(final_accuracy, 4)),
    "Precision": str(round(final_precision, 4)),
    "Recall": str(round(final_recall, 4)),
    "F1 Score": str(round(final_f1, 4)),
    "ROC-AUC": str(round(final_roc_auc, 4)),
}


# %%
import os

mr = project.get_model_registry()

model_dir = "tennis_model"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# Use get_booster() method for XGBoost
final_model.get_booster().save_model(model_dir + "/model.json")
print(f"Model saved to {model_dir}/model.json")

# Register in Hopsworks
tennis_prediction_model = mr.python.create_model(
    name="tennis_match_predictor",
    metrics=metrics_dict,
    model_schema=model_schema,
    input_example=X_test.sample().values,
    description="XGBoost Classifier for predicting tennis match outcomes (Player 1 Win/Loss)",
)

tennis_prediction_model.save(model_dir)

print("Model registered successfully!")
print("Name: tennis_match_predictor")
print(f"   Version: {tennis_prediction_model.version}")
print(f"   Accuracy: {metrics_dict['Accuracy']}")


# %%
