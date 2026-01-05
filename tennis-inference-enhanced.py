"""
Enhanced Inference Pipeline for ATP Match Predictions.
Generates predictions for both historical test data and upcoming matches.
"""

import os
from datetime import datetime
from pathlib import Path

import dotenv
import hopsworks
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load environment variables
dotenv.load_dotenv()

# Import utilities
from utils import (
    encode_categorical_features,
    compute_player_match_history,
    compute_match_percentages,
    compute_derived_features,
)

# Try to import upcoming matches fetcher
try:
    from fetch_upcoming_matches import UpcomingMatchesFetcher

    UPCOMING_AVAILABLE = True
except ImportError:
    UPCOMING_AVAILABLE = False
    print("⚠️ fetch_upcoming_matches.py not found - skipping upcoming matches")


def load_model_and_feature_view(project):
    """Load the trained model and feature view from Hopsworks."""
    print("📦 Loading model from Hopsworks...")

    # Get model registry
    mr = project.get_model_registry()

    # Retrieve the best model
    model = mr.get_model(
        name="atp_match_predictor",
        version=1,  # Or get the latest version
    )

    model_dir = model.download()
    model_path = Path(model_dir) / "xgboost_model.json"

    # Load XGBoost model
    import xgboost as xgb

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(model_path))

    print(f"✅ Model loaded from: {model_path}")

    # Get feature view
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(name="atp_matches_view", version=1)

    return xgb_model, feature_view


def prepare_upcoming_matches(upcoming_df: pd.DataFrame, historical_df: pd.DataFrame):
    """
    Prepare upcoming matches for prediction by computing features
    based on historical data.

    Args:
        upcoming_df: DataFrame with upcoming matches
        historical_df: DataFrame with historical match data for feature computation

    Returns:
        Prepared DataFrame ready for prediction
    """
    print("🔧 Preparing upcoming matches with historical features...")

    # Ensure date is datetime
    if "date" in upcoming_df.columns:
        upcoming_df["date"] = pd.to_datetime(upcoming_df["date"])
    elif "Date" in upcoming_df.columns:
        upcoming_df.rename(columns={"Date": "date"}, inplace=True)
        upcoming_df["date"] = pd.to_datetime(upcoming_df["date"])

    # Map column names to expected format
    column_mapping = {
        "player_1": "Player_1",
        "player_2": "Player_2",
        "tournament": "Tournament",
        "surface": "Surface",
        "series": "Series",
        "round": "Round",
        "rank_1": "WRank_1",
        "rank_2": "WRank_2",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in upcoming_df.columns and new_col not in upcoming_df.columns:
            upcoming_df.rename(columns={old_col: new_col}, inplace=True)

    # Add missing columns with default values
    if "Court" not in upcoming_df.columns:
        upcoming_df["Court"] = "Outdoor"  # Default assumption

    if "Best_of" not in upcoming_df.columns:
        # Best of 5 for Grand Slams, 3 for others
        upcoming_df["Best_of"] = upcoming_df["Series"].apply(
            lambda x: 5 if "Grand Slam" in str(x) else 3
        )

    # Compute player statistics from historical data
    print("  Computing player historical stats...")
    upcoming_with_stats = compute_player_match_history(upcoming_df, historical_df)

    # Compute match percentages
    print("  Computing win percentages...")
    upcoming_with_pct = compute_match_percentages(upcoming_with_stats, historical_df)

    # Compute derived features
    print("  Computing derived features...")
    upcoming_prepared = compute_derived_features(upcoming_with_pct)

    # Encode categorical features
    print("  Encoding categorical features...")
    upcoming_encoded = encode_categorical_features(upcoming_prepared, historical_df)

    print(f"✅ Prepared {len(upcoming_encoded)} upcoming matches")

    return upcoming_encoded


def generate_predictions(
    model, feature_view, match_data: pd.DataFrame, match_type: str = "test"
):
    """
    Generate predictions for given matches.

    Args:
        model: Trained XGBoost model
        feature_view: Hopsworks feature view
        match_data: DataFrame with match data
        match_type: 'test' or 'upcoming'

    Returns:
        DataFrame with predictions
    """
    print(f"\n🔮 Generating predictions for {match_type} matches...")

    # Get feature columns (excluding target and metadata)
    feature_cols = [
        col
        for col in match_data.columns
        if col
        not in [
            "date",
            "Date",
            "player_1_won",
            "Winner",
            "Player_1",
            "Player_2",
            "Tournament",
            "match_id",
            "source",
        ]
    ]

    X = match_data[feature_cols]

    # Convert to DMatrix for XGBoost
    import xgboost as xgb

    dtest = xgb.DMatrix(X)

    # Generate predictions
    predictions = model.predict(dtest)

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "date": match_data["date"].values
            if "date" in match_data.columns
            else match_data["Date"].values,
            "player_1": match_data["Player_1"].values,
            "player_2": match_data["Player_2"].values,
            "tournament": match_data["Tournament"].values,
            "predicted_winner": predictions.round().astype(int),
            "confidence": predictions,
            "match_type": match_type,
        }
    )

    # Add winner names
    results["predicted_winner_name"] = results.apply(
        lambda row: row["player_1"] if row["predicted_winner"] == 1 else row["player_2"],
        axis=1,
    )

    # Classify confidence levels
    results["confidence_level"] = pd.cut(
        results["confidence"],
        bins=[0, 0.4, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    # Sort by date and confidence
    results = results.sort_values(["date", "confidence"], ascending=[True, False])

    print(f"✅ Generated {len(results)} predictions")

    return results


def main():
    """Main inference pipeline."""
    print("🎾 ATP Match Prediction - Enhanced Inference Pipeline\n")

    # Connect to Hopsworks
    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(project="ATP_Tennis_Prediction")

    # Load model and feature view
    model, feature_view = load_model_and_feature_view(project)

    # Get feature store for historical data
    fs = project.get_feature_store()
    tennis_fg = fs.get_feature_group(name="tennis_matches", version=2)
    historical_df = tennis_fg.read()

    all_predictions = []

    # ===== 1. Predictions on Test Set (Historical) =====
    print("\n" + "=" * 60)
    print("📊 PART 1: Test Set Predictions (2025 Season)")
    print("=" * 60)

    # Filter for 2025 season test data
    season_2025_start = pd.Timestamp("2024-12-27")
    season_2025_end = pd.Timestamp("2025-11-16")

    test_df = historical_df[
        (historical_df["date"] >= season_2025_start)
        & (historical_df["date"] <= season_2025_end)
    ].copy()

    if not test_df.empty:
        test_predictions = generate_predictions(model, feature_view, test_df, "test")
        all_predictions.append(test_predictions)

        # Calculate accuracy on test set
        if "player_1_won" in test_df.columns:
            accuracy = (
                test_predictions["predicted_winner"] == test_df["player_1_won"].values
            ).mean()
            print(f"\n📈 Test Set Accuracy: {accuracy:.2%}")
    else:
        print("⚠️ No test data found for 2025 season")

    # ===== 2. Predictions on Upcoming Matches =====
    if UPCOMING_AVAILABLE:
        print("\n" + "=" * 60)
        print("🔮 PART 2: Upcoming Match Predictions")
        print("=" * 60)

        # Fetch upcoming matches
        fetcher = UpcomingMatchesFetcher()
        upcoming_df = fetcher.get_upcoming_matches(sources=["sofascore"])

        if not upcoming_df.empty:
            # Prepare upcoming matches with features
            upcoming_prepared = prepare_upcoming_matches(upcoming_df, historical_df)

            # Generate predictions
            upcoming_predictions = generate_predictions(
                model, feature_view, upcoming_prepared, "upcoming"
            )
            all_predictions.append(upcoming_predictions)

            print("\n🎯 Top 5 Upcoming Match Predictions:")
            print(
                upcoming_predictions[
                    [
                        "date",
                        "tournament",
                        "player_1",
                        "player_2",
                        "predicted_winner_name",
                        "confidence",
                        "confidence_level",
                    ]
                ]
                .head()
                .to_string(index=False)
            )
        else:
            print("⚠️ No upcoming matches found")
    else:
        print("\n⚠️ Skipping upcoming matches (fetch_upcoming_matches.py not available)")

    # ===== 3. Save All Predictions =====
    print("\n" + "=" * 60)
    print("💾 Saving Predictions")
    print("=" * 60)

    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        # Save to CSV
        output_file = "tennis_predictions_enhanced.csv"
        combined_predictions.to_csv(output_file, index=False)
        print(f"✅ Saved {len(combined_predictions)} predictions to {output_file}")

        # Summary statistics
        print("\n📊 Prediction Summary:")
        print(
            combined_predictions.groupby(["match_type", "confidence_level"])
            .size()
            .to_frame("count")
        )

        # Save timestamp
        timestamp_file = "last_prediction_update.txt"
        with open(timestamp_file, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
        print(f"✅ Updated timestamp: {timestamp_file}")

    else:
        print("❌ No predictions generated")

    print("\n✅ Inference pipeline complete!")


if __name__ == "__main__":
    main()
