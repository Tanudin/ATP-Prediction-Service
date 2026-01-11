# ATP Tennis Match Prediction Service

A machine learning system that predicts ATP tennis match outcomes using historical data and XGBoost classification.

## Live Application

[View Live Predictions](https://atp-prediction-service-egkznj7x7vyfty9fnagxpa.streamlit.app/)

## Overview

- **Model**: XGBoost Classifier with 45 engineered features
- **Accuracy**: 61.01% on 2025 test data
- **Training Data**: 128,328 samples from 64,166 matches (2000-2024)
- **Daily Updates**: Automated via GitHub Actions

## Architecture

```
Data Pipeline:
Kaggle Dataset → Feature Engineering → Hopsworks Feature Store → XGBoost Model → Streamlit Dashboard
```

## Results

### 2025 Backtest Performance

| Strategy | Matches | Accuracy | Profit | ROI |
|----------|---------|----------|--------|-----|
| All Bets | 5,030 | 61.0% | -$379.24 | -7.54% |
| Smart Betting (≥65% confidence) | 2,945 | 67.4% | -$203.15 | -6.90% |

Smart betting improves accuracy by 6.4 percentage points while reducing losses by $176.

## Installation

### Requirements
- Python 3.11
- Hopsworks account
- Kaggle API credentials

### Setup

1. Clone repository and install dependencies:
```bash
git clone https://github.com/Tanudin/ATP-Prediction-Service.git
cd ATP-Prediction-Service
pip install -r requirements.txt
```

2. Create `.env` file with credentials:
```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=ATP_Tennis_Prediction
KAGGLE_NAME=your_username
KAGGLE_KEY=your_key
```

3. Run the pipeline:
```bash
python 1_train_historical.py  # Train model on historical data
python 2_backtest_2025.py     # Backtest on 2025 data
streamlit run streamlit_app.py # Launch dashboard
```

## Project Structure

```
├── 1_train_historical.py      # Model training on 2000-2024 data
├── 2_backtest_2025.py          # Backtesting and evaluation
├── 3_daily_update.py           # Daily predictions and retraining
├── streamlit_app.py            # Interactive dashboard
├── utils.py                    # Feature engineering functions
├── requirements.txt            # Dependencies
├── .github/workflows/
│   └── daily_update.yml        # Automated daily updates
└── tennis_model/               # Trained model artifacts
```

## Features

The model uses 45 engineered features including:
- Player rankings and ATP points
- Win percentages (overall and surface-specific)
- Historical performance metrics
- Head-to-head statistics
- Surface and tournament type advantages

Top 5 most important features:
1. Rank_Diff (131.66) - Ranking differential
2. Win_Pct_Diff (47.56) - Win percentage differential
3. Pts_Diff (25.54) - ATP points differential
4. Rank_1 (20.24) - Player 1 ranking
5. Rank_2 (19.85) - Player 2 ranking

## Technology Stack

- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Feature Store**: Hopsworks
- **Data Source**: Kaggle ATP Tennis Dataset
- **Dashboard**: Streamlit
- **Automation**: GitHub Actions
- **Deployment**: Streamlit Community Cloud

## Dashboard

The Streamlit dashboard provides:
1. **Match Predictor** - Interactive prediction tool with player selection
2. **Backtest Results** - 2025 season performance analysis
3. **Betting Strategy** - Comparison of betting approaches
4. **Model Performance** - Evaluation metrics and feature importance

## Automation

GitHub Actions runs daily at 3 AM UTC to:
1. Fetch new match data from Kaggle
2. Generate predictions for upcoming matches
3. Retrain model with latest data
4. Update Hopsworks feature store

Required GitHub Secrets:
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_HOST`
- `KAGGLE_NAME`
- `KAGGLE_KEY`

## Model Performance

**Evaluation Metrics (2025 Test Set):**
- Accuracy: 61.01%
- Precision: 61.01%
- Recall: 60.51%
- F1 Score: 60.76%
- ROC-AUC: 0.6630

## References

- [Hopsworks](https://app.hopsworks.ai) - Feature store and model registry
- [Kaggle Dataset](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) - ATP Tennis data
- [Streamlit](https://streamlit.io) - Dashboard framework

## License

MIT License
