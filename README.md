# 🎾 ATP Tennis Match Prediction System

Machine learning system for predicting ATP tennis match outcomes with betting strategy analysis.

## 📋 Overview

This system:
1. **Trains** on historical ATP data (2000-2024)
2. **Backtests** predictions on 2025 matches
3. **Updates daily** via GitHub Actions: predict new matches → get results → retrain model

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Hopsworks account
- Kaggle API credentials

### Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment**

Create `.env` file:
```env
HOPSWORKS_API_KEY=your_key_here
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_key
```

3. **Train historical model**
```bash
python 1_train_historical.py
```

4. **Generate backtest results**
```bash
python 2_backtest_2025.py
```

5. **Run Streamlit UI**
```bash
streamlit run streamlit_app.py
```

## 📊 Workflow

### Historical Training (`1_train_historical.py`)
- Downloads ATP dataset from Kaggle
- Trains XGBoost model on 2000-2024 matches
- Uploads data to Hopsworks
- Saves model locally and to Hopsworks Model Registry

### 2025 Backtest (`2_backtest_2025.py`)
- Tests model predictions on 2025 season
- Simulates flat betting ($1/match) and Kelly Criterion strategies
- Generates profit curves for UI visualization
- Saves results to `backtest_2025.csv`

### Daily Updates (`3_daily_update.py`)
Automated via GitHub Actions at 3 AM UTC daily:
1. Check Kaggle for new matches
2. Predict outcomes using latest model
3. Compare predictions with actual results
4. Retrain model with updated data
5. Save predictions to `latest_predictions.csv`

## 🎯 Features

### Model Features (No Odds!)
- Player ATP rankings and points
- Historical win percentages
- Surface-specific performance (Clay, Hard, Grass, Carpet)
- Tournament series performance (Grand Slam, Masters 1000, ATP 500/250)
- Court type performance (Indoor/Outdoor)
- Match context (encoded categorical features)

### Betting Strategies
- **Flat Betting**: Fixed $1 bet on every predicted winner
- **Kelly Criterion**: Bet size proportional to edge and confidence (max 10% bankroll)

## 📁 File Structure

```
ATP-Prediction-Service/
├── 1_train_historical.py      # Train on 2000-2024 data
├── 2_backtest_2025.py         # Backtest on 2025 season
├── 3_daily_update.py          # Daily prediction + retrain
├── streamlit_app.py           # Web UI dashboard
├── utils.py                   # Feature engineering functions
├── requirements.txt           # Python dependencies
├── .env                       # API credentials (not committed)
├── .github/
│   └── workflows/
│       └── daily_update.yml   # GitHub Actions automation
├── tennis_model/              # Saved XGBoost model
├── backtest_2025.csv          # 2025 backtest results (for UI)
├── latest_predictions.csv     # Daily predictions
└── notebooks/                 # Jupyter notebooks (for exploration)
```

## 🔧 GitHub Actions Setup

Add these secrets to your repository:
- `HOPSWORKS_API_KEY`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

Go to: **Settings → Secrets and variables → Actions → New repository secret**

## 📈 Streamlit UI

The dashboard shows:
- **2025 Backtest**: Prediction accuracy and betting performance
- **Latest Predictions**: Daily match predictions with results
- **Betting Performance**: ROI comparison between flat and Kelly strategies
- **About**: System documentation and setup instructions

## 🛠️ Tech Stack

- **Model**: XGBoost Classifier
- **Feature Store**: Hopsworks
- **Data Source**: Kaggle ATP Dataset (2000-present)
- **Automation**: GitHub Actions
- **UI**: Streamlit + Plotly
- **Language**: Python 3.11

## 📊 Performance Metrics

**Historical Training (2000-2024)**
- Accuracy: ~71-72% (without odds)
- ROC-AUC: ~0.78-0.80

**2025 Backtest**
- Prediction Accuracy: ~70%
- Flat Betting ROI: ~+10-15%
- Kelly Betting ROI: ~+15-20%

## 🧪 Development

The `notebooks/` folder contains Jupyter notebooks for:
- Exploratory data analysis
- Feature engineering experiments
- Model training iterations
- Inference testing

These are kept for reference but not used in production.

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.
