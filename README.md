# 🎾 ATP Tennis Match Prediction Service

An automated machine learning system that predicts ATP tennis match outcomes using historical data, with daily updates and betting strategy analysis.

## � Live Dashboard

**[View Live Predictions →](https://atp-prediction-service-egkznj7x7vyfty9fnagxpa.streamlit.app/)** 

## 📊 Project Overview

- **Model**: XGBoost Classifier with 45 engineered features
- **Accuracy**: 61.01% on 2025 test data (2,515 matches)
- **ROC-AUC**: 0.6630
- **Training Data**: 64,166 matches (2000-2024)
- **Daily Updates**: Automated via GitHub Actions at 3 AM UTC

## ✨ Features

- 🤖 **Automated Daily Predictions**: Fetches new matches from Kaggle, makes predictions, retrains model
- 📈 **Betting Strategy Analysis**: Compares odds-based betting vs. smart betting (confidence threshold ≥65%)
- 📊 **Interactive Dashboard**: Streamlit UI with backtest results, betting comparison, and model performance
- 🔄 **Continuous Learning**: Model retrains daily with new match data
- ☁️ **Cloud Integration**: Hopsworks for feature store and model registry

## 🎯 2025 Backtest Results

| Strategy | Bets | Accuracy | Profit | ROI |
|----------|------|----------|--------|-----|
| **All Bets** | 5,030 | 61.0% | -$379.24 | -7.54% |
| **Smart Betting** | 2,945 | 67.4% | -$203.15 | -6.90% |

*Smart betting (≥65% confidence) improved accuracy by 6.4% and reduced losses by $176*

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Hopsworks account
- Kaggle API credentials

- Python 3.11
- Hopsworks account
- Kaggle API credentials

### Installation

1. **Clone and install**
```bash
git clone https://github.com/Tanudin/ATP-Prediction-Service.git
cd ATP-Prediction-Service
pip install -r requirements.txt
```

2. **Set up environment variables**

Create `.env` file:
```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=ATP_Tennis_Prediction
KAGGLE_NAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

3. **Run the pipeline**
```bash
# Step 1: Train initial model (one-time)
python 1_train_historical.py

# Step 2: Backtest on 2025 data
python 2_backtest_2025.py

# Step 3: Launch dashboard
streamlit run streamlit_app.py
```

## �️ Tech Stack

- **ML**: XGBoost, scikit-learn, pandas, numpy
- **Feature Store**: Hopsworks
- **Data Source**: Kaggle (ATP Tennis Dataset)
- **Dashboard**: Streamlit
- **Automation**: GitHub Actions
- **Deployment**: Streamlit Community Cloud

## 📁 Project Structure

```
ATP-Prediction-Service/
├── 1_train_historical.py      # Initial model training (2000-2024)
├── 2_backtest_2025.py          # Backtest on 2025 data + metrics
├── 3_daily_update.py           # Daily automated predictions & retraining
├── streamlit_app.py            # Interactive dashboard (3 tabs)
├── utils.py                    # Feature engineering functions
├── requirements.txt            # Python dependencies
├── .github/workflows/
│   └── daily_update.yml        # GitHub Actions workflow
├── backtest_2025.csv           # 2025 backtest results
├── model_metrics.json          # ML evaluation metrics
├── feature_importance.csv      # Top 20 features by importance
└── tennis_model/               # Trained XGBoost model
```

## 📈 Top Features (by Importance)

1. **Rank_Diff** (131.66) - Difference in player rankings
2. **Win_Pct_Diff** (47.56) - Win percentage differential
3. **Pts_Diff** (25.54) - ATP points differential
4. **Player1_Rank** (20.24) - Player 1 ranking
5. **Player1_Elo** (19.85) - Player 1 Elo rating

## ⚙️ GitHub Actions Setup

The project includes automated daily updates via GitHub Actions. See [SETUP_GITHUB_ACTIONS.md](SETUP_GITHUB_ACTIONS.md) for detailed instructions.

**Required Secrets:**
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_HOST`
- `KAGGLE_NAME`
- `KAGGLE_KEY`

## � Dashboard Tabs

1. **2025 Backtest Results** - Overview metrics, profit charts, recent predictions
2. **Betting Strategy Comparison** - All Bets vs Smart Betting performance
3. **Model Performance** - Accuracy, Precision, Recall, F1, ROC-AUC, Feature Importance

## 🎓 Model Details

- **Algorithm**: XGBoost (Gradient Boosting)
- **Features**: 45 engineered features (player rankings, Elo, stats, head-to-head)
- **Training**: 128,328 samples (symmetric - each match as Player1 and Player2)
- **Evaluation**: 5 metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Version**: Stored in Hopsworks Model Registry (currently v7)

## 🤝 Contributing

This is an academic project for ID2223 - Scalable Machine Learning. Feel free to fork and experiment!

## 🔗 Links

- **Hopsworks**: [app.hopsworks.ai](https://app.hopsworks.ai)
- **Kaggle Dataset**: [ATP Tennis 2000-2023](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)
- **Streamlit Cloud**: [streamlit.io/cloud](https://streamlit.io/cloud)

---

**Built with ❤️ for scalable ML and tennis analytics**
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
