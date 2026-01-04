# 🎾 ATP Tennis Match Prediction Service

**Machine Learning system for predicting ATP tennis match outcomes using XGBoost and Hopsworks Feature Store**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Hopsworks](https://img.shields.io/badge/MLOps-Hopsworks-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## 🎯 Project Overview

This service predicts professional tennis match outcomes by analyzing:
- **Player historical performance** (133K+ matches from 2000-2025)
- **Surface-specific statistics** (Hard, Clay, Grass, Carpet)
- **Tournament context** (Grand Slam, Masters 1000, ATP 500/250)
- **Head-to-head records** and recent form
- **Betting odds** and ranking differentials

**Model Performance:**
- 🎯 Accuracy: **72.3%**
- 📊 F1 Score: **0.72**
- 📈 ROC-AUC: **0.78**

## 🏗️ Architecture

```
┌─────────────────┐
│  Kaggle Dataset │ (Daily ATP matches 2000-2025)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  1_HistoricalData.ipynb     │ Feature Engineering
│  - Player statistics        │
│  - Win percentages          │
│  - Categorical encoding     │
│  - Symmetric dataset        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Hopsworks Feature Store    │ (133K matches)
│  - tennis_matches v2        │
│  - Primary key: date,p1,p2  │
└────────┬────────────────────┘
         │
         ├─────────────────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│ 3_Training      │   │ 2_Feature       │
│    Pipeline     │   │    Pipeline     │
│                 │   │                 │
│ - XGBoost       │   │ - Daily updates │
│ - Random Forest │   │ - New matches   │
│ - Logistic Reg  │   │ - Incremental   │
└────────┬────────┘   └─────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4_Inference_Pipeline       │
│  - Load latest model        │
│  - Generate predictions     │
│  - Save to CSV              │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Streamlit Dashboard        │
│  - Match predictions        │
│  - Player statistics        │
│  - Model performance        │
└─────────────────────────────┘
```

## 📁 Project Structure

```
ATP-Prediction-Service/
│
├── 1_HistoricalData.ipynb      # Initial data ingestion & preprocessing
├── 2_FeaturePipeline.ipynb     # Daily feature updates
├── 3_TrainingPipeline.ipynb    # Model training & evaluation
├── 4_Inference_Pipeline.ipynb  # Generate predictions
│
├── streamlit_app.py            # Interactive dashboard
├── utils.py                    # Shared preprocessing functions
│
├── tennis-historical.py        # Script version of notebook 1
├── tennis-feature.py           # Script version of notebook 2
├── tennis-training.py          # Script version of notebook 3
├── tennis-inference.py         # Script version of notebook 4
│
├── requirements.txt            # Python dependencies
├── tennis_predictions.csv      # Latest predictions (auto-updated)
│
└── .github/workflows/
    └── daily_update.yml        # Automated daily pipeline
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd ATP-Prediction-Service

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### 3. Run Notebooks (One-time Setup)

```bash
# 1. Load historical data (first time only)
jupyter notebook 1_HistoricalData.ipynb

# 2. Train initial model
jupyter notebook 3_TrainingPipeline.ipynb

# 3. Generate predictions
jupyter notebook 4_Inference_Pipeline.ipynb
```

### 4. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Dashboard opens at `http://localhost:8501`

## 🤖 Automated Updates (GitHub Actions)

### Schedule:
- **Daily (3 AM UTC):** Update features + generate predictions
- **Weekly (Sundays):** Retrain model with new data

### Setup Repository Secrets:

Go to your GitHub repo → Settings → Secrets and variables → Actions:

```
HOPSWORKS_API_KEY=<your-key>
KAGGLE_USERNAME=<your-username>
KAGGLE_KEY=<your-key>
```

### Manual Trigger:

```bash
# Go to GitHub Actions tab → "Daily ATP Data Update" → "Run workflow"
```

## 📊 Model Features

### Input Features (50+ columns):

**Player Statistics:**
- Total matches, wins, losses
- Overall win percentage
- Experience differential

**Surface Performance:**
- Hard/Clay/Grass/Carpet win rates
- Surface advantage metric

**Tournament Context:**
- Series (Grand Slam, Masters, ATP 500/250)
- Round (Finals, Semifinals, etc.)
- Court (Indoor/Outdoor)

**Match Context:**
- Rankings (current + differential)
- ATP points (current + differential)
- Betting odds (if available)

**Encoded Categories:**
- Tournament, Surface, Series, Round, Court

### Target Variable:
- `player_1_won` (1 = Player 1 wins, 0 = Player 2 wins)

## 🧪 Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **XGBoost** ⭐      | 0.7234   | 0.7156    | 0.7312 | 0.7233   | 0.7845  |
| Random Forest       | 0.7198   | 0.7124    | 0.7289 | 0.7205   | 0.7812  |
| Logistic Regression | 0.6891   | 0.6823    | 0.6978 | 0.6899   | 0.7456  |
| Decision Tree       | 0.6534   | 0.6467    | 0.6612 | 0.6538   | 0.6789  |

## 📈 Dashboard Features

### 🔮 Predictions Tab
- Latest match predictions with confidence scores
- Filter by date, player, confidence threshold
- Color-coded confidence levels (High/Medium/Low)

### 📊 Model Insights
- Performance metrics visualization
- Prediction confidence distribution
- Feature importance analysis

### 🏆 Player Stats
- Individual player statistics
- Win rate trends
- Surface-specific performance

### ⚡ Live Matches (Coming Soon)
- Real-time ATP schedule integration
- Live prediction updates
- Match status tracking

## 🛠️ Development

### Run Individual Scripts:

```bash
# Update features
python tennis-feature.py

# Train model
python tennis-training.py

# Generate predictions
python tennis-inference.py
```

### Modify Feature Engineering:

Edit `utils.py` functions:
- `compute_player_match_history()` - Historical statistics
- `compute_match_percentages()` - Win rate calculations
- `encode_categorical_features()` - Label encoding
- `create_symmetric_dataset()` - Duplicate from both perspectives

## 📝 Dataset

**Source:** [ATP Tennis 2000-2023 Daily Pull](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)

**Size:** 133,358 matches (after preprocessing)

**Columns:** Date, Tournament, Surface, Series, Round, Court, Best of, Players, Rankings, Points, Odds

## 🔍 Next Steps

- [ ] Add live ATP API integration
- [ ] Implement player injury data
- [ ] Add weather conditions (for outdoor matches)
- [ ] Create player head-to-head module
- [ ] Add tournament-specific models
- [ ] Implement ensemble stacking
- [ ] Deploy dashboard to Streamlit Cloud

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Built with:** Python 🐍 | XGBoost 🚀 | Hopsworks 🏗️ | Streamlit 📊 | GitHub Actions ⚙️