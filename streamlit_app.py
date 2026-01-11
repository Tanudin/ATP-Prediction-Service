import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import numpy as np
import xgboost as xgb
from utils import preprocess_data, final_train_data
import kagglehub
from datetime import datetime

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾", layout="wide")

st.markdown("""
<style>
    div[data-baseweb="select"] > div {
        min-width: 250px !important;
    }
    [data-baseweb="popover"] {
        min-width: 300px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("ATP Tennis Match Prediction System")
st.markdown("Machine learning-powered tennis match predictions with betting strategy analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "Match Predictor",
    "2025 Backtest Results",
    "Betting Strategy Comparison",
    "Model Performance"
])

@st.cache_resource
def load_model():
    """Load the XGBoost model from local file or Hopsworks model registry"""
    try:
        # Try loading from local file first (much faster)
        local_model_path = Path("tennis_model/model.json")
        if local_model_path.exists():
            booster = xgb.Booster()
            booster.load_model(str(local_model_path))
            return booster
    except Exception as e:
        st.warning(f"Could not load local model: {e}. Trying Hopsworks...")
    
    # Fallback to Hopsworks if local file doesn't exist or fails
    try:
        import hopsworks
        import dotenv
        dotenv.load_dotenv()
        
        project = hopsworks.login(project="ATP_Tennis_Prediction")
        mr = project.get_model_registry()
        
        all_models = mr.get_models("tennis_match_predictor")
        latest_version = max([m.version for m in all_models])
        
        model_meta = mr.get_model("tennis_match_predictor", version=latest_version)
        model_dir = model_meta.download()
        
        booster = xgb.Booster()
        booster.load_model(f"{model_dir}/model.json")
        return booster
    except Exception as e:
        st.error(f"Could not load model from Hopsworks: {e}")
        return None

@st.cache_data
def load_and_process_historical_data():
    """Load historical data and compute player statistics"""
    try:
        with st.spinner("Loading historical match data..."):
            path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
            dataset_dir = Path(path)
            data_file = dataset_dir / "atp_tennis.csv"
            df = pd.read_csv(data_file)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"] < pd.Timestamp.now()].copy()
        
        with st.spinner("Computing player statistics..."):
            processed_df = preprocess_data(df)
            final_df = final_train_data(processed_df)
            
        return df, final_df
    except Exception as e:
        st.error(f"Could not load historical data: {e}")
        return None, None

def extract_player_features(processed_df, historical_df, player_name):
    """Extract the latest statistics for a specific player"""
    player_rows = processed_df[processed_df['Player_1'] == player_name]
    
    if len(player_rows) == 0:
        return None
    
    latest_row = player_rows.iloc[-1]
    
    player_features = {}
    for col in latest_row.index:
        if col.startswith('P1_'):
            feature_name = col[3:]
            if feature_name not in ['Rank', 'Pts']:
                player_features[feature_name] = latest_row[col]
    
    p1_matches = historical_df[historical_df['Player_1'] == player_name].copy()
    p2_matches = historical_df[historical_df['Player_2'] == player_name].copy()
    
    latest_rank = 50
    latest_pts = 0
    
    if len(p1_matches) > 0:
        last_p1 = p1_matches.iloc[-1]
        latest_rank = last_p1['Rank_1'] if last_p1['Rank_1'] > 0 else 50
        latest_pts = last_p1['Pts_1'] if last_p1['Pts_1'] > 0 else 0
    
    if len(p2_matches) > 0:
        last_p2 = p2_matches.iloc[-1]
        if len(p1_matches) == 0 or last_p2['Date'] > last_p1['Date']:
            latest_rank = last_p2['Rank_2'] if last_p2['Rank_2'] > 0 else 50
            latest_pts = last_p2['Pts_2'] if last_p2['Pts_2'] > 0 else 0
    
    player_features['_latest_rank'] = latest_rank
    player_features['_latest_pts'] = latest_pts
    
    return player_features

def build_match_features(player1_stats, player2_stats, surface, round_type, series, court, best_of):
    """Build feature vector for a match prediction"""
    match_features = {}
    
    match_features['Best_of'] = best_of
    match_features['Rank_1'] = player1_stats.get('_latest_rank', 50)
    match_features['Rank_2'] = player2_stats.get('_latest_rank', 50)
    match_features['Rank_Diff'] = match_features['Rank_2'] - match_features['Rank_1']
    match_features['Pts_1'] = player1_stats.get('_latest_pts', 0)
    match_features['Pts_2'] = player2_stats.get('_latest_pts', 0)
    match_features['Pts_Diff'] = match_features['Pts_1'] - match_features['Pts_2']
    
    for feat, val in player1_stats.items():
        if not feat.startswith('_'):
            match_features[f'P1_{feat}'] = val
    
    for feat, val in player2_stats.items():
        if not feat.startswith('_'):
            match_features[f'P2_{feat}'] = val
    
    match_features['Tournament_Encoded'] = 0
    
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    match_features['Surface_Encoded'] = surface_map.get(surface, 0)
    
    series_map = {'ATP250': 0, 'ATP500': 1, 'Grand Slam': 2, 'Masters 1000': 3}
    match_features['Series_Encoded'] = series_map.get(series, 0)
    
    round_map = {
        '1st Round': 0, '2nd Round': 1, '3rd Round': 2, '4th Round': 3,
        'Quarterfinals': 4, 'Semifinals': 5, 'The Final': 6
    }
    match_features['Round_Encoded'] = round_map.get(round_type, 0)
    
    court_map = {'Indoor': 0, 'Outdoor': 1}
    match_features['Court_Encoded'] = court_map.get(court, 0)
    
    match_features['Win_Pct_Diff'] = match_features.get('P1_Win_Pct', 0.5) - match_features.get('P2_Win_Pct', 0.5)
    match_features['Experience_Diff'] = match_features.get('P1_Total_Matches', 0) - match_features.get('P2_Total_Matches', 0)
    
    surface_feat_map = {
        'Hard': 'WinPct_Hard',
        'Clay': 'WinPct_Clay', 
        'Grass': 'WinPct_Grass',
        'Carpet': 'WinPct_Carpet'
    }
    surf_feat = surface_feat_map.get(surface, 'WinPct_Hard')
    match_features['Surface_Advantage'] = match_features.get(f'P1_{surf_feat}', 0) - match_features.get(f'P2_{surf_feat}', 0)
    
    if series == "Grand Slam":
        match_features['Series_Advantage'] = match_features.get('P1_WinPct_Grand_Slam', 0) - match_features.get('P2_WinPct_Grand_Slam', 0)
    elif series == "Masters 1000":
        match_features['Series_Advantage'] = match_features.get('P1_WinPct_Masters_1000', 0) - match_features.get('P2_WinPct_Masters_1000', 0)
    else:
        match_features['Series_Advantage'] = 0
    
    court_feat = 'WinPct_Indoor' if court == 'Indoor' else 'WinPct_Outdoor'
    match_features['Court_Advantage'] = match_features.get(f'P1_{court_feat}', 0) - match_features.get(f'P2_{court_feat}', 0)
    
    match_features['Odds_Diff'] = 0
    
    return match_features

def make_prediction(player1, player2, surface, round_type, series, court, best_of, historical_df, processed_df):
    """Make a prediction for a match"""
    player1_stats = extract_player_features(processed_df, historical_df, player1)
    player2_stats = extract_player_features(processed_df, historical_df, player2)
    
    if player1_stats is None:
        st.error(f"No historical data found for {player1}")
        return None
    if player2_stats is None:
        st.error(f"No historical data found for {player2}")
        return None
    
    match_features = build_match_features(
        player1_stats, player2_stats,
        surface, round_type, series, court, best_of
    )
    
    match_df = pd.DataFrame([match_features])
    
    model = load_model()
    if model is None:
        return None
    
    drop_cols = ['Date', 'Player_1', 'Player_2', 'Winner', 'timestamp', 
                 'Tournament', 'Surface', 'Series', 'Round', 'Court', 
                 'Tournament_Clean', 'Player_1_Won']
    odds_cols = [col for col in match_df.columns if 'odd' in col.lower() or 'Odd' in col]
    drop_cols.extend(odds_cols)
    
    X_pred = match_df.drop([col for col in drop_cols if col in match_df.columns], axis=1)
    
    expected_feature_order = [
        'Best_of', 'Rank_1', 'Rank_2', 'Rank_Diff', 'Pts_1', 'Pts_2', 'Pts_Diff',
        'P1_Total_Matches', 'P1_Wins', 'P1_Losses', 'P2_Total_Matches', 'P2_Wins', 'P2_Losses',
        'P1_Win_Pct', 'P2_Win_Pct',
        'P1_WinPct_Hard', 'P2_WinPct_Hard', 'P1_WinPct_Clay', 'P2_WinPct_Clay',
        'P1_WinPct_Grass', 'P2_WinPct_Grass', 'P1_WinPct_Carpet', 'P2_WinPct_Carpet',
        'P1_WinPct_ATP250', 'P2_WinPct_ATP250', 'P1_WinPct_ATP500', 'P2_WinPct_ATP500',
        'P1_WinPct_Grand_Slam', 'P2_WinPct_Grand_Slam',
        'P1_WinPct_Masters_1000', 'P2_WinPct_Masters_1000',
        'P1_WinPct_Indoor', 'P2_WinPct_Indoor', 'P1_WinPct_Outdoor', 'P2_WinPct_Outdoor',
        'Tournament_Encoded', 'Surface_Encoded', 'Series_Encoded', 'Round_Encoded', 'Court_Encoded',
        'Win_Pct_Diff', 'Experience_Diff', 'Surface_Advantage', 'Series_Advantage', 'Court_Advantage'
    ]
    
    available_features = [f for f in expected_feature_order if f in X_pred.columns]
    X_pred = X_pred[available_features]
    
    dmatrix = xgb.DMatrix(X_pred)
    prob_player1_wins = model.predict(dmatrix)[0]
    
    return {
        'player1': player1,
        'player2': player2,
        'prob_player1': prob_player1_wins,
        'prob_player2': 1 - prob_player1_wins,
        'predicted_winner': player1 if prob_player1_wins > 0.5 else player2,
        'confidence': max(prob_player1_wins, 1 - prob_player1_wins),
        'player1_stats': player1_stats,
        'player2_stats': player2_stats
    }

with tab1:
    st.header("Match Prediction")
    st.markdown("Select two players and match conditions to predict the winner.")
    
    historical_df, processed_df = load_and_process_historical_data()
    
    if historical_df is None or processed_df is None:
        st.error("Could not load historical data. Please check your internet connection.")
    else:
        players = sorted(set(historical_df['Player_1'].tolist() + historical_df['Player_2'].tolist()))
        
        st.success(f"Loaded {len(historical_df):,} historical matches with {len(players)} players")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Player 1")
            default_p1 = players.index("Djokovic N.") if "Djokovic N." in players else 0
            player1 = st.selectbox("Select Player 1", players, key="p1", index=default_p1)
        
        with col2:
            st.subheader("Player 2")
            available_p2 = [p for p in players if p != player1]
            default_p2 = available_p2.index("Alcaraz C.") if "Alcaraz C." in available_p2 else 0
            player2 = st.selectbox("Select Player 2", available_p2, key="p2", index=default_p2)
        
        st.markdown("---")
        
        st.subheader("Match Conditions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
        
        with col2:
            round_type = st.selectbox("Round", ["1st Round", "2nd Round", "3rd Round", "4th Round", 
                 "Quarterfinals", "Semifinals", "The Final"], index=6)
        
        with col3:
            best_of = st.selectbox("Best Of", [3, 5])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            series = st.selectbox("Tournament Series", ["Masters 1000", "ATP250", "ATP500", "Grand Slam"])
        
        with col2:
            court = st.selectbox("Court Type", ["Outdoor", "Indoor"])
        
        with col3:
            st.markdown("")
        
        st.markdown("---")
        
        if st.button("Predict Match Winner", type="primary", use_container_width=True):
            with st.spinner("Computing prediction..."):
                result = make_prediction(player1, player2, surface, round_type, series, court, best_of, historical_df, processed_df)
                
                if result:
                    st.markdown("### Prediction Result")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.metric("Predicted Winner", result['predicted_winner'],
                            delta="High Confidence" if result['confidence'] > 0.65 else "Low Confidence")
                    
                    with col2:
                        winner_prob = result['prob_player1'] if result['predicted_winner'] == player1 else result['prob_player2']
                        st.metric("Win Probability", f"{winner_prob*100:.1f}%")
                    
                    with col3:
                        if result['confidence'] > 0.70:
                            conf_level = "Strong"
                        elif result['confidence'] > 0.60:
                            conf_level = "Moderate"
                        else:
                            conf_level = "Weak"
                        st.metric("Confidence Level", conf_level)
                    
                    st.markdown("### Match-up Analysis")
                    
                    # Create simple bar chart showing win probabilities
                    prob_data = pd.DataFrame({
                        'Player': [player1, player2],
                        'Win Probability': [result['prob_player1'] * 100, result['prob_player2'] * 100]
                    })
                    
                    fig = px.bar(
                        prob_data,
                        x='Win Probability',
                        y='Player',
                        orientation='h',
                        text='Win Probability',
                        color='Win Probability',
                        color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
                        range_color=[0, 100]
                    )
                    
                    fig.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='inside',
                        textfont_size=16,
                        textfont_color='white'
                    )
                    
                    fig.update_layout(
                        height=250,
                        xaxis_title="Win Probability (%)",
                        yaxis_title="",
                        showlegend=False,
                        xaxis=dict(range=[0, 100], ticksuffix='%'),
                        margin=dict(l=20, r=20, t=20, b=20),
                        font=dict(size=14)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### Player Statistics Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{player1}**")
                        p1_stats = result['player1_stats']
                        st.metric("ATP Ranking", f"#{int(p1_stats.get('_latest_rank', 50))}")
                        st.metric("ATP Points", f"{int(p1_stats.get('_latest_pts', 0)):,}")
                        st.metric("Total Matches", f"{p1_stats.get('Total_Matches', 0):.0f}")
                        st.metric("Win Rate", f"{p1_stats.get('Win_Pct', 0)*100:.1f}%")
                        st.metric(f"{surface} Win %", f"{p1_stats.get(f'WinPct_{surface}', 0)*100:.1f}%")
                    
                    with col2:
                        st.markdown(f"**{player2}**")
                        p2_stats = result['player2_stats']
                        st.metric("ATP Ranking", f"#{int(p2_stats.get('_latest_rank', 50))}")
                        st.metric("ATP Points", f"{int(p2_stats.get('_latest_pts', 0)):,}")
                        st.metric("Total Matches", f"{p2_stats.get('Total_Matches', 0):.0f}")
                        st.metric("Win Rate", f"{p2_stats.get('Win_Pct', 0)*100:.1f}%")
                        st.metric(f"{surface} Win %", f"{p2_stats.get(f'WinPct_{surface}', 0)*100:.1f}%")
                    
                    st.markdown("### Match Details")
                    st.info(f"""
                    **Surface:** {surface}  
                    **Round:** {round_type}  
                    **Series:** {series}  
                    **Best Of:** {best_of} sets  
                    **Court:** {court}
                    """)
                    
                    st.markdown("### Betting Recommendation")
                    if result['confidence'] >= 0.65:
                        st.success(f"""
                        **RECOMMENDED BET**  
                        Based on backtest analysis, predictions with >=65% confidence have ~67% accuracy.  
                        **Suggested bet:** {result['predicted_winner']} to win
                        """)
                    else:
                        st.warning(f"""
                        **LOW CONFIDENCE - AVOID BETTING**  
                        Prediction confidence is below 65%. Historical data shows lower accuracy for such predictions.  
                        **Suggestion:** Skip this bet or wait for more information.
                        """)

with tab2:
    st.header("2025 Season Backtest")
    st.markdown("Model trained on 2000-2024, tested on 2025 matches")
    
    if Path("backtest_2025.csv").exists():
        backtest = pd.read_csv("backtest_2025.csv")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            accuracy = backtest["correct"].mean()
            st.metric("Prediction Accuracy", f"{accuracy:.1%}")
        
        with col2:
            final_profit = backtest["cumulative_profit"].iloc[-1]
            st.metric("All Bets Profit", f"${final_profit:+.2f}")
        
        with col3:
            smart_profit = backtest["smart_cumulative_profit"].iloc[-1]
            st.metric("Smart Betting Profit", f"${smart_profit:+.2f}")
        
        st.subheader("Cumulative Profit Over 2025 Season")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(backtest))),
            y=backtest["cumulative_profit"],
            name="All Bets",
            line=dict(color="blue", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(backtest))),
            y=backtest["smart_cumulative_profit"],
            name="Smart Betting (High Confidence)",
            line=dict(color="green", width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Match Number",
            yaxis_title="Profit ($)",
            hovermode="x unified",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent 2025 Matches")
        display_df = backtest[["Date", "Player_1", "Player_2", "Winner", "predicted_winner", "confidence", "correct"]].tail(20)
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.warning("Run `python 2_backtest_2025.py` to generate results")

with tab3:
    st.header("Betting Strategy Comparison")
    st.markdown("**2025 Season Results**: Testing two betting strategies on 5,030 matches")
    
    if Path("backtest_2025.csv").exists():
        backtest = pd.read_csv("backtest_2025.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("All Bets Strategy")
            st.markdown("**Approach**: Bet $1 on every match using bookmaker odds")
            
            total_bet = len(backtest) * 1
            final_profit = backtest["cumulative_profit"].iloc[-1]
            roi = (final_profit / total_bet) * 100
            wins = backtest["correct"].sum()
            losses = len(backtest) - wins
            
            st.metric("Total Matches", f"{len(backtest):,}")
            st.metric("Total Wagered", f"${total_bet:,}")
            st.metric("Wins / Losses", f"{wins:,} / {losses:,}")
            st.metric("Final Profit", f"${final_profit:+.2f}")
            st.metric("ROI", f"{roi:+.2f}%")
            
            st.markdown(f"""
            **Analysis:**
            - Prediction accuracy: **61.0%**
            - Despite good accuracy, bookmaker odds result in **-7.5% ROI**
            - Lost **$379** over the season
            - Bookmakers set odds to ensure they profit
            """)
        
        with col2:
            st.subheader("Smart Betting Strategy")
            st.markdown("**Approach**: Only bet when prediction confidence >= 65%")
            
            smart_bets = (backtest["smart_bet_amount"] > 0).sum()
            smart_wagered = smart_bets * 1
            smart_profit = backtest["smart_cumulative_profit"].iloc[-1]
            smart_roi = (smart_profit / smart_wagered) * 100 if smart_wagered > 0 else 0
            smart_accuracy = backtest[backtest["smart_bet_amount"] > 0]["correct"].mean()
            smart_wins = backtest[backtest["smart_bet_amount"] > 0]["correct"].sum()
            smart_losses = smart_bets - smart_wins
            
            st.metric("Total Bets", f"{smart_bets:,} ({smart_bets/len(backtest)*100:.1f}% of matches)")
            st.metric("Total Wagered", f"${smart_wagered:,}")
            st.metric("Wins / Losses", f"{smart_wins:,} / {smart_losses:,}")
            st.metric("Final Profit", f"${smart_profit:+.2f}")
            st.metric("ROI", f"{smart_roi:+.2f}%")
            
            st.markdown(f"""
            **Analysis:**
            - Prediction accuracy: **{smart_accuracy:.1%}** (6.4% improvement)
            - Bet on only **{smart_bets:,}** matches (58.5% selectivity)
            - Lost **$203** vs $379 (saved $176)
            - Better risk management by avoiding uncertain bets
            """)
        
        st.subheader("Win Rate by Confidence Level")
        
        backtest["confidence_bin"] = pd.cut(
            backtest["confidence"], 
            bins=[0.5, 0.6, 0.7, 0.8, 1.0],
            labels=["50-60%", "60-70%", "70-80%", "80%+"]
        )
        
        conf_stats = backtest.groupby("confidence_bin")["correct"].agg(["mean", "count"]).reset_index()
        conf_stats.columns = ["Confidence", "Win Rate", "Count"]
        
        fig = px.bar(
            conf_stats, 
            x="Confidence", 
            y="Win Rate",
            text="Count",
            title="Prediction Accuracy by Confidence Level",
            color="Win Rate",
            color_continuous_scale="RdYlGn"
        )
        fig.update_traces(texttemplate='%{text} matches', textposition='outside')
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Run `python 2_backtest_2025.py` first")

with tab4:
    st.header("Model Performance & Feature Importance")
    
    if Path("model_metrics.json").exists() and Path("feature_importance.csv").exists():
        with open("model_metrics.json", "r") as f:
            metrics = json.load(f)
        
        feature_importance = pd.read_csv("feature_importance.csv")
        
        st.subheader("Model Evaluation")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        with col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        st.markdown("""
        **Model Performance:**
        - **Accuracy > 60%** means the model correctly predicts match winners more often than not
        - **Precision & Recall ~61%** shows balanced performance in predicting wins and losses
        - **ROC-AUC > 0.66** indicates good ability to distinguish between wins and losses
        - Model is better than random guessing (50%) but bookmakers still have the edge on odds
        """)
        
        st.subheader("Top 20 Most Important Features")
        st.markdown("These features have the strongest influence on predictions:")
        
        top_features = feature_importance.head(20)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="What Drives Match Predictions?",
            color='importance',
            color_continuous_scale='Viridis',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=700,
            xaxis_title="Importance Score (Higher = More Influential)",
            yaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Key Insights:**
        - **{top_features.iloc[0]['feature']}** is the most important predictor (score: {top_features.iloc[0]['importance']:.1f})
        - **{top_features.iloc[1]['feature']}** is the second most important (score: {top_features.iloc[1]['importance']:.1f})
        - Ranking, win percentages, and surface advantages drive most predictions
        - The model uses {len(feature_importance)} total features
        """)
        
        with st.expander("View All Features & Importance Scores"):
            st.dataframe(feature_importance, use_container_width=True, height=400)
    
    else:
        st.warning("Run `python 2_backtest_2025.py` to generate model metrics")
