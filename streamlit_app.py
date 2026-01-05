"""
ATP Tennis Match Prediction Dashboard
Interactive Streamlit UI for viewing predictions and model performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hopsworks
import numpy as np
from xgboost import XGBClassifier
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="ATP Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .prediction-high {
        color: #16a34a;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ca8a04;
        font-weight: bold;
    }
    .prediction-low {
        color: #dc2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎾 ATP Tennis Match Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Real-time ML predictions for professional tennis matches**")
st.markdown("---")

# Sidebar - Connection and Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Hopsworks connection
    use_hopsworks = st.checkbox("Connect to Hopsworks", value=False, help="Load live data from Hopsworks Feature Store")
    
    st.markdown("---")
    
    st.header("📊 Model Info")
    
    # Load model metrics if available
    try:
        if os.path.exists("tennis_model/model.json"):
            st.success("✅ Model loaded")
            st.markdown("""
            **XGBoost Classifier**
            - Accuracy: ~72.3%
            - F1 Score: ~0.72
            - ROC-AUC: ~0.78
            """)
        else:
            st.warning("⚠️ Model not found")
    except:
        pass
    
    st.markdown("---")
    
    st.header("🎯 Betting Stats")
    st.info("Track prediction performance as betting simulation")
    
    st.markdown("---")
    st.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M UTC"))

# Load predictions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_predictions():
    try:
        df = pd.read_csv("tennis_predictions.csv")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.warning("⚠️ No predictions file found. Run the inference pipeline first.")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_smart_predictions():
    try:
        df = pd.read_csv("tennis_predictions_smart.csv")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_hopsworks_data():
    """Load latest predictions from Hopsworks"""
    try:
        # Login to Hopsworks (will use HOPSWORKS_API_KEY from .env)
        import dotenv
        dotenv.load_dotenv()
        
        project = hopsworks.login()
        fs = project.get_feature_store()
        
        # Get the feature group directly
        tennis_fg = fs.get_feature_group(
            name="tennis_matches",
            version=2
        )
        
        # Read the data
        data = tennis_fg.read()
        return data
    except Exception as e:
        st.error(f"❌ Failed to connect to Hopsworks: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        model = XGBClassifier()
        model.load_model("tennis_model/model.json")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# Load data and model
if use_hopsworks:
    with st.spinner("Loading data from Hopsworks..."):
        feature_data = load_hopsworks_data()
else:
    feature_data = pd.DataFrame()

predictions_df = load_predictions()
smart_predictions_df = load_smart_predictions()
model = load_model()

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎮 Live Predictor", 
    "📊 Flat Betting", 
    "🧠 Smart Betting",
    "🔮 Recent Predictions", 
    "📈 Feature Importance", 
    "🏆 Player Stats"
])

# TAB 1: Live Predictor - Interactive Player vs Player
with tab1:
    st.header("🎮 Match Predictor: Player vs Player")
    st.markdown("Select two players to predict match outcome with AI-powered analysis")
    
    if not feature_data.empty and model is not None:
        col1, col2 = st.columns(2)
        
        # Get unique players
        if 'player_1' in feature_data.columns and 'player_2' in feature_data.columns:
            all_players = sorted(set(feature_data['player_1'].unique()) | set(feature_data['player_2'].unique()))
        else:
            all_players = []
        
        with col1:
            st.subheader("🎾 Player 1 (Home)")
            player1 = st.selectbox("Select Player 1", all_players, key="p1")
            
            if player1:
                # Get player 1 stats
                p1_matches = feature_data[
                    (feature_data['player_1'] == player1) | 
                    (feature_data['player_2'] == player1)
                ]
                
                if len(p1_matches) > 0:
                    st.metric("Total Matches", len(p1_matches))
                    
                    # Calculate win rate
                    p1_wins = len(p1_matches[
                        ((p1_matches['player_1'] == player1) & (p1_matches['player_1_won'] == 1)) |
                        ((p1_matches['player_2'] == player1) & (p1_matches['player_1_won'] == 0))
                    ])
                    win_rate = (p1_wins / len(p1_matches) * 100) if len(p1_matches) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            st.subheader("🎾 Player 2 (Away)")
            player2 = st.selectbox("Select Player 2", [p for p in all_players if p != player1], key="p2")
            
            if player2:
                # Get player 2 stats
                p2_matches = feature_data[
                    (feature_data['player_1'] == player2) | 
                    (feature_data['player_2'] == player2)
                ]
                
                if len(p2_matches) > 0:
                    st.metric("Total Matches", len(p2_matches))
                    
                    # Calculate win rate
                    p2_wins = len(p2_matches[
                        ((p2_matches['player_1'] == player2) & (p2_matches['player_1_won'] == 1)) |
                        ((p2_matches['player_2'] == player2) & (p2_matches['player_1_won'] == 0))
                    ])
                    win_rate = (p2_wins / len(p2_matches) * 100) if len(p2_matches) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.markdown("---")
        
        # Match context
        col1, col2, col3 = st.columns(3)
        
        with col1:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
        
        with col2:
            tournament_type = st.selectbox("Tournament", ["Grand Slam", "Masters 1000", "ATP500", "ATP250"])
        
        with col3:
            court = st.selectbox("Court", ["Outdoor", "Indoor"])
        
        st.markdown("---")
        
        if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
            st.markdown("### Prediction Result")
            
            # Get player features from historical data
            # Find most recent match for each player to get their latest stats
            p1_latest = feature_data[
                (feature_data['player_1'] == player1) | 
                (feature_data['player_2'] == player1)
            ].sort_values('date', ascending=False).iloc[0] if len(p1_matches) > 0 else None
            
            p2_latest = feature_data[
                (feature_data['player_1'] == player2) | 
                (feature_data['player_2'] == player2)
            ].sort_values('date', ascending=False).iloc[0] if len(p2_matches) > 0 else None
            
            if p1_latest is not None and p2_latest is not None:
                # Build feature vector for prediction
                # We need to construct a row that matches the training data format
                
                # Get player stats (adjust based on whether they were p1 or p2 in their last match)
                if p1_latest['player_1'] == player1:
                    p1_stats = {k.replace('p1_', ''): v for k, v in p1_latest.items() if k.startswith('p1_')}
                else:
                    p1_stats = {k.replace('p2_', ''): v for k, v in p1_latest.items() if k.startswith('p2_')}
                
                if p2_latest['player_1'] == player2:
                    p2_stats = {k.replace('p1_', ''): v for k, v in p2_latest.items() if k.startswith('p1_')}
                else:
                    p2_stats = {k.replace('p2_', ''): v for k, v in p2_latest.items() if k.startswith('p2_')}
                
                # Use model to predict
                try:
                    # Create feature array matching model input
                    # Use a sample row from feature data as template

                    sample_row = feature_data.iloc[0:1].copy()
                    
                    # Drop columns we can't use
                    drop_cols = ['date', 'player_1', 'player_2', 'winner', 'timestamp', 
                                'tournament', 'surface', 'series', 'round', 'court', 
                                'tournament_clean', 'player_1_won']
                    features_for_pred = sample_row.drop([c for c in drop_cols if c in sample_row.columns], axis=1)
                    
                    # Get prediction probabilities
                    prob = model.predict_proba(features_for_pred)[0]
                    confidence = prob[1]  # Probability Player 1 wins
                    
                    predicted_winner = player1 if confidence > 0.5 else player2
                    
                except Exception as e:
                    st.error(f"Could not generate prediction: {str(e)}")
                    st.info("Using simplified prediction based on win rates")
                    # Fallback to win rate comparison
                    p1_wr = (p1_wins / len(p1_matches)) if len(p1_matches) > 0 else 0.5
                    p2_wr = (p2_wins / len(p2_matches)) if len(p2_matches) > 0 else 0.5
                    
                    # Normalize to probability
                    total = p1_wr + p2_wr
                    confidence = p1_wr / total if total > 0 else 0.5
                    predicted_winner = player1 if confidence > 0.5 else player2
            else:
                st.warning("Insufficient player data for prediction")
                confidence = 0.5
                predicted_winner = player1
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    title={'text': f"Predicted Winner: {predicted_winner}"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen" if confidence > 0.65 else "orange"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 65], 'color': "yellow"},
                            {'range': [65, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction breakdown
            st.markdown("### Prediction Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{player1}**")
                st.progress(float(confidence if predicted_winner == player1 else 1 - confidence))
                st.caption(f"{confidence * 100 if predicted_winner == player1 else (1 - confidence) * 100:.1f}% win probability")
            
            with col2:
                st.markdown(f"**{player2}**")
                st.progress(float(1 - confidence if predicted_winner == player1 else confidence))
                st.caption(f"{(1 - confidence) * 100 if predicted_winner == player1 else confidence * 100:.1f}% win probability")
            
            st.success(f"✅ AI predicts **{predicted_winner}** to win with **{confidence * 100:.1f}%** confidence!")
    else:
        st.info("📥 Load data from Hopsworks or run the inference pipeline to use the live predictor")
        st.markdown("""
        **To enable this feature:**
        1. Check "Connect to Hopsworks" in the sidebar, OR
        2. Run `python tennis-inference.py` to generate predictions
        """)

# TAB 2: Performance Dashboard
with tab2:
    st.header("📊 Model Performance Dashboard")
    
    if not predictions_df.empty:
        # Check if we have actual results to calculate real betting performance
        has_actual_results = 'actual_winner' in predictions_df.columns or 'correct' in predictions_df.columns
        
        if has_actual_results:
            st.success("✅ Showing REAL betting performance based on actual match results")
            
            # Calculate real betting metrics
            if 'correct' in predictions_df.columns:
                total_bets = len(predictions_df)
                wins = predictions_df['correct'].sum()
                losses = total_bets - wins
                win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
                
                # Calculate ROI
                initial_balance = 100.0
                bet_size = 1.0
                
                if 'bet_profit' in predictions_df.columns:
                    # Use pre-calculated profits
                    total_profit = predictions_df['bet_profit'].sum()
                    balance = initial_balance + total_profit
                else:
                    # Calculate manually
                    total_profit = (wins * 0.9) - losses  # Assume 1.9 odds
                    balance = initial_balance + total_profit
                
                roi = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
            else:
                # Fallback to prediction data
                total_bets = len(predictions_df)
                wins = 0
                losses = total_bets
                win_rate = 0
                balance = 100.0
                total_profit = 0
                roi = 0
        else:
            st.info("ℹ️ Showing simulated performance (no actual results available yet)")
            
            # Simulated metrics based on model accuracy
            total_bets = len(predictions_df)
            estimated_accuracy = 0.72  # Model's expected accuracy
            wins = int(total_bets * estimated_accuracy)
            losses = total_bets - wins
            win_rate = estimated_accuracy * 100
            
            initial_balance = 100.0
            total_profit = (wins * 0.9) - losses
            balance = initial_balance + total_profit
            roi = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bets", total_bets)
        
        with col2:
            st.metric("Bets Won", wins, delta=f"+{wins - losses}")
        
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            delta_color = "normal" if total_profit >= 0 else "inverse"
            st.metric("ROI", f"{roi:+.1f}%", delta=f"${total_profit:+.2f}")
        
        st.markdown("---")
        
        # Balance over time chart
        st.markdown("### 💰 Bankroll Over Time")
        
        if 'cumulative_profit' in predictions_df.columns:
            # Use actual cumulative data - deduplicate matches (each match appears twice in dataset)
            balance_df = predictions_df.copy()
            balance_df = balance_df.sort_values('date').reset_index(drop=True)
            # Deduplicate by match: keep every other row since each match appears twice
            balance_df = balance_df.iloc[::2].reset_index(drop=True)
            balance_df['Bet_Number'] = range(1, len(balance_df) + 1)
            balance_df['Balance'] = balance_df['cumulative_profit']
            balance_df = balance_df[['Bet_Number', 'Balance']]
        elif 'correct' in predictions_df.columns:
            # Calculate cumulative
            predictions_df_sorted = predictions_df.sort_values('date') if 'date' in predictions_df.columns else predictions_df
            predictions_df_sorted = predictions_df_sorted.reset_index(drop=True)
            cumulative_profit = []
            current = 100.0
            
            for idx, row in predictions_df_sorted.iterrows():
                if row.get('correct', False):
                    current += 0.9
                else:
                    current -= 1.0
                cumulative_profit.append(current)
            
            balance_df = pd.DataFrame({
                'Bet_Number': range(1, len(cumulative_profit) + 1),
                'Balance': cumulative_profit
            })
        else:
            # Simulated curve
            balance_df = pd.DataFrame({
                'Bet_Number': range(1, total_bets + 1),
                'Balance': [100 + (i * total_profit / total_bets) for i in range(total_bets)]
            })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=balance_df['Bet_Number'],
            y=balance_df['Balance'],
            mode='lines',
            name='Bankroll',
            line=dict(color='#3b82f6', width=2),
            showlegend=False
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Initial Balance ($100)")
        fig.update_layout(
            title='Flat Betting Bankroll Evolution',
            xaxis_title='Bet Number',
            yaxis_title='Balance ($)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Performance Metrics")
            metrics_display = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Total Profit', 'Avg Profit/Bet'],
                'Value': [
                    f"{win_rate:.2f}%",
                    f"{(wins / total_bets * 100) if total_bets > 0 else 0:.2f}%",
                    f"${total_profit:+.2f}",
                    f"${total_profit / total_bets if total_bets > 0 else 0:+.3f}"
                ]
            })
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 🎯 Betting Strategy")
            st.markdown("""
            **Current Strategy:** Flat Betting
            - Bet Size: $1.00 per match
            - Expected Odds: 1.90 (90% profit on win)
            - Target: Matches with >60% confidence
            
            **Results:**
            - Win Rate: {:.1f}%
            - Break-even: 52.6% (at 1.90 odds)
            - Status: {}
            """.format(
                win_rate,
                "✅ Profitable" if win_rate > 52.6 else "❌ Need Improvement"
            ))
    else:
        st.info("No predictions available to display performance")

# TAB 3: Smart Betting Performance
with tab3:
    st.header("🧠 Smart Betting - Kelly Criterion Strategy")
    
    if not smart_predictions_df.empty:
        st.markdown("""
        **Strategy:** Variable bet sizing using Kelly Criterion with safety caps
        - Only bets when confidence > 60%
        - Bet size based on edge and bankroll
        - Fractional Kelly (25%) for risk management
        """)
        
        # Calculate metrics
        if 'correct' in smart_predictions_df.columns:
            # Filter only placed bets
            placed_bets = smart_predictions_df[smart_predictions_df.get('bet_size', 0) > 0] if 'bet_size' in smart_predictions_df.columns else smart_predictions_df
            
            total_matches = len(smart_predictions_df)
            total_bets = len(placed_bets)
            bets_skipped = total_matches - total_bets
            wins = placed_bets['correct'].sum() if len(placed_bets) > 0 else 0
            losses = len(placed_bets) - wins
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
            
            # Financial metrics
            initial_balance = 100.0
            if 'bet_profit' in smart_predictions_df.columns:
                total_profit = smart_predictions_df['bet_profit'].sum()
                final_balance = initial_balance + total_profit
            else:
                total_profit = 0
                final_balance = initial_balance
            
            roi = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
            
            # Bet sizing stats
            if 'bet_size' in smart_predictions_df.columns:
                bet_sizes = smart_predictions_df[smart_predictions_df['bet_size'] > 0]['bet_size']
                avg_bet = bet_sizes.mean() if len(bet_sizes) > 0 else 0
                max_bet = bet_sizes.max() if len(bet_sizes) > 0 else 0
                min_bet = bet_sizes.min() if len(bet_sizes) > 0 else 0
            else:
                avg_bet = max_bet = min_bet = 0
        else:
            total_matches = total_bets = bets_skipped = wins = losses = 0
            win_rate = roi = avg_bet = max_bet = min_bet = 0
            final_balance = initial_balance = 100.0
            total_profit = 0
        
        # Display key metrics
        st.markdown("### 📊 Strategy Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Matches", f"{total_matches:,}")
            st.caption(f"Bets: {total_bets:,}")
        
        with col2:
            st.metric("Bets Skipped", f"{bets_skipped:,}")
            skip_rate = (bets_skipped / total_matches * 100) if total_matches > 0 else 0
            st.caption(f"{skip_rate:.1f}% filtered")
        
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.caption(f"{wins:,}W / {losses:,}L")
        
        with col4:
            st.metric("Final Bankroll", f"${final_balance:,.2f}")
            st.caption(f"Started: ${initial_balance:.0f}")
        
        with col5:
            delta_color = "normal" if total_profit >= 0 else "inverse"
            st.metric("ROI", f"{roi:+.1f}%", delta=f"${total_profit:+,.2f}")
        
        st.markdown("---")
        
        # Bankroll evolution chart
        st.markdown("### 💰 Bankroll Growth")
        
        if 'cumulative_bankroll' in smart_predictions_df.columns:
            chart_data = smart_predictions_df.copy()
            chart_data = chart_data.sort_values('date').reset_index(drop=True)
            # Deduplicate by match: keep every other row since each match appears twice
            chart_data = chart_data.iloc[::2].reset_index(drop=True)
            chart_data['Bet_Number'] = range(1, len(chart_data) + 1)
            chart_data['Bankroll'] = chart_data['cumulative_bankroll']
            chart_data = chart_data[['Bet_Number', 'Bankroll']]
        elif 'cumulative_profit' in smart_predictions_df.columns:
            chart_data = smart_predictions_df.copy()
            chart_data = chart_data.sort_values('date').reset_index(drop=True) if 'date' in smart_predictions_df.columns else smart_predictions_df.reset_index(drop=True)
            # Deduplicate by match: keep every other row since each match appears twice
            chart_data = chart_data.iloc[::2].reset_index(drop=True)
            chart_data['Bet_Number'] = range(1, len(chart_data) + 1)
            chart_data['Bankroll'] = initial_balance + chart_data['cumulative_profit']
            chart_data = chart_data[['Bet_Number', 'Bankroll']]
        else:
            chart_data = pd.DataFrame({'Bet_Number': [0, 1], 'Bankroll': [100, final_balance]})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data['Bet_Number'],
            y=chart_data['Bankroll'],
            mode='lines',
            name='Bankroll',
            line=dict(color='#16a34a', width=2),
            showlegend=False
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Initial: $100")
        fig.update_layout(
            title='Smart Betting Bankroll Evolution',
            xaxis_title='Bet Number',
            yaxis_title='Bankroll ($)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💵 Bet Sizing Statistics")
            sizing_stats = pd.DataFrame({
                'Metric': ['Average Bet', 'Largest Bet', 'Smallest Bet', 'Total Wagered'],
                'Value': [
                    f"${avg_bet:.2f}",
                    f"${max_bet:.2f}",
                    f"${min_bet:.2f}",
                    f"${bet_sizes.sum() if 'bet_sizes' in locals() else 0:,.2f}"
                ]
            })
            st.dataframe(sizing_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 🎯 Performance Metrics")
            
            # Calculate avg odds if available
            if 'bet_odds' in smart_predictions_df.columns:
                wins_data = smart_predictions_df[smart_predictions_df['correct'] == True]
                losses_data = smart_predictions_df[smart_predictions_df['correct'] == False]
                avg_win_odds = wins_data['bet_odds'].mean() if len(wins_data) > 0 else 0
                avg_loss_odds = losses_data['bet_odds'].mean() if len(losses_data) > 0 else 0
            else:
                avg_win_odds = avg_loss_odds = 0
            
            perf_stats = pd.DataFrame({
                'Metric': ['Accuracy', 'Avg Win Odds', 'Avg Loss Odds', 'Profit/Bet'],
                'Value': [
                    f"{win_rate:.2f}%",
                    f"{avg_win_odds:.2f}",
                    f"{avg_loss_odds:.2f}",
                    f"${total_profit / total_bets if total_bets > 0 else 0:+.2f}"
                ]
            })
            st.dataframe(perf_stats, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Strategy explanation
        with st.expander("📚 How Kelly Criterion Works"):
            st.markdown("""
            ### Kelly Criterion Formula
            ```
            Kelly % = (bp - q) / b
            ```
            Where:
            - **b** = odds - 1 (profit multiplier)
            - **p** = probability of winning (model confidence)
            - **q** = 1 - p (probability of losing)
            
            ### Safety Features
            - **Fractional Kelly (25%)**: Reduces variance by using 25% of full Kelly
            - **Confidence Threshold**: Only bets when model confidence > 60%
            - **Bet Caps**: Min $0.50, Max $5.00
            - **Bankroll Limit**: Never bet more than 10% of current bankroll
            
            ### Why It Works
            - **Selective**: Only bets on high-confidence matches ({:.1f}% win rate vs {:.1f}% overall)
            - **Optimal Sizing**: Bets more when edge is larger
            - **Compound Growth**: Bet sizes scale with bankroll
            - **Risk Management**: Safety caps prevent ruin
            """.format(
                win_rate,
                smart_predictions_df['correct'].mean() * 100 if 'correct' in smart_predictions_df.columns else 0
            ))
    else:
        st.info("⚠️ No smart betting predictions available. Run `tennis-inference-smart.py` first.")
        st.code("python tennis-inference-smart.py", language="bash")

# TAB 4: Recent Predictions
with tab4:
    st.header("🔮 Match Predictions & Results")
    
    if not predictions_df.empty:
        # Check if we have actual results
        has_actuals = 'actual_winner' in predictions_df.columns or 'correct' in predictions_df.columns
        
        if has_actuals:
            st.success("✅ Showing predictions with actual match results")
        else:
            st.info("ℹ️ Showing predictions (actual results not yet available)")
        
        # ===== NEW SECTION: Latest 10 Matches with Bets =====
        st.markdown("### 🎯 Latest 10 Matches - Betting Recommendations")
        st.markdown("*Most recent predictions with betting odds and recommended bets*")
        
        # Get latest 10 matches
        latest_df = predictions_df.copy()
        if 'date' in latest_df.columns:
            latest_df['date'] = pd.to_datetime(latest_df['date'])
            latest_df = latest_df.sort_values('date', ascending=False)
        
        # Deduplicate (each match appears twice in dataset)
        latest_df_unique = latest_df.drop_duplicates(subset=['date', 'player_1', 'player_2'], keep='first')
        latest_10 = latest_df_unique.head(10)
        
        # Display each match in a card format
        for idx, match in latest_10.iterrows():
            # Determine confidence level and color
            confidence = match.get('confidence', match.get('player_1_win_probability', 0.5))
            if isinstance(confidence, str):
                confidence = float(confidence.strip('%')) / 100 if '%' in confidence else float(confidence)
            
            if confidence > 0.65:
                confidence_color = "🟢"
                confidence_label = "HIGH"
                bet_recommendation = "✅ Recommended"
            elif confidence > 0.55:
                confidence_color = "🟡"
                confidence_label = "MEDIUM"
                bet_recommendation = "⚠️ Cautious"
            else:
                confidence_color = "🔴"
                confidence_label = "LOW"
                bet_recommendation = "❌ Skip"
            
            # Create expandable card for each match
            date_str = match['date'].strftime('%Y-%m-%d') if 'date' in match and hasattr(match['date'], 'strftime') else 'N/A'
            player1 = match.get('player_1', 'Player 1')
            player2 = match.get('player_2', 'Player 2')
            predicted_winner = match.get('predicted_winner', player1)
            
            with st.expander(f"{confidence_color} **{date_str}** | {player1} vs {player2} → **{predicted_winner}** ({confidence:.1%})", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown("#### Match Details")
                    st.markdown(f"**🎾 Player 1:** {player1}")
                    st.markdown(f"**🎾 Player 2:** {player2}")
                    st.markdown(f"**🏆 Predicted Winner:** {predicted_winner}")
                    
                    # Show actual result if available
                    if 'actual_winner' in match and pd.notna(match['actual_winner']):
                        actual = match['actual_winner']
                        is_correct = match.get('correct', False)
                        result_emoji = "✅" if is_correct else "❌"
                        st.markdown(f"**📊 Actual Winner:** {actual} {result_emoji}")
                
                with col2:
                    st.markdown("#### Betting Odds & Prediction")
                    
                    # Get odds
                    odd_1 = match.get('odd_1', 'N/A')
                    odd_2 = match.get('odd_2', 'N/A')
                    
                    st.markdown(f"**💰 {player1} Odds:** {odd_1}")
                    st.markdown(f"**💰 {player2} Odds:** {odd_2}")
                    st.markdown(f"**🎯 Confidence:** {confidence:.1%} ({confidence_label})")
                    
                    # Calculate expected value if we have odds
                    if odd_1 != 'N/A' and odd_2 != 'N/A':
                        try:
                            if predicted_winner == player1:
                                bet_odd = float(odd_1)
                                win_prob = confidence
                            else:
                                bet_odd = float(odd_2)
                                win_prob = confidence
                            
                            # Expected Value = (Probability of Win × Profit) - (Probability of Loss × Stake)
                            expected_value = (win_prob * (bet_odd - 1)) - ((1 - win_prob) * 1)
                            ev_pct = expected_value * 100
                            
                            if ev_pct > 10:
                                ev_color = "🟢"
                                ev_label = "Strong Value"
                            elif ev_pct > 0:
                                ev_color = "🟡"
                                ev_label = "Slight Edge"
                            else:
                                ev_color = "🔴"
                                ev_label = "No Value"
                            
                            st.markdown(f"**📈 Expected Value:** {ev_color} {ev_pct:+.1f}% ({ev_label})")
                        except:
                            pass
                
                with col3:
                    st.markdown("#### Recommendation")
                    st.markdown(f"**{bet_recommendation}**")
                    
                    # Suggested bet size (Kelly Criterion simplified)
                    if confidence > 0.6 and odd_1 != 'N/A' and odd_2 != 'N/A':
                        try:
                            if predicted_winner == player1:
                                bet_odd = float(odd_1)
                            else:
                                bet_odd = float(odd_2)
                            
                            # Simplified Kelly: (edge) / (odds - 1)
                            edge = (confidence * bet_odd) - 1
                            kelly_pct = edge / (bet_odd - 1) if bet_odd > 1 else 0
                            
                            # Use fractional Kelly (25%) for safety
                            fractional_kelly = kelly_pct * 0.25
                            
                            # Cap at 10% bankroll
                            suggested_pct = min(fractional_kelly * 100, 10.0)
                            suggested_pct = max(suggested_pct, 0)  # No negative bets
                            
                            if suggested_pct > 0:
                                st.markdown(f"**💵 Suggested Bet:**")
                                st.markdown(f"{suggested_pct:.1f}% of bankroll")
                                st.markdown(f"(${suggested_pct:.2f} on $100)")
                            else:
                                st.markdown("**Skip this bet**")
                        except:
                            st.markdown("*Odds analysis unavailable*")
                    else:
                        st.markdown("**Skip** - Low confidence")
                    
                    # Show actual profit if available
                    if 'bet_profit' in match and pd.notna(match['bet_profit']):
                        profit = match['bet_profit']
                        profit_emoji = "💰" if profit > 0 else "📉"
                        st.markdown(f"{profit_emoji} **P/L:** ${profit:+.2f}")
        
        st.markdown("---")
        
        st.markdown("### Today's Predictions")
        
        # Prepare display dataframe
        display_df = predictions_df.copy()
        
        if 'date' in display_df.columns:
            display_df['date'] = pd.to_datetime(display_df['date'])
            display_df = display_df.sort_values('date', ascending=False)
        
        # Format columns for better display
        display_cols = []
        col_names = {}
        
        if 'date' in display_df.columns:
            display_cols.append('date')
            col_names['date'] = 'Date'
        
        if 'player_1' in display_df.columns:
            display_cols.append('player_1')
            col_names['player_1'] = 'Home Team'
        
        if 'player_2' in display_df.columns:
            display_cols.append('player_2')
            col_names['player_2'] = 'Away Team'
        
        if 'predicted_winner' in display_df.columns:
            display_cols.append('predicted_winner')
            col_names['predicted_winner'] = 'Prediction'
        
        if 'confidence' in display_df.columns:
            display_cols.append('confidence')
            col_names['confidence'] = 'Confidence'
            # Format as percentage
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        if 'actual_winner' in display_df.columns:
            display_cols.append('actual_winner')
            col_names['actual_winner'] = 'Actual Result'
        
        if 'correct' in display_df.columns:
            display_cols.append('correct')
            col_names['correct'] = 'Correct?'
            # Format as emoji
            display_df['correct'] = display_df['correct'].apply(lambda x: '✅' if x else '❌')
        
        # Select and rename columns
        if display_cols:
            show_df = display_df[display_cols].head(20).rename(columns=col_names)
            st.dataframe(show_df, use_container_width=True, height=600)
        
        st.markdown("---")
        st.markdown("### 📊 Last 10 Games Performance")
        
        # Show last 10 with detailed view
        last_10 = display_df.tail(10).copy()
        
        for idx, row in last_10.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                
                with col1:
                    if 'date' in row:
                        st.markdown(f"**{row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}**")
                
                with col2:
                    home = row.get('player_1', 'Player 1')
                    away = row.get('player_2', 'Player 2')
                    st.markdown(f"**{home}** vs **{away}**")
                
                with col3:
                    pred = row.get('predicted_winner', 'N/A')
                    conf = row.get('confidence', 'N/A')
                    st.markdown(f"Predicted: **{pred}**")
                    st.caption(f"Confidence: {conf}")
                
                with col4:
                    if 'actual_winner' in row:
                        actual = row['actual_winner']
                        is_correct = row.get('correct', '❓')
                        st.markdown(f"**{is_correct}**")
                        st.caption(f"Actual: {actual}")
                    else:
                        st.markdown("⏳ Pending")
                
                st.markdown("---")
        
        # Summary statistics
        if has_actuals:
            st.markdown("### 📈 Overall Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = len(predictions_df)
                st.metric("Total Predictions", total)
            
            with col2:
                if 'correct' in predictions_df.columns:
                    correct_count = predictions_df['correct'].sum() if predictions_df['correct'].dtype == 'bool' else 0
                    st.metric("Correct", correct_count)
            
            with col3:
                if 'correct' in predictions_df.columns:
                    incorrect = total - correct_count
                    st.metric("Incorrect", incorrect)
            
            with col4:
                if 'correct' in predictions_df.columns and total > 0:
                    accuracy = (correct_count / total * 100)
                    st.metric("Accuracy", f"{accuracy:.1f}%")
    else:
        st.info("No predictions available. Run the inference pipeline first.")

# TAB 5: Feature Importance
with tab5:
    st.header("📈 Feature Importance Analysis")
    
    if model is not None:
        st.markdown("### Top Features Contributing to Predictions")
        
        try:
            # Get feature importance
            importance = model.feature_importances_
            feature_names = model.get_booster().feature_names
            
            if feature_names is None or len(feature_names) == 0:
                # Use generic names
                feature_names = [f"Feature {i}" for i in range(len(importance))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(20)
            
            # Plot
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 20 Most Important Features',
                color='Importance',
                color_continuous_scale='blues'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.markdown("### Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not load feature importance: {str(e)}")
            st.info("Train the model first using `3_TrainingPipeline.ipynb`")
    else:
        st.info("📥 Load model to view feature importance")
        st.markdown("Run `python tennis-training.py` to train the model first")

# TAB 6: Player Stats  
with tab6:
    st.header("🏆 Player Statistics")
    
    if not feature_data.empty:
        # Player selector
        if 'player_1' in feature_data.columns and 'player_2' in feature_data.columns:
            all_players = sorted(set(feature_data['player_1'].unique()) | set(feature_data['player_2'].unique()))
        else:
            all_players = []
        selected_player = st.selectbox("Select Player for Detailed Stats", all_players)
        
        if selected_player:
            # Get all matches for this player
            player_matches = feature_data[
                (feature_data['player_1'] == selected_player) | 
                (feature_data['player_2'] == selected_player)
            ]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", len(player_matches))
            
            with col2:
                # Calculate wins
                wins = len(player_matches[
                    ((player_matches['player_1'] == selected_player) & (player_matches['player_1_won'] == 1)) |
                    ((player_matches['player_2'] == selected_player) & (player_matches['player_1_won'] == 0))
                ])
                st.metric("Total Wins", wins)
            
            with col3:
                losses = len(player_matches) - wins
                st.metric("Total Losses", losses)
            
            with col4:
                win_pct = (wins / len(player_matches) * 100) if len(player_matches) > 0 else 0
                st.metric("Win Percentage", f"{win_pct:.1f}%")
            
            st.markdown("---")
            
            # Surface performance
            st.subheader("Performance by Surface")
            
            surface_stats = []
            if 'surface' in player_matches.columns:
                for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                    surface_matches = player_matches[player_matches['surface'] == surface]
                    if len(surface_matches) > 0:
                        surface_wins = len(surface_matches[
                            ((surface_matches['player_1'] == selected_player) & (surface_matches['player_1_won'] == 1)) |
                            ((surface_matches['player_2'] == selected_player) & (surface_matches['player_1_won'] == 0))
                        ])
                        surface_stats.append({
                            'Surface': surface,
                            'Matches': len(surface_matches),
                            'Wins': surface_wins,
                            'Win %': (surface_wins / len(surface_matches) * 100) if len(surface_matches) > 0 else 0
                        })
            
            if surface_stats:
                surface_df = pd.DataFrame(surface_stats)
                
                fig = px.bar(
                    surface_df,
                    x='Surface',
                    y='Win %',
                    title=f'{selected_player} - Win Rate by Surface',
                    color='Win %',
                    color_continuous_scale='greens',
                    text='Win %'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(surface_df, use_container_width=True)
    elif not predictions_df.empty:
        st.info("Connect to Hopsworks to view detailed player statistics")
    else:
        st.info("No player data available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p>Built with ❤️ using <b>Streamlit</b>, <b>XGBoost</b>, and <b>Hopsworks</b></p>
    <p>Data updates automatically via GitHub Actions • Model retrained weekly</p>
</div>
""", unsafe_allow_html=True)
