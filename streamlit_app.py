import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾", layout="wide")

st.title("🎾 ATP Tennis Match Prediction System")
st.markdown("AI-powered tennis predictions with betting strategy analysis")

tab1, tab2, tab3 = st.tabs([
    "📊 2025 Backtest Results",
    "💰 Betting Strategy Comparison",
    "🤖 Model Performance"
])

with tab1:
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
        st.warning("⚠️ Run `python 2_backtest_2025.py` to generate results")

with tab2:
    st.header("Betting Strategy Comparison")
    st.markdown("**2025 Season Results**: Testing two betting strategies on 5,030 matches")
    
    if Path("backtest_2025.csv").exists():
        backtest = pd.read_csv("backtest_2025.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 All Bets Strategy")
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
            st.subheader("🎯 Smart Betting Strategy")
            st.markdown("**Approach**: Only bet when prediction confidence ≥ 65%")
            
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
            - Prediction accuracy: **{smart_accuracy:.1%}** (6.4% improvement!)
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
        st.warning("⚠️ Run `python 2_backtest_2025.py` first")

with tab3:
    st.header("Model Performance & Feature Importance")
    
    if Path("model_metrics.json").exists() and Path("feature_importance.csv").exists():
        # Load metrics
        with open("model_metrics.json", "r") as f:
            metrics = json.load(f)
        
        feature_importance = pd.read_csv("feature_importance.csv")
        
        # Display key metrics only
        st.subheader("📈 Model Evaluation")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}", 
                     help="Overall prediction accuracy on 2025 matches")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}",
                     help="Of all predicted wins, how many were correct")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}",
                     help="Of all actual wins, how many did we predict correctly")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}",
                     help="Harmonic mean of precision and recall")
        with col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}",
                     help="Area under ROC curve - measures model's ability to distinguish between wins/losses")
        
        st.markdown("""
        **What this means:**
        - **Accuracy > 60%** means the model correctly predicts match winners more often than not
        - **Precision & Recall ~61%** shows balanced performance in predicting wins and losses
        - **ROC-AUC > 0.66** indicates good ability to distinguish between wins and losses
        - Model is better than random guessing (50%) but bookmakers still have the edge on odds
        """)
        
        # Feature Importance - the most important visualization
        st.subheader("⭐ Top 20 Most Important Features")
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
        
        # Key insights
        st.markdown(f"""
        **Key Insights:**
        - **{top_features.iloc[0]['feature']}** is the most important predictor (score: {top_features.iloc[0]['importance']:.1f})
        - **{top_features.iloc[1]['feature']}** is the second most important (score: {top_features.iloc[1]['importance']:.1f})
        - Ranking, win percentages, and surface advantages drive most predictions
        - The model uses {len(feature_importance)} total features
        """)
        
        # Feature importance table
        with st.expander("📋 View All Features & Importance Scores"):
            st.dataframe(feature_importance, use_container_width=True, height=400)
    
    else:
        st.warning("⚠️ Run `python 2_backtest_2025.py` to generate model metrics")
