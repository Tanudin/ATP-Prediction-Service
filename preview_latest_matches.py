"""
Demo script to show what the new Latest 10 Matches section looks like.
This generates a preview of the data structure used in the Streamlit UI.
"""

import pandas as pd

# Load sample data
try:
    df = pd.read_csv("tennis_predictions.csv")
    print("🎾 ATP Prediction Service - Latest 10 Matches Preview\n")
    print("=" * 80)
    
    # Get unique matches (deduplicate)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    df_unique = df.drop_duplicates(subset=['date', 'player_1', 'player_2'], keep='first')
    
    # Get latest 10
    latest_10 = df_unique.head(10)
    
    print(f"\n📊 Showing {len(latest_10)} most recent matches:\n")
    
    for i, (idx, match) in enumerate(latest_10.iterrows(), 1):
        date = match['date'].strftime('%Y-%m-%d')
        p1 = match['player_1']
        p2 = match['player_2']
        pred = match['predicted_winner']
        conf = match.get('confidence', match.get('player_1_win_probability', 0.5))
        
        # Confidence level
        if conf > 0.65:
            conf_emoji = "🟢"
            conf_label = "HIGH"
        elif conf > 0.55:
            conf_emoji = "🟡"
            conf_label = "MEDIUM"
        else:
            conf_emoji = "🔴"
            conf_label = "LOW"
        
        # Odds
        odd_1 = match.get('odd_1', 'N/A')
        odd_2 = match.get('odd_2', 'N/A')
        
        # Expected Value
        try:
            if pred == p1:
                bet_odd = float(odd_1)
            else:
                bet_odd = float(odd_2)
            
            ev = (conf * (bet_odd - 1)) - ((1 - conf) * 1)
            ev_pct = ev * 100
            
            if ev_pct > 10:
                ev_str = f"🟢 +{ev_pct:.1f}% (Strong)"
            elif ev_pct > 0:
                ev_str = f"🟡 +{ev_pct:.1f}% (Slight)"
            else:
                ev_str = f"🔴 {ev_pct:.1f}% (No Value)"
        except:
            ev_str = "N/A"
        
        # Kelly bet size
        try:
            edge = (conf * bet_odd) - 1
            kelly_pct = edge / (bet_odd - 1) if bet_odd > 1 else 0
            frac_kelly = kelly_pct * 0.25
            suggested = min(frac_kelly * 100, 10.0)
            suggested = max(suggested, 0)
            bet_str = f"{suggested:.1f}% (${suggested:.2f})"
        except:
            bet_str = "Skip"
        
        # Actual result
        actual = match.get('actual_winner', '')
        correct = match.get('correct', None)
        if actual and correct is not None:
            result_emoji = "✅" if correct else "❌"
            result_str = f"{actual} {result_emoji}"
        else:
            result_str = "Pending"
        
        print(f"{conf_emoji} Match {i}: {date}")
        print(f"   Players:    {p1} vs {p2}")
        print(f"   Prediction: {pred} ({conf:.1%} {conf_label})")
        print(f"   Odds:       {p1}@{odd_1} | {p2}@{odd_2}")
        print(f"   EV:         {ev_str}")
        print(f"   Bet Size:   {bet_str}")
        print(f"   Result:     {result_str}")
        
        if 'bet_profit' in match and pd.notna(match['bet_profit']):
            profit = match['bet_profit']
            profit_emoji = "💰" if profit > 0 else "📉"
            print(f"   P/L:        {profit_emoji} ${profit:+.2f}")
        
        print()
    
    print("=" * 80)
    print("\n✅ Preview complete!")
    print("\nTo see this in the interactive Streamlit UI:")
    print("  streamlit run streamlit_app.py")
    print("\nThen navigate to: 🔮 Recent Predictions tab")
    
except FileNotFoundError:
    print("❌ tennis_predictions.csv not found!")
    print("\nGenerate predictions first:")
    print("  python tennis-inference.py")
except Exception as e:
    print(f"❌ Error: {e}")
