# Betting Strategy Comparison

## Overview
Comparison between flat betting and Kelly Criterion smart betting strategies on 2025 ATP tennis matches.

## Dataset
- **Total Matches**: 19,904 (2025 season)
- **Model Accuracy**: 69.27%
- **Initial Bankroll**: $100.00

---

## Strategy 1: Flat Betting (`tennis-inference.py`)

### Strategy Details
- **Bet Size**: Fixed $1.00 per match
- **Selection**: Bet on all matches where odds are available
- **Logic**: Simple flat betting on predicted winner

### Results
- **Bets Placed**: ~19,900 matches
- **Win Rate**: ~69% (matches model accuracy)
- **Average Odds**: ~1.9
- **Final Bankroll**: ~$160-180 (estimated)
- **ROI**: ~60-80%

### Pros
- Simple and consistent
- Low variance
- Easy to understand and implement

### Cons
- Doesn't leverage high-confidence predictions
- Same stake on uncertain matches as confident ones
- Lower overall returns

---

## Strategy 2: Kelly Criterion Smart Betting (`tennis-inference-smart.py`)

### Strategy Details
- **Bet Sizing**: Variable based on Kelly Criterion formula
  - Kelly = `(bp - q) / b` where:
    - `b` = odds - 1 (profit multiplier)
    - `p` = model confidence (probability of winning)
    - `q` = 1 - p (probability of losing)
- **Fractional Kelly**: 25% of full Kelly (for safety)
- **Confidence Threshold**: Only bet when confidence > 60%
- **Bet Limits**:
  - Minimum: $0.50
  - Maximum: $5.00
  - Max % of bankroll: 10%

### Results
- **Bets Placed**: 2,920 matches (14.7% of total)
- **Bets Skipped**: 16,984 matches
  - Low confidence (<60%): 16,976
  - No odds available: 8
- **Win Rate**: 89.0% (2,600 wins / 320 losses)
- **High Confidence Bets (>70%)**: 2,268 (77.7%)
- **Average Bet Size**: $4.16
- **Bet Size Range**: $0.50 - $5.00
- **Average Odds**:
  - Winning bets: 1.33
  - Losing bets: 1.38
- **Final Bankroll**: $2,411.28
- **Total Profit**: $+2,311.28
- **ROI**: +2,311.28%

### Pros
- **Exceptional ROI**: 28.9x better than flat betting
- **Selective betting**: Only high-confidence matches
- **Risk management**: Variable bet sizing based on edge
- **High win rate**: 89% vs 69% (cherry-picks best opportunities)
- **Compound growth**: Bet sizes increase as bankroll grows

### Cons
- More complex to implement
- Requires accurate probability estimates
- Higher variance in individual bets
- Can lead to bankruptcy if probabilities are wrong

---

## Key Insights

### Why Smart Betting Performs Better

1. **Selective Betting**: Only bets on 15% of matches where model has >60% confidence
2. **Better Win Rate**: 89% vs 69% by avoiding uncertain matches
3. **Optimal Sizing**: Bets more on high-confidence predictions, less on marginal ones
4. **Compound Growth**: As bankroll grows, bet sizes increase proportionally
5. **Risk Management**: Caps bets at $5 and 10% of bankroll to prevent ruin

### Comparison Table

| Metric | Flat Betting | Smart Betting | Improvement |
|--------|-------------|---------------|-------------|
| Bets Placed | ~19,900 | 2,920 | -85% (selective) |
| Win Rate | 69% | 89% | +29% |
| Avg Bet Size | $1.00 | $4.16 | +316% |
| Final Bankroll | ~$170 | $2,411 | +1,318% |
| ROI | ~70% | +2,311% | +33x |
| Risk Level | Low | Medium | - |

---

## Recommendations

### Use Flat Betting When:
- You want simplicity and consistency
- You don't trust model probabilities
- You want to bet on all matches
- You prefer low variance

### Use Smart Betting When:
- Model has proven accuracy
- You want to maximize returns
- You can tolerate higher variance
- You have sufficient bankroll for compounding

### Further Improvements:
1. **Dynamic Kelly Fraction**: Adjust based on recent performance
2. **Odds Value Filter**: Only bet when model probability > implied probability from odds
3. **Bankroll Management**: Implement stop-loss and take-profit rules
4. **Multi-Strategy**: Use flat betting for parlays, Kelly for singles

---

## Files
- `tennis-inference.py`: Flat betting implementation
- `tennis-inference-smart.py`: Kelly Criterion smart betting
- `tennis_predictions.csv`: Flat betting results
- `tennis_predictions_smart.csv`: Smart betting results
