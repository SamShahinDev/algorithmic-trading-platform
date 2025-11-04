# Manus AI Integration Specification

## Project Chimera - Quantitative Strategy Development

### Overview
This document specifies the integration between Manus AI (research and backtesting) and the XTrading platform (live execution). Manus AI serves as the quantitative research specialist, developing and validating trading strategies that are then imported and executed by our production system.

---

## 1. Architecture

### Separation of Concerns
- **Manus AI**: Pure research, backtesting, optimization
- **XTrading**: Live execution, risk management, broker integration

### Data Flow
```
Manus AI → Strategy Export → Validation Pipeline → Strategy Import → Live Trading
```

---

## 2. Strategy Metadata Format

Each strategy developed by Manus AI must include the following metadata:

```json
{
  "strategy_metadata": {
    "id": "unique_strategy_identifier",
    "name": "Human readable strategy name",
    "version": "1.0.0",
    "created_date": "2024-08-24T00:00:00Z",
    "author": "manus_ai",
    
    "classification": {
      "tier": 1,  // 1=highest priority, 2=medium, 3=lowest
      "category": "mean_reversion",  // or "momentum", "microstructure", etc.
      "timeframe": "5m",  // 1m, 5m, 15m, 1h, 4h, 1d
      "instruments": ["NQ"],  // List of compatible instruments
      "topstepx_compatible": true
    },
    
    "performance_metrics": {
      "backtest_period": "2023-01-01 to 2024-08-01",
      "total_trades": 1250,
      "win_rate": 0.58,
      "profit_factor": 1.85,
      "sharpe_ratio": 1.75,
      "sortino_ratio": 2.1,
      "max_drawdown": 450.00,
      "max_drawdown_duration": "5 days",
      "avg_win": 125.50,
      "avg_loss": 85.25,
      "avg_trade_duration": "45 minutes",
      "annual_return": 0.35,
      "calmar_ratio": 2.8
    },
    
    "risk_parameters": {
      "max_contracts": 1,
      "stop_loss_points": 5,
      "take_profit_points": 7,
      "max_daily_trades": 5,
      "max_concurrent_positions": 1,
      "required_margin": 500,
      "trailing_stop": false,
      "time_stop_minutes": 240
    },
    
    "market_conditions": {
      "best_regimes": ["ranging", "high_volatility"],
      "worst_regimes": ["trending_up"],
      "min_volume": 50000,
      "max_spread": 0.25,
      "avoid_times": ["09:00-09:30", "15:45-16:00"],
      "best_times": ["10:00-14:00"]
    },
    
    "validation": {
      "walk_forward_periods": 12,
      "out_of_sample_sharpe": 1.65,
      "monte_carlo_confidence": 0.95,
      "stress_test_passed": true,
      "correlation_group": "momentum_cluster_1"
    },
    
    "confidence_score": 0.85,  // 0-1 scale
    "deployment_priority": 1,  // 1=highest
    "notes": "Optimized for NQ futures during ranging markets"
  }
}
```

---

## 3. Strategy Logic Format

### Entry Conditions
```json
{
  "entry_rules": {
    "long": {
      "conditions": [
        {
          "indicator": "RSI",
          "period": 14,
          "operator": "<",
          "value": 30,
          "weight": 0.3
        },
        {
          "indicator": "SUPPORT",
          "lookback": 20,
          "operator": "touch",
          "tolerance": 2,
          "weight": 0.4
        },
        {
          "pattern": "BULLISH_ENGULFING",
          "confirmation": true,
          "weight": 0.3
        }
      ],
      "min_total_weight": 0.7,
      "require_all": false
    },
    "short": {
      // Similar structure for short entries
    }
  }
}
```

### Exit Conditions
```json
{
  "exit_rules": {
    "take_profit": {
      "type": "fixed_points",
      "value": 7
    },
    "stop_loss": {
      "type": "fixed_points",
      "value": 5
    },
    "time_exit": {
      "type": "minutes_in_trade",
      "value": 240
    },
    "trailing_stop": {
      "enabled": false,
      "activation_points": 5,
      "trail_distance": 3
    }
  }
}
```

---

## 4. Integration API

### Strategy Import Endpoint
```python
POST /api/manus/import-strategy
Content-Type: application/json

{
  "strategy_metadata": {...},
  "entry_rules": {...},
  "exit_rules": {...},
  "backtest_results": {...}
}
```

### Validation Pipeline
1. **Schema Validation**: Ensure all required fields present
2. **Performance Validation**: Verify metrics meet minimum thresholds
3. **Risk Validation**: Check TopStepX compliance
4. **Correlation Check**: Ensure not highly correlated with existing strategies
5. **Paper Trade Test**: Run 100 simulated trades before live deployment

---

## 5. Minimum Requirements for Strategy Acceptance

### Performance Thresholds
- Win Rate: ≥ 50%
- Sharpe Ratio: ≥ 1.0
- Max Drawdown: ≤ $1,000
- Profit Factor: ≥ 1.3
- Minimum 500 trades in backtest

### TopStepX Compliance
- Max daily loss compatible with $1,500 limit
- No overnight holding (initially)
- Commission-adjusted profitability
- Trailing drawdown consideration

### Risk Requirements
- Stop loss defined for every trade
- Maximum 1 contract per trade (initially)
- Clear exit conditions
- Time-based stops

---

## 6. Strategy Categories

### Tier 1 (Priority)
- Mean Reversion at Support/Resistance
- Momentum Breakouts
- Engulfing Pattern Reversals

### Tier 2 (Secondary)
- Microstructure/Order Flow
- Volume Profile Trading
- VWAP Deviations

### Tier 3 (Experimental)
- Pairs Trading (NQ/ES spread)
- Statistical Arbitrage
- Machine Learning Models

---

## 7. Backtesting Requirements

### Data Requirements
- Minimum 6 months historical data
- Tick or 1-minute resolution
- Include extended hours for futures
- Account for holidays and half-days

### Realistic Assumptions
```json
{
  "backtest_config": {
    "slippage_ticks": 1,
    "commission_per_side": 2.50,
    "partial_fills": true,
    "reject_rate": 0.01,
    "latency_ms": 50,
    "spread_modeling": true
  }
}
```

---

## 8. Strategy Lifecycle

### Development Phase (Manus AI)
1. Hypothesis generation
2. Strategy coding
3. Initial backtest
4. Parameter optimization
5. Walk-forward analysis
6. Monte Carlo simulation
7. Stress testing

### Validation Phase (XTrading)
1. Import strategy
2. Schema validation
3. Performance verification
4. Paper trading (min 50 trades)
5. Risk assessment
6. Correlation analysis

### Deployment Phase
1. Add to strategy pool
2. Assign priority tier
3. Monitor initial performance
4. Adjust confidence score
5. Full production deployment

### Monitoring Phase
1. Real-time performance tracking
2. Deviation detection
3. Confidence adjustment
4. Rotation decisions
5. Retirement criteria

---

## 9. Communication Protocol

### Strategy Updates
```json
{
  "update_type": "performance_degradation",
  "strategy_id": "mean_rev_sr_v2",
  "metrics": {
    "recent_win_rate": 0.42,
    "recent_sharpe": 0.8
  },
  "recommendation": "reduce_allocation"
}
```

### Feedback Loop
```json
{
  "feedback_type": "live_performance",
  "strategy_id": "momentum_break_v1",
  "period": "2024-08-01 to 2024-08-24",
  "actual_metrics": {
    "win_rate": 0.55,
    "avg_slippage": 1.2,
    "execution_issues": ["fast_market_rejections"]
  }
}
```

---

## 10. Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Set up Manus AI environment
- [ ] Configure NQ data feed
- [ ] Implement backtest framework
- [ ] Create first 3 strategies

### Phase 2: Development (Weeks 2-3)
- [ ] Develop 5+ additional strategies
- [ ] Run walk-forward optimization
- [ ] Perform Monte Carlo simulations
- [ ] Complete stress testing

### Phase 3: Integration (Week 4)
- [ ] Build import API
- [ ] Create validation pipeline
- [ ] Set up paper trading
- [ ] Test integration

### Phase 4: Deployment (Week 5)
- [ ] Deploy top 5 strategies
- [ ] Monitor initial performance
- [ ] Adjust parameters
- [ ] Full production release

---

## 11. Success Metrics

### Strategy Development
- 10+ strategies developed
- 5+ strategies pass validation
- 3+ strategies in production

### Performance Targets
- Portfolio Sharpe > 1.5
- Monthly win rate > 55%
- Max drawdown < $1,000
- Positive expectancy after costs

### Operational Goals
- < 1% execution errors
- < 100ms strategy decision time
- 99.9% uptime
- Full TopStepX compliance

---

## Contact & Support

**Manus AI Integration Team**
- Strategy Development: manus_ai@quantresearch
- Technical Integration: xtrading_dev@platform
- Risk & Compliance: risk_team@topstepx

---

*Last Updated: August 24, 2024*
*Version: 1.0.0*