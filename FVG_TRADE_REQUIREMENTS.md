# Complete FVG Trade Requirements & Parameters

## 1. MARKET STRUCTURE REQUIREMENTS (Pre-Detection)

### Swing Detection (`_update_swings`)
- **Swing High**: Price must be higher than 2 bars before AND 2 bars after
- **Swing Low**: Price must be lower than 2 bars before AND 2 bars after
- **Lookback**: Last 20 bars tracked for swing points

### Liquidity Sweep (MANDATORY - Lines 245-247, 295-298)
- **Bullish FVG**: Must sweep below a recent swing low by at least 2 ticks (0.50 pts)
- **Bearish FVG**: Must sweep above a recent swing high by at least 2 ticks (0.50 pts)
- **Close Requirement**: Must close back above/below the swept level
- **⚠️ WITHOUT LIQUIDITY SWEEP, NO FVG IS CREATED**

## 2. FVG DETECTION PARAMETERS

### Displacement Bar Requirements (Middle Bar)
- **Body Fraction**: ≥ 60% (`min_body_frac: 0.60`)
- **Range Size (Dynamic Mode)**: MAX(3.0 pts, 0.6 × ATR14)
  - Base: 3.0 points minimum
  - ATR Multiplier: 0.6
- **Volume**: ≥ 1.2× of 20-bar average (`min_vol_mult: 1.2`)

### Gap Requirements
- **Bullish Gap**: Bar[i] high < Bar[i+2] low (gap up)
- **Bearish Gap**: Bar[i] low > Bar[i+2] high (gap down)
- **Minimum Bars**: Need 30+ bars in cache for detection

### Quality Score Calculation
```
Quality = (body_frac × 0.3) + (atr_mult × 0.4) + (vol_mult × 0.3)
```
- **Minimum Quality**: 0.55 (`min_quality: 0.55`)
- Quality < 0.55 = FVG rejected

## 3. FVG LIFECYCLE STATES

### FRESH → ARMED Transition
**Arming Conditions**:
1. Price must wick into FVG zone (touch)
2. Close must defend outside inner 25% of zone
   - Long: Close ≥ (top - 0.25×zone_size)
   - Short: Close ≤ (bottom + 0.25×zone_size)
3. Timeout: 600 seconds (`arm_timeout_sec: 600`)

### Invalidation Conditions
1. **Zone Consumption**: 75% of gap consumed (`invalidate_frac: 0.75`)
2. **Structure Break**:
   - Long: Close < bottom - 2 pts
   - Short: Close > top + 2 pts
3. **Timeout**: Not armed within 600 seconds
4. **Age**: Removed after 3600 seconds (1 hour)

## 4. PRE-TRADE GUARDS (Before Entry)

### Position Management
- **Max Concurrent**: 1 position (`max_concurrent: 1`)
- **Cooldown**: 60 seconds between trades (`cooldown_secs: 60`)
- **One-and-Done**: Each FVG traded only once (`one_and_done_per_fvg: true`)

### Data Freshness
- **Max Staleness**: 3.5 seconds (3500ms)
- **Required**: Latest bar timestamp must be recent

### RSI Veto
- **Long Blocked**: RSI > 72
- **Short Blocked**: RSI < 28
- **Period**: 14 bars on 1-minute

## 5. ENTRY PARAMETERS

### Primary Entry (Mid-Gap)
- **Location**: Middle of FVG zone (`use_mid_entry: true`)
- **Order Type**: Limit order
- **TTL**: 90 seconds (`ttl_sec: 90`)
- **Cancel Distance**: 8 ticks away (`cancel_if_runs_ticks: 8`)

### Edge Retry (Optional)
- **Enabled**: Yes (`use_edge_retry: true`)
- **Offset**: 2 ticks from edge (`edge_offset_ticks: 2`)
- **TTL**: 45 seconds (`edge_retry.ttl_sec: 45`)

## 6. RISK MANAGEMENT

### Stop Loss
- **Location**:
  - Long: Bottom - 2 ticks
  - Short: Top + 2 ticks
- **Max Stop**: 7.5 points (`stop_pts: 7.5`)
- **Rejection**: Trade skipped if stop > 7.5 pts

### Take Profit
- **Target**: 17.5 points from entry (`tp_pts: 17.5`)

### Trailing Stop
- **Trigger**: 12.0 points profit (`trigger_pts: 12.0`)
- **Giveback**: 10 ticks (2.5 pts) (`giveback_ticks: 10`)

### Breakeven
- **Move to BE**: At 9.0 points profit (`breakeven_pts: 9.0`)

## 7. EXECUTION FLOW

1. **Detection Phase**:
   - Scan every 3 seconds
   - Check for liquidity sweep → displacement → gap
   - Calculate quality score
   - Create FVG if all criteria met

2. **Arming Phase**:
   - Wait for price to wick into zone
   - Verify close defends zone
   - Transition to ARMED state

3. **Entry Phase**:
   - Check all pre-trade guards
   - Place limit order at mid-point
   - Apply 90-second TTL
   - Optional edge retry after 45s

4. **Management Phase**:
   - Monitor for breakeven at +9 pts
   - Activate trail at +12 pts
   - Exit at TP (+17.5 pts) or stop

## 8. SUMMARY OF KEY PARAMETERS

```python
FVG_REQUIREMENTS = {
    # Detection
    "min_bars": 30,
    "liquidity_sweep_required": True,
    "min_body_frac": 0.60,
    "min_displacement": "MAX(3.0, 0.6×ATR)",
    "min_volume_mult": 1.2,
    "min_quality": 0.55,

    # Lifecycle
    "arm_timeout": 600,  # seconds
    "invalidate_consumption": 0.75,  # 75%
    "defend_inner_zone": 0.25,  # 25%

    # Risk
    "max_stop": 7.5,  # points
    "target": 17.5,  # points
    "breakeven": 9.0,  # points
    "trail_trigger": 12.0,  # points

    # Guards
    "max_positions": 1,
    "cooldown": 60,  # seconds
    "rsi_long_veto": 72,
    "rsi_short_veto": 28,
    "data_staleness": 3.5  # seconds
}
```

## CRITICAL INSIGHT
**The bot requires a liquidity sweep before the displacement bar to create an FVG. This is why no trades have been taken - the market hasn't produced this specific institutional pattern of sweep → displacement → gap.**