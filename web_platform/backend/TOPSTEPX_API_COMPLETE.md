# TopStepX ProjectX Gateway API - Complete Integration

## ✅ Integration Status: FULLY OPERATIONAL

### Authentication
- **Username**: exotictrades
- **API Key**: Configured in `.env.topstepx`
- **Session Token**: 24-hour validity, auto-renewed
- **Status**: ✅ Connected and authenticated

## API Endpoints Implemented

### 1. Authentication & Session
- `POST /Auth/loginKey` - Initial authentication ✅
- `POST /Auth/validate` - Session validation ✅

### 2. Account Management
- `POST /Account/search` - Get account information ✅

### 3. Market Data
- `POST /History/retrieveBars` - Get price data ✅
- `POST /Contract/search` - Search contracts ✅
- `POST /Contract/available` - List available contracts ✅

### 4. Order Management
- `POST /Order/place` - Place new orders ✅
- `POST /Order/cancel` - Cancel orders ✅
- `POST /Order/modify` - Modify open orders ✅
- `POST /Order/searchOpen` - Get open orders ✅
- `POST /Order/search` - Search order history ✅

### 5. Position Management
- `POST /Position/closeContract` - Close full position ✅
- `POST /Position/partialCloseContract` - Partial close ✅

## Contract Mappings

```python
contract_map = {
    "NQ": "CON.F.US.ENQ.U25",   # E-mini NASDAQ-100
    "MNQ": "CON.F.US.MNQ.U25",  # Micro E-mini NASDAQ-100
    "ES": "CON.F.US.EP.U25",    # E-mini S&P 500
    "MES": "CON.F.US.MES.U25",  # Micro E-mini S&P 500
}
```

## Order Type Mappings

```python
class OrderType(Enum):
    LIMIT = 1           # Limit order
    MARKET = 2          # Market order
    STOP = 4            # Stop order
    TRAILING_STOP = 5   # Trailing stop
    JOIN_BID = 6        # Join bid
    JOIN_ASK = 7        # Join ask

class OrderSide(Enum):
    BUY = 0   # Bid (going long)
    SELL = 1  # Ask (going short)
```

## Trading Capabilities

### Order Placement
```python
# Example: Place a market buy order
order_result = await topstepx_client.place_order(
    symbol="NQ",
    side=OrderSide.BUY,
    quantity=1,
    order_type=OrderType.MARKET,
    custom_tag="scalp_001"
)
```

### Order Modification
```python
# Modify stop price on existing order
success = await topstepx_client.modify_order(
    order_id=12345,
    stop_price=23500.00
)
```

### Position Management
```python
# Close full position
await topstepx_client.close_position("CON.F.US.ENQ.U25")

# Partial close (reduce by 1 contract)
await topstepx_client.partial_close_position("CON.F.US.ENQ.U25", 1)

# Emergency close all
await topstepx_client.close_all_positions()
```

## Risk Management Features

### TopStepX Evaluation Rules Enforced
1. **Daily Loss Limit**: $1,500 max
2. **Trailing Drawdown**: $2,000 max
3. **Profit Target**: $3,000 to pass
4. **Position Size**: 1 contract max (configurable)
5. **Daily Trade Limit**: 10 trades max
6. **Overnight Holding**: Disabled initially

### Automatic Safety Features
- Recovery mode activates at 50% of daily loss limit
- Emergency stop at 90% of daily loss limit
- Position sizing with Kelly Criterion (25% fraction)
- Correlation limits between strategies (0.6 max)
- Automatic brackets from TopStepX account settings

## Current Market Data (Real-Time)

- **NQ Price**: ~$23,550
- **Bid/Ask Spread**: Typically $3-5
- **Contract Value**: ~$471,000 (NQ @ 23,550 × $20/point)
- **Margin Required**: Set by TopStepX account

## System Architecture

```
┌─────────────────────────────────────┐
│         Web Interface               │
│    (React Dashboard + WebSocket)    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         FastAPI Backend             │
│        (app.py - Port 8000)         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     Strategy Orchestrator           │
│   (7 strategies, 3 tiers)           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    TopStepX Compliance Engine       │
│  (Rules enforcement, P&L tracking)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Risk Management System         │
│  (Position sizing, stop losses)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      TopStepX Client (API)          │
│   (Order execution, market data)    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    TopStepX ProjectX Gateway        │
│    (api.topstepx.com/api)           │
└─────────────────────────────────────┘
```

## Trading Strategies Active

### Tier 1 (Primary)
1. **Mean Reversion (S/R)** - Trade bounces at support/resistance
2. **Momentum Breakout** - Trade continuation on breakouts

### Tier 2 (Secondary)
3. **Microstructure Analysis** - Order flow imbalances
4. **Volume Profile** - Trade at high volume nodes

### Tier 3 (Advanced)
5. **Pairs Trading** - Relative value between correlated assets
6. **Statistical Arbitrage** - Mean reversion of price deviations
7. **Engulfing Patterns** - Candlestick reversal patterns

## Monitoring & Logs

### Key Files
- **Main App**: `/backend/app.py`
- **TopStepX Client**: `/backend/brokers/topstepx_client.py`
- **Compliance**: `/backend/topstepx/compliance.py`
- **Strategies**: `/backend/strategies/orchestrator.py`
- **Database**: `/backend/database/trades.db`

### Health Checks
```bash
# Check system status
curl http://localhost:8000/api/status

# Start trading
curl -X POST http://localhost:8000/api/control/start

# Emergency stop
curl -X POST http://localhost:8000/api/emergency/stop

# Test TopStepX connection
python3 test_topstepx.py
```

## Troubleshooting

### Common Issues & Solutions

1. **No Account ID Returned**
   - TopStepX may require account activation
   - Check if evaluation account is funded
   - Verify username matches account

2. **Market Data Shows 0**
   - Market may be closed (check CME hours)
   - Session token may have expired (24hr limit)
   - Try reconnecting: restart app

3. **Order Rejected**
   - Check compliance limits (daily loss, trade count)
   - Verify market is open
   - Ensure sufficient margin

4. **Connection Failed**
   - Verify API key is correct
   - Check username is configured
   - Ensure internet connectivity

## Next Steps

1. **Monitor First Trades**
   - System will auto-execute when signals appear
   - Watch compliance dashboard for P&L tracking
   - Review trade logs in database

2. **Optimize Settings**
   - Adjust bracket profit/stop in `.env.topstepx`
   - Fine-tune strategy parameters
   - Monitor win rate and adjust

3. **Scale Up Carefully**
   - Start with 1 contract (current setting)
   - Build consistency before increasing size
   - Pass evaluation before going live

---

**System Status**: 🟢 FULLY OPERATIONAL
**Market Status**: 🟢 OPEN (Sunday 5PM - Friday 4PM CT)
**Connection**: ✅ Authenticated as `exotictrades`
**Ready to Trade**: YES