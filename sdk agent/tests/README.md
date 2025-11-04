# SDK Trading Agent - Test Suite

Comprehensive test suite for the SDK Trading Agent with slippage monitoring and latency protection.

## Test Structure

```
tests/
├── test_indicators.py          # Unit tests for VWAP, ADX, RSI, EMA, MACD
├── test_strategies.py          # Strategy logic and confidence scoring
├── test_topstep_api.py         # TopStepX API integration (requires credentials)
├── test_websocket.py           # WebSocket connection and callbacks
├── test_agent_integration.py   # End-to-end workflow tests
├── test_latency_protection.py  # Pre-filter and post-validation tests
└── test_slippage_tracking.py   # Slippage measurement and logging
```

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Test File
```bash
pytest tests/test_indicators.py -v
```

### Specific Test Class
```bash
pytest tests/test_indicators.py::TestVWAPAnalyzer -v
```

### Specific Test Function
```bash
pytest tests/test_latency_protection.py::TestPreFilter::test_prefilter_low_confidence_skips_claude -v
```

### By Markers
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### Unit Tests (test_indicators.py, test_strategies.py)
- Test individual components in isolation
- Mock external dependencies
- Fast execution
- No API credentials required

### Integration Tests (test_agent_integration.py)
- Test component interaction
- Mock external services (Claude, TopStepX)
- Verify end-to-end workflow

### API Tests (test_topstep_api.py)
- Require TopStepX credentials
- Test real API calls (practice account only)
- Skip if credentials not available

### WebSocket Tests (test_websocket.py)
- Test WebSocket connection and data handling
- Require `--run-websocket` flag for live tests

### Latency Protection Tests (test_latency_protection.py)
Tests the three-layer latency protection:
1. **Pre-filter**: Score < 8 → Skip Claude
2. **Post-validation**: Re-check setup after Claude response
3. **Slippage Check**: Max 3 ticks acceptable

### Slippage Tracking Tests (test_slippage_tracking.py)
Tests slippage measurement from signal to fill:
- Running averages
- Cost calculation ($5/tick for NQ)
- Alerts on excessive slippage (>3 tick average)

## Environment Setup

### Required
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Set environment variables (for API tests)
export TOPSTEPX_EMAIL="your@email.com"
export TOPSTEPX_PASSWORD="your_password"
```

### Optional
```bash
# For coverage reports
pip install pytest-cov

# Run with coverage
pytest --cov=. --cov-report=html
```

## Test Execution Order

Before running live trading, follow this order:

### 1. Unit Tests (Fast, No Credentials)
```bash
pytest tests/test_indicators.py tests/test_strategies.py -v
```

**Expected**: All tests pass, indicators calculate correctly, strategies score properly.

### 2. Latency Protection Tests
```bash
pytest tests/test_latency_protection.py -v
```

**Expected**: Pre-filter skips low scores, post-validation catches degraded setups, slippage is measured.

### 3. Slippage Tracking Tests
```bash
pytest tests/test_slippage_tracking.py -v
```

**Expected**: Slippage tracked correctly, running averages calculated, alerts work.

### 4. Integration Tests
```bash
pytest tests/test_agent_integration.py -v
```

**Expected**: Full workflow works, risk manager enforces limits, trades are validated.

### 5. API Tests (Requires Credentials)
```bash
pytest tests/test_topstep_api.py -v
```

**Expected**: Authentication works, account info retrieved, market data accessible.

### 6. Dry-Run Mode (1 Full Trading Day)
```bash
python main.py --dry-run
```

**What to Monitor**:
- ✓ Strategies generate signals
- ✓ Claude makes decisions (track latency)
- ✓ Slippage is calculated
- ✓ No actual orders placed
- ✓ All logging works

**Expected Metrics**:
- Claude latency: 300-500ms typical
- Pre-filter rate: 70-80% (only 8+ scores reach Claude)
- Slippage: < 3 ticks average
- Validation success: > 90%

### 7. Live Trading (After Dry-Run Validation)
```bash
python main.py
```

## Dry-Run Mode

### Purpose
Validate entire system with real market data WITHOUT placing actual orders.

### What It Does
- ✓ Connects to real TopStepX WebSocket
- ✓ Evaluates all strategies every minute
- ✓ Calls Claude API for 8+ confidence setups
- ✓ Re-validates setups (post-Claude)
- ✓ Calculates theoretical slippage
- ✓ Logs all decisions and metrics
- ✗ Does NOT place actual orders

### How to Run
```bash
# Start dry-run
python main.py --dry-run

# Will prompt for confirmation
# Let it run for 1 full trading day (9:30 AM - 4:00 PM CT)

# Review logs/dry_run_summary.json
```

### Dry-Run Checklist

After 1 full day of dry-run:

- [ ] At least 5 high-confidence setups detected (8+/10)
- [ ] Claude latency < 500ms average
- [ ] Slippage < 3 ticks average
- [ ] No validation errors or crashes
- [ ] All strategies fired at least once
- [ ] Risk limits work correctly
- [ ] Logs are complete and readable

**Only proceed to live trading if all items checked.**

## Key Metrics to Monitor

### During Testing
- **Indicator Accuracy**: RSI, EMA, ATR match expected values
- **Strategy Confidence**: Scores 0-10, high-confidence setups rare
- **Pre-Filter Rate**: 70-80% filtered (only best reach Claude)
- **Latency**: 300-500ms typical for Claude calls
- **Slippage**: < 3 ticks average acceptable

### During Dry-Run
- **Signals Per Day**: 10-20 expected
- **Claude Approvals**: 30-50% of high-confidence setups
- **Validation Failures**: < 10%
- **Theoretical Win Rate**: 55-65% target

## Troubleshooting

### Tests Fail: "No module named 'indicators'"
```bash
# Add parent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Tests Fail: "TopStepX credentials not available"
```bash
# Expected - set credentials to run API tests
export TOPSTEPX_EMAIL="your@email.com"
export TOPSTEPX_PASSWORD="your_password"
```

### Async Tests Fail
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Verify pytest.ini has: asyncio_mode = auto
```

### WebSocket Tests Skip
```bash
# WebSocket tests require flag
pytest tests/test_websocket.py --run-websocket
```

## Best Practices

1. **Run unit tests first** - Fast, no dependencies
2. **Run dry-run for full day** - Validate with real data
3. **Review dry-run metrics** - Check latency and slippage
4. **Start live trading small** - Monitor first few trades closely
5. **Never skip tests** - They catch bugs before they cost money

## Coverage Goals

- **Unit Tests**: > 80% code coverage
- **Integration Tests**: All critical paths tested
- **Edge Cases**: Error handling verified
- **Performance**: Latency < 500ms, slippage < 3 ticks
