# SDK Trading Agent - Analysis Scripts

Utility scripts for analyzing logs, monitoring performance, and exporting metrics.

## Available Scripts

### 1. `analyze_logs.py` - Log Analysis

Analyze trading logs and generate comprehensive reports.

**Usage:**

```bash
# Analyze all logs
python scripts/analyze_logs.py

# Analyze specific date
python scripts/analyze_logs.py --date 2025-01-07

# Analyze last 7 days
python scripts/analyze_logs.py --days 7

# Filter by strategy
python scripts/analyze_logs.py --strategy VWAP

# Export results to JSON
python scripts/analyze_logs.py --date 2025-01-07 --export analysis.json
```

**Output:**

```
================================================================================
TRADING LOG ANALYSIS REPORT
================================================================================

📊 DECISION ANALYSIS
--------------------------------------------------------------------------------
Total Decisions: 45
Pre-Filter Pass Rate: 78.0% (35/45)
Claude Called: 35
Claude Approval Rate: 45.7% (16/35)
Post-Validation Pass Rate: 87.5% (14/16)
Executions: 12
Avg Claude Latency: 387.2ms
Avg Validation Slippage: 1.85 ticks
Avg Total Slippage: 2.45 ticks

📈 TRADE ANALYSIS
--------------------------------------------------------------------------------
Total Trades: 12
Win Rate: 66.7% (8W / 4L)
Total P&L: $145.00
Avg P&L: $12.08
Avg Winner: $32.50
Avg Loser: -$18.75
Avg Duration: 15.3 minutes
Avg Entry Slippage: 2.15 ticks
Avg Exit Slippage: 1.85 ticks
Total Commission: $24.00

⚡ SLIPPAGE ANALYSIS
--------------------------------------------------------------------------------
Total Events: 24
Avg Slippage: 2.15 ticks
Max Slippage: 3.20 ticks
Min Slippage: 0.80 ticks
Total Cost: $54.00
```

### 2. `monitor_performance.py` - Real-Time Monitoring

Monitor trading performance in real-time with auto-refresh.

**Usage:**

```bash
# Start monitoring (updates every 10 seconds)
python scripts/monitor_performance.py

# Custom update interval (30 seconds)
python scripts/monitor_performance.py --interval 30

# Alerts-only mode (compact)
python scripts/monitor_performance.py --alerts-only

# Custom thresholds
python scripts/monitor_performance.py --slippage-threshold 2.5 --latency-threshold 400
```

**Dashboard Output:**

```
================================================================================
SDK TRADING AGENT - REAL-TIME PERFORMANCE MONITOR
================================================================================

Last Updated: 2025-01-07 14:35:22

Session Duration: 125.3 minutes

📈 TRADING PERFORMANCE
--------------------------------------------------------------------------------
Total Trades: 8 (W: 6 / L: 2)
Win Rate: 75.0%
Total P&L: $125.50
Avg P&L: $15.69
Open Positions: 1

🎯 DECISION STATISTICS
--------------------------------------------------------------------------------
Total Decisions: 32
Pre-Filter Pass Rate: 75.0%
Claude Approval Rate: 50.0%
Post-Validation Pass Rate: 88.9%
Total Claude Calls: 24
Total Executions: 8

🕐 LATENCY
--------------------------------------------------------------------------------
Avg Claude Latency: 387.5ms [✓ OK]
Threshold: 500ms

⚡ SLIPPAGE
--------------------------------------------------------------------------------
Avg Slippage: 2.15 ticks
Running Avg (20): 2.10 ticks [✓ OK]
Max Slippage: 2.80 ticks
Total Cost: $53.75
Threshold: 3.0 ticks

✓ No Active Alerts

================================================================================
```

**Alerts-Only Mode:**

```
[14:35:22] No alerts
[14:35:52] ⚠️ [WARNING] HIGH_SLIPPAGE: Running average slippage (3.25 ticks) exceeds threshold (3.0 ticks)
[14:36:22] 🚨 [CRITICAL] DAILY_LOSS_LIMIT: Daily loss limit reached: $-152.50 (limit: $-150.00)
```

### 3. `export_metrics.py` - Metrics Export

Export trading metrics to CSV or JSON for external analysis.

**Usage:**

```bash
# Export all to CSV (creates multiple files)
python scripts/export_metrics.py --format csv --output exports/ --type all

# Export decisions only
python scripts/export_metrics.py --format csv --output decisions.csv --type decisions

# Export trades for last 7 days
python scripts/export_metrics.py --format csv --output trades.csv --type trades --days 7

# Export everything to JSON
python scripts/export_metrics.py --format json --output metrics.json

# Export last 30 days to JSON
python scripts/export_metrics.py --format json --output metrics_30d.json --days 30
```

**CSV Format (decisions.csv):**

```csv
timestamp,strategy,setup_score,pre_filter,claude_called,claude_decision,claude_confidence,claude_latency_ms,...
2025-01-07T10:43:00.123Z,vwap,9.2,PASS,true,ENTER,0.84,387,...
2025-01-07T10:45:30.456Z,breakout,7.5,SKIP,false,,,,...
```

**JSON Format:**

```json
{
  "export_timestamp": "2025-01-07T14:35:00.000Z",
  "days_included": 7,
  "statistics": {
    "decisions": {
      "total_decisions": 145,
      "avg_claude_latency_ms": 387.2,
      ...
    },
    "slippage": {
      "avg_slippage_ticks": 2.15,
      "total_cost_dollars": 125.50,
      ...
    }
  },
  "data": {
    "decisions": [...],
    "trades": [...],
    "slippage_events": [...]
  }
}
```

## Common Workflows

### Daily Review

After each trading day:

```bash
# 1. Analyze the day's performance
python scripts/analyze_logs.py --date $(date +%Y-%m-%d)

# 2. Export for record-keeping
python scripts/export_metrics.py --format json --output daily_reports/$(date +%Y-%m-%d).json --days 1
```

### Weekly Analysis

Review weekly performance:

```bash
# Analyze last 7 days
python scripts/analyze_logs.py --days 7 --export weekly_report.json

# Compare strategies
python scripts/analyze_logs.py --days 7 --strategy VWAP
python scripts/analyze_logs.py --days 7 --strategy Breakout
python scripts/analyze_logs.py --days 7 --strategy Momentum
```

### Live Monitoring

During trading hours:

```bash
# Terminal 1: Full dashboard
python scripts/monitor_performance.py --interval 10

# Terminal 2: Alerts only
python scripts/monitor_performance.py --alerts-only --interval 5
```

### Data Export for Analysis

Export data for spreadsheet analysis:

```bash
# Export all to CSV
mkdir -p exports/$(date +%Y-%m-%d)
python scripts/export_metrics.py \
  --format csv \
  --output exports/$(date +%Y-%m-%d)/ \
  --type all \
  --days 30
```

## Integration with Other Tools

### Excel/Google Sheets

1. Export to CSV:
   ```bash
   python scripts/export_metrics.py --format csv --output trades.csv --type trades
   ```

2. Open in Excel/Sheets for pivot tables and charts

### Python Analysis (Jupyter/Pandas)

```python
import json
import pandas as pd

# Load exported JSON
with open('metrics.json') as f:
    data = json.load(f)

# Convert to DataFrames
decisions_df = pd.DataFrame(data['data']['decisions'])
trades_df = pd.DataFrame(data['data']['trades'])
slippage_df = pd.DataFrame(data['data']['slippage_events'])

# Analyze
print(trades_df.groupby('strategy')['pnl_net'].sum())
```

### Automated Daily Reports

Add to crontab for daily email reports:

```bash
# crontab -e
0 17 * * 1-5 cd /path/to/sdk-agent && python scripts/analyze_logs.py --date $(date +\%Y-\%m-\%d) --export /tmp/daily_report.json && mail -s "Daily Trading Report" you@email.com < /tmp/daily_report.json
```

## Troubleshooting

### Script Won't Run

```bash
# Make sure scripts are executable
chmod +x scripts/*.py

# Check Python path
python --version  # Should be 3.8+

# Run with python3 explicitly
python3 scripts/analyze_logs.py
```

### No Data Found

```bash
# Check log directory exists
ls -la logs/

# Verify JSONL files exist
ls -la logs/*.jsonl

# Check file contents
head logs/decisions.jsonl
```

### Import Errors

```bash
# Ensure you're in the project root
pwd  # Should be .../sdk agent/

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd /path/to/sdk-agent
python scripts/analyze_logs.py
```

## Advanced Usage

### Custom Analysis

Create your own analysis scripts using the logging modules:

```python
from pathlib import Path
from logging.decision_logger import DecisionLogger
from logging.trade_logger import TradeLogger

# Load data
log_dir = Path('logs')
decision_logger = DecisionLogger(log_dir=log_dir)
trade_logger = TradeLogger(log_dir=log_dir)

# Get data
decisions = decision_logger.get_recent_decisions(count=100)
trades = trade_logger.get_recent_trades(count=100)

# Analyze
for trade in trades:
    if trade['pnl_net'] > 50:
        print(f"Big winner: {trade['strategy']} ${trade['pnl_net']:.2f}")
```

### Streaming Alerts

Monitor alerts in real-time and send to Discord/Slack:

```python
from monitoring.alert_system import AlertSystem

def send_to_discord(alert):
    # Your Discord webhook logic
    print(f"Sending to Discord: {alert}")

alert_system = AlertSystem(alert_callback=send_to_discord)

while True:
    alerts = alert_system.check_all_alerts()
    time.sleep(30)
```

## Best Practices

1. **Run daily analysis** - Review each day's performance
2. **Monitor during trading** - Use `monitor_performance.py` to catch issues early
3. **Export regularly** - Backup your data weekly/monthly
4. **Compare strategies** - Use `--strategy` filter to identify best performers
5. **Track trends** - Export to CSV and plot in spreadsheet for visual analysis
6. **Set alerts** - Use custom thresholds for your risk tolerance
