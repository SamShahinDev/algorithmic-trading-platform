# XTRADING FVG Bot Makefile
# Provides convenient shortcuts for FVG triage, bot operations, and development tasks

# Project configuration
PROJECT_ROOT := $(shell pwd)
PYTHON := python3
VENV_PATH := .venv
BIN_DIR := bin
TOOLS_DIR := tools
LOGS_DIR := logs
NQ_BOT_DIR := nq_bot

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

##@ Help
.PHONY: help
help: ## Display this help message
	@echo "$(CYAN)XTRADING FVG Bot - Make Targets$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ FVG Triage & Diagnostics
.PHONY: fvg-triage
fvg-triage: ## Run quick FVG triage analysis (bash)
	@echo "$(BLUE)[INFO]$(NC) Running FVG triage analysis..."
	@$(BIN_DIR)/fvg_triage.sh

.PHONY: fvg-triage-verbose
fvg-triage-verbose: ## Run verbose FVG triage analysis (bash)
	@echo "$(BLUE)[INFO]$(NC) Running verbose FVG triage analysis..."
	@$(BIN_DIR)/fvg_triage.sh -v

.PHONY: fvg-triage-8h
fvg-triage-8h: ## Run FVG triage for last 8 hours
	@echo "$(BLUE)[INFO]$(NC) Running FVG triage for last 8 hours..."
	@$(BIN_DIR)/fvg_triage.sh -t 8

.PHONY: fvg-triage-config
fvg-triage-config: ## Show FVG configuration from logs
	@echo "$(BLUE)[INFO]$(NC) Showing FVG configuration..."
	@$(BIN_DIR)/fvg_triage.sh -c

.PHONY: fvg-analysis
fvg-analysis: ## Run advanced Python FVG analysis
	@echo "$(BLUE)[INFO]$(NC) Running advanced FVG analysis..."
	@$(PYTHON) $(TOOLS_DIR)/fvg_triage.py

.PHONY: fvg-analysis-verbose
fvg-analysis-verbose: ## Run verbose Python FVG analysis
	@echo "$(BLUE)[INFO]$(NC) Running verbose FVG analysis..."
	@$(PYTHON) $(TOOLS_DIR)/fvg_triage.py -v

.PHONY: fvg-analysis-export
fvg-analysis-export: ## Export FVG analysis to JSON
	@echo "$(BLUE)[INFO]$(NC) Exporting FVG analysis to JSON..."
	@$(PYTHON) $(TOOLS_DIR)/fvg_triage.py -o fvg_analysis_$(shell date +%Y%m%d_%H%M%S).json

.PHONY: fvg-sanity
fvg-sanity: ## Quick sanity check for FVG bot
	@echo "$(BLUE)[INFO]$(NC) Running FVG bot sanity check..."
	@$(BIN_DIR)/fvg_triage.sh --no-funnel --no-near-misses --no-blocks

##@ Bot Operations
.PHONY: fvg-start
fvg-start: ## Start FVG bot in dry-run mode
	@echo "$(GREEN)[START]$(NC) Starting FVG bot in dry-run mode..."
	@export FVG_DRY_RUN=true && $(PYTHON) $(NQ_BOT_DIR)/fvg_runner.py

.PHONY: fvg-start-live
fvg-start-live: ## Start FVG bot in practice mode (CAUTION)
	@echo "$(YELLOW)[WARNING]$(NC) Starting FVG bot in practice mode..."
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@$(PYTHON) $(NQ_BOT_DIR)/fvg_runner.py

.PHONY: fvg-stop
fvg-stop: ## Stop running FVG bot process
	@echo "$(RED)[STOP]$(NC) Stopping FVG bot..."
	@pkill -f "fvg_runner.py" || echo "No FVG bot process found"

.PHONY: fvg-status
fvg-status: ## Check FVG bot status
	@echo "$(BLUE)[STATUS]$(NC) Checking FVG bot status..."
	@if pgrep -f "fvg_runner.py" > /dev/null; then \
		echo "$(GREEN)FVG bot is running$(NC)"; \
		pgrep -f "fvg_runner.py" | while read pid; do \
			echo "  PID: $$pid"; \
		done; \
	else \
		echo "$(YELLOW)FVG bot is not running$(NC)"; \
	fi
	@if [ -f "$(LOGS_DIR)/nq_bot.heartbeat.json" ]; then \
		echo "Heartbeat:"; \
		cat "$(LOGS_DIR)/nq_bot.heartbeat.json" | $(PYTHON) -m json.tool 2>/dev/null || cat "$(LOGS_DIR)/nq_bot.heartbeat.json"; \
	fi

##@ Log Analysis
.PHONY: logs-tail
logs-tail: ## Tail FVG bot logs
	@echo "$(BLUE)[LOGS]$(NC) Tailing FVG bot logs..."
	@tail -f $(LOGS_DIR)/fvg_bot.log 2>/dev/null || tail -f $(LOGS_DIR)/nq_*.log 2>/dev/null || echo "No log files found"

.PHONY: logs-errors
logs-errors: ## Show recent error messages
	@echo "$(RED)[ERRORS]$(NC) Recent error messages:"
	@grep -i "error\|exception\|failed\|critical" $(LOGS_DIR)/*.log 2>/dev/null | tail -10 || echo "No recent errors found"

.PHONY: logs-config
logs-config: ## Show CONFIG_ECHO lines from logs
	@echo "$(CYAN)[CONFIG]$(NC) Configuration echo from logs:"
	@grep "CONFIG_ECHO:" $(LOGS_DIR)/*.log 2>/dev/null | tail -20 || echo "No CONFIG_ECHO lines found"

.PHONY: logs-telemetry
logs-telemetry: ## Show recent telemetry data
	@echo "$(PURPLE)[TELEMETRY]$(NC) Recent telemetry data:"
	@grep "TELEMETRY:" $(LOGS_DIR)/*.log 2>/dev/null | tail -5 | while read line; do \
		echo "$$line" | sed 's/.*TELEMETRY: //' | $(PYTHON) -m json.tool 2>/dev/null || echo "$$line"; \
	done || echo "No telemetry data found"

.PHONY: logs-blocks
logs-blocks: ## Show recent blocking reasons
	@echo "$(YELLOW)[BLOCKS]$(NC) Recent blocking reasons:"
	@grep -E "(veto|blocked|Burst guard|too stale|Daily trade cap)" $(LOGS_DIR)/*.log 2>/dev/null | tail -10 || echo "No recent blocks found"

##@ Development & Testing
.PHONY: test-fvg
test-fvg: ## Run FVG strategy tests
	@echo "$(BLUE)[TEST]$(NC) Running FVG tests..."
	@$(PYTHON) -m pytest tests/ -k fvg -v

.PHONY: test-fvg-detection
test-fvg-detection: ## Test FVG detection specifically
	@echo "$(BLUE)[TEST]$(NC) Testing FVG detection..."
	@$(PYTHON) test_fvg_detection.py

.PHONY: dry-run
dry-run: ## Quick FVG dry run test
	@echo "$(BLUE)[DRY-RUN]$(NC) Running FVG dry run test..."
	@$(PYTHON) test_fvg_dryrun.py

.PHONY: validate-config
validate-config: ## Validate FVG configuration
	@echo "$(BLUE)[VALIDATE]$(NC) Validating FVG configuration..."
	@$(PYTHON) -c "from $(NQ_BOT_DIR).pattern_config import validate_pattern_config; validate_pattern_config(); print('Configuration valid')"

##@ Data & Monitoring
.PHONY: telemetry-csv
telemetry-csv: ## Show recent CSV telemetry
	@echo "$(CYAN)[CSV]$(NC) Recent CSV telemetry data:"
	@if [ -f "$(LOGS_DIR)/fvg_telemetry.csv" ]; then \
		tail -10 "$(LOGS_DIR)/fvg_telemetry.csv"; \
	else \
		echo "No telemetry CSV found"; \
	fi

.PHONY: heartbeat
heartbeat: ## Show bot heartbeat status
	@echo "$(GREEN)[HEARTBEAT]$(NC) Bot heartbeat status:"
	@if [ -f "$(LOGS_DIR)/nq_bot.heartbeat.json" ]; then \
		cat "$(LOGS_DIR)/nq_bot.heartbeat.json" | $(PYTHON) -m json.tool 2>/dev/null || cat "$(LOGS_DIR)/nq_bot.heartbeat.json"; \
	else \
		echo "No heartbeat file found"; \
	fi

.PHONY: monitor
monitor: ## Real-time monitoring dashboard
	@echo "$(PURPLE)[MONITOR]$(NC) Starting real-time FVG monitoring..."
	@$(PYTHON) monitor_fvg_bot.py

##@ Cleanup & Maintenance
.PHONY: clean-logs
clean-logs: ## Clean old log files (keeps last 7 days)
	@echo "$(YELLOW)[CLEAN]$(NC) Cleaning old log files..."
	@find $(LOGS_DIR) -name "*.log" -mtime +7 -delete 2>/dev/null || true
	@find $(LOGS_DIR) -name "*.csv.backup" -mtime +7 -delete 2>/dev/null || true
	@echo "Old log files cleaned"

.PHONY: backup-logs
backup-logs: ## Backup current logs with timestamp
	@echo "$(BLUE)[BACKUP]$(NC) Backing up current logs..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "logs_backup_$$timestamp.tar.gz" $(LOGS_DIR)/ && \
	echo "Logs backed up to logs_backup_$$timestamp.tar.gz"

.PHONY: reset-telemetry
reset-telemetry: ## Reset telemetry data (backup first)
	@echo "$(YELLOW)[RESET]$(NC) Resetting telemetry data..."
	@if [ -f "$(LOGS_DIR)/fvg_telemetry.csv" ]; then \
		cp "$(LOGS_DIR)/fvg_telemetry.csv" "$(LOGS_DIR)/fvg_telemetry.csv.backup.$$(date +%Y%m%d_%H%M%S)"; \
		echo "timestamp,event,id,direction,entry,stop,tp,mae,mfe,pnl,duration,details" > "$(LOGS_DIR)/fvg_telemetry.csv"; \
		echo "Telemetry reset (backup created)"; \
	else \
		echo "No telemetry file to reset"; \
	fi

##@ Quick Diagnostic Workflows
.PHONY: diagnose-no-trades
diagnose-no-trades: ## Full diagnostic for "no trades" issue
	@echo "$(PURPLE)[DIAGNOSE]$(NC) Running full 'no trades' diagnostic..."
	@echo ""
	@echo "$(CYAN)Step 1: Quick triage$(NC)"
	@$(MAKE) fvg-triage
	@echo ""
	@echo "$(CYAN)Step 2: Advanced analysis$(NC)"
	@$(MAKE) fvg-analysis
	@echo ""
	@echo "$(CYAN)Step 3: Configuration check$(NC)"
	@$(MAKE) logs-config
	@echo ""
	@echo "$(GREEN)Diagnostic complete. Review output above for issues.$(NC)"

.PHONY: diagnose-quick
diagnose-quick: ## Quick 2-hour diagnostic
	@echo "$(BLUE)[QUICK]$(NC) Running quick 2-hour diagnostic..."
	@$(BIN_DIR)/fvg_triage.sh -t 2

.PHONY: case-a
case-a: ## Diagnose Case A: No FVGs detected at all
	@echo "$(YELLOW)[CASE A]$(NC) Diagnosing: No FVGs detected at all"
	@echo "Checking gap and displacement requirements..."
	@$(BIN_DIR)/fvg_triage.sh -c | grep -E "(min_gap_ticks|min_displacement|tick_size)"
	@$(MAKE) logs-telemetry

.PHONY: case-b
case-b: ## Diagnose Case B: FVGs detected but none get armed
	@echo "$(YELLOW)[CASE B]$(NC) Diagnosing: FVGs detected but none get armed"
	@echo "Checking quality and lifecycle settings..."
	@$(BIN_DIR)/fvg_triage.sh -c | grep -E "(min_quality|invalidate_frac)"
	@$(MAKE) fvg-analysis

.PHONY: case-c
case-c: ## Diagnose Case C: FVGs armed but no orders placed
	@echo "$(YELLOW)[CASE C]$(NC) Diagnosing: FVGs armed but no orders placed"
	@echo "Checking blocking reasons..."
	@$(MAKE) logs-blocks
	@$(BIN_DIR)/fvg_triage.sh --no-funnel --no-near-misses

.PHONY: case-d
case-d: ## Diagnose Case D: Orders placed but no fills
	@echo "$(YELLOW)[CASE D]$(NC) Diagnosing: Orders placed but no fills"
	@echo "Checking entry levels and timeouts..."
	@$(BIN_DIR)/fvg_triage.sh -c | grep -E "(entry_pct|ttl_sec|cancel_if_runs)"
	@grep "ENTRY_PLACED\|CANCEL" $(LOGS_DIR)/*.log 2>/dev/null | tail -5

.PHONY: case-e
case-e: ## Diagnose Case E: Data issues
	@echo "$(YELLOW)[CASE E]$(NC) Diagnosing: Data issues"
	@echo "Checking data feed and staleness..."
	@$(MAKE) logs-errors
	@grep "Data too stale\|connection\|broker" $(LOGS_DIR)/*.log 2>/dev/null | tail -10

.PHONY: case-f
case-f: ## Diagnose Case F: No market activity
	@echo "$(YELLOW)[CASE F]$(NC) Diagnosing: No market activity"
	@echo "Checking market conditions and time restrictions..."
	@$(BIN_DIR)/fvg_triage.sh -c | grep -E "(exchange_tz|rth_open|time)"
	@$(MAKE) logs-telemetry

##@ Environment
.PHONY: install-deps
install-deps: ## Install Python dependencies
	@echo "$(BLUE)[INSTALL]$(NC) Installing Python dependencies..."
	@$(PYTHON) -m pip install -r requirements.txt

.PHONY: check-env
check-env: ## Check environment setup
	@echo "$(BLUE)[CHECK]$(NC) Checking environment..."
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Project root: $(PROJECT_ROOT)"
	@echo "Logs directory: $(LOGS_DIR)"
	@echo "Scripts executable:"
	@ls -la $(BIN_DIR)/fvg_triage.sh $(TOOLS_DIR)/fvg_triage.py
	@echo "Environment files:"
	@ls -la .env* 2>/dev/null || echo "No .env files found"

##@ Examples & Usage
.PHONY: examples
examples: ## Show usage examples
	@echo "$(CYAN)FVG Triage Toolkit - Usage Examples$(NC)"
	@echo ""
	@echo "$(WHITE)Quick Diagnostics:$(NC)"
	@echo "  make fvg-triage              # Quick 4-hour analysis"
	@echo "  make fvg-triage-8h           # 8-hour analysis"
	@echo "  make diagnose-quick          # 2-hour quick check"
	@echo ""
	@echo "$(WHITE)Specific Issue Diagnosis:$(NC)"
	@echo "  make case-a                  # No FVGs detected"
	@echo "  make case-c                  # FVGs armed but no orders"
	@echo "  make case-d                  # Orders placed but no fills"
	@echo ""
	@echo "$(WHITE)Advanced Analysis:$(NC)"
	@echo "  make fvg-analysis-verbose    # Detailed Python analysis"
	@echo "  make fvg-analysis-export     # Export to JSON"
	@echo ""
	@echo "$(WHITE)Bot Operations:$(NC)"
	@echo "  make fvg-start               # Start in dry-run mode"
	@echo "  make fvg-status              # Check bot status"
	@echo "  make fvg-stop                # Stop bot"
	@echo ""
	@echo "$(WHITE)Monitoring:$(NC)"
	@echo "  make logs-tail               # Watch live logs"
	@echo "  make logs-blocks             # See blocking reasons"
	@echo "  make heartbeat               # Check bot heartbeat"

# Phony target to ensure make targets always run
.PHONY: all clean install test