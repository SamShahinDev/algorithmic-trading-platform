#!/bin/bash

#
# FVG Triage Toolkit - Bash Script
# Diagnoses "no trades" issues by analyzing the FVG pipeline funnel
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"
FVG_LOG="$LOGS_DIR/fvg_bot.log"
FVG_TELEMETRY="$LOGS_DIR/fvg_telemetry.csv"
NQ_HEARTBEAT="$LOGS_DIR/nq_bot.heartbeat.json"

# Default parameters
HOURS_BACK=4
SHOW_CONFIG=false
SHOW_FUNNEL=true
SHOW_NEAR_MISSES=true
SHOW_TOP_BLOCKS=true
SANITY_CHECK=true
VERBOSE=false

usage() {
    cat << EOF
FVG Triage Toolkit - Diagnose "no trades" issues

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --hours HOURS       Hours to look back (default: 4)
    -c, --config            Show configuration echo from logs
    -f, --funnel            Show pipeline funnel stats (default: enabled)
    -n, --near-misses       Show near-miss analysis (default: enabled)
    -b, --blocks            Show top blocking reasons (default: enabled)
    -s, --sanity            Run sanity checks (default: enabled)
    -v, --verbose           Verbose output
    --no-funnel            Disable funnel stats
    --no-near-misses       Disable near-miss analysis
    --no-blocks            Disable blocking analysis
    --no-sanity            Disable sanity checks

Examples:
    $0                      # Full triage with defaults
    $0 -t 8 -v             # 8 hours back, verbose
    $0 -c                  # Show config only
    $0 --no-sanity         # Skip sanity checks

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--hours)
            HOURS_BACK="$2"
            shift 2
            ;;
        -c|--config)
            SHOW_CONFIG=true
            shift
            ;;
        -f|--funnel)
            SHOW_FUNNEL=true
            shift
            ;;
        -n|--near-misses)
            SHOW_NEAR_MISSES=true
            shift
            ;;
        -b|--blocks)
            SHOW_TOP_BLOCKS=true
            shift
            ;;
        -s|--sanity)
            SANITY_CHECK=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-funnel)
            SHOW_FUNNEL=false
            shift
            ;;
        --no-near-misses)
            SHOW_NEAR_MISSES=false
            shift
            ;;
        --no-blocks)
            SHOW_TOP_BLOCKS=false
            shift
            ;;
        --no-sanity)
            SANITY_CHECK=false
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# Check if file exists and is readable
check_file() {
    local file="$1"
    local desc="$2"

    if [[ ! -f "$file" ]]; then
        log_error "$desc not found: $file"
        return 1
    fi

    if [[ ! -r "$file" ]]; then
        log_error "$desc not readable: $file"
        return 1
    fi

    return 0
}

# Get recent log entries within time window
get_recent_logs() {
    local file="$1"
    local hours="$2"

    if [[ ! -f "$file" ]]; then
        echo ""
        return
    fi

    # Get logs from last N hours
    local since_date
    if command -v gdate >/dev/null 2>&1; then
        # macOS with GNU coreutils
        since_date=$(gdate -d "$hours hours ago" --iso-8601=seconds)
    elif date --version >/dev/null 2>&1; then
        # GNU date
        since_date=$(date -d "$hours hours ago" --iso-8601=seconds)
    else
        # BSD date (macOS default)
        since_date=$(date -u -v-"${hours}H" '+%Y-%m-%dT%H:%M:%S')
    fi

    # Extract logs since the specified time
    awk -v since="$since_date" '
        /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}/ {
            if ($1 >= since) print $0
        }
    ' "$file" 2>/dev/null || cat "$file"
}

# Show configuration from logs
show_config() {
    log_section "FVG Configuration Echo"

    if ! check_file "$FVG_LOG" "FVG log file"; then
        log_warn "Cannot show config - log file missing"
        return
    fi

    local recent_logs
    recent_logs=$(get_recent_logs "$FVG_LOG" "$HOURS_BACK")

    if [[ -z "$recent_logs" ]]; then
        log_warn "No recent logs found in the last $HOURS_BACK hours"
        return
    fi

    # Extract CONFIG_ECHO lines
    local config_lines
    config_lines=$(echo "$recent_logs" | grep "CONFIG_ECHO:" | tail -50)

    if [[ -z "$config_lines" ]]; then
        log_warn "No CONFIG_ECHO lines found in recent logs"
        return
    fi

    echo "$config_lines" | while IFS= read -r line; do
        # Extract the config part after CONFIG_ECHO:
        local config_part
        config_part=$(echo "$line" | sed 's/.*CONFIG_ECHO: //')
        echo -e "${CYAN}  $config_part${NC}"
    done
}

# Show pipeline funnel stats
show_funnel_stats() {
    log_section "Pipeline Funnel Analysis"

    if ! check_file "$FVG_LOG" "FVG log file"; then
        log_warn "Cannot show funnel stats - log file missing"
        return
    fi

    local recent_logs
    recent_logs=$(get_recent_logs "$FVG_LOG" "$HOURS_BACK")

    if [[ -z "$recent_logs" ]]; then
        log_warn "No recent logs found"
        return
    fi

    # Extract telemetry and rollup data
    local telemetry_lines
    telemetry_lines=$(echo "$recent_logs" | grep "TELEMETRY:" | tail -10)

    local rollup_lines
    rollup_lines=$(echo "$recent_logs" | grep "ROLLUP" | tail -10)

    echo -e "${WHITE}Recent Telemetry Snapshots:${NC}"
    if [[ -n "$telemetry_lines" ]]; then
        echo "$telemetry_lines" | tail -3 | while IFS= read -r line; do
            # Parse JSON telemetry
            local json_part
            json_part=$(echo "$line" | sed 's/.*TELEMETRY: //')
            echo "  $json_part" | python3 -m json.tool 2>/dev/null || echo "  $json_part"
        done
    else
        log_warn "No telemetry data found"
    fi

    echo -e "\n${WHITE}Recent Rollup Stats:${NC}"
    if [[ -n "$rollup_lines" ]]; then
        echo "$rollup_lines" | tail -3 | while IFS= read -r line; do
            echo -e "${CYAN}  $line${NC}"
        done
    else
        log_warn "No rollup data found"
    fi

    # Count key events
    local bars_seen
    bars_seen=$(echo "$recent_logs" | grep -c "bars_seen" || echo "0")

    local fvg_detected
    fvg_detected=$(echo "$recent_logs" | grep -c "FVG_DETECTED\|sweep_fvg_detected\|trend_fvg_detected" || echo "0")

    local orders_placed
    orders_placed=$(echo "$recent_logs" | grep -c "ENTRY_PLACED\|orders_placed" || echo "0")

    echo -e "\n${WHITE}Event Counts (last $HOURS_BACK hours):${NC}"
    echo -e "  ${CYAN}Bars processed: $bars_seen${NC}"
    echo -e "  ${CYAN}FVGs detected: $fvg_detected${NC}"
    echo -e "  ${CYAN}Orders placed: $orders_placed${NC}"
}

# Show top blocking reasons
show_top_blocks() {
    log_section "Top Blocking Reasons"

    if ! check_file "$FVG_LOG" "FVG log file"; then
        log_warn "Cannot show blocking analysis - log file missing"
        return
    fi

    local recent_logs
    recent_logs=$(get_recent_logs "$FVG_LOG" "$HOURS_BACK")

    if [[ -z "$recent_logs" ]]; then
        log_warn "No recent logs found"
        return
    fi

    echo -e "${WHITE}Blocking Reasons Analysis:${NC}"

    # Count different blocking reasons
    local rsi_blocks
    rsi_blocks=$(echo "$recent_logs" | grep -c "RSI veto\|rsi_range" || echo "0")

    local cooldown_blocks
    cooldown_blocks=$(echo "$recent_logs" | grep -c "cooldown" || echo "0")

    local burst_guard_blocks
    burst_guard_blocks=$(echo "$recent_logs" | grep -c "burst_guard\|Burst guard" || echo "0")

    local displacement_blocks
    displacement_blocks=$(echo "$recent_logs" | grep -c "displacement_body" || echo "0")

    local gap_blocks
    gap_blocks=$(echo "$recent_logs" | grep -c "gap_min" || echo "0")

    local daily_cap_blocks
    daily_cap_blocks=$(echo "$recent_logs" | grep -c "daily_trade_cap\|Daily trade cap" || echo "0")

    local data_stale_blocks
    data_stale_blocks=$(echo "$recent_logs" | grep -c "Data too stale" || echo "0")

    echo -e "  ${CYAN}RSI veto blocks: $rsi_blocks${NC}"
    echo -e "  ${CYAN}Cooldown blocks: $cooldown_blocks${NC}"
    echo -e "  ${CYAN}Burst guard blocks: $burst_guard_blocks${NC}"
    echo -e "  ${CYAN}Displacement blocks: $displacement_blocks${NC}"
    echo -e "  ${CYAN}Gap size blocks: $gap_blocks${NC}"
    echo -e "  ${CYAN}Daily cap blocks: $daily_cap_blocks${NC}"
    echo -e "  ${CYAN}Stale data blocks: $data_stale_blocks${NC}"

    # Show most recent blocking examples
    echo -e "\n${WHITE}Recent Blocking Examples:${NC}"
    local block_examples
    block_examples=$(echo "$recent_logs" | grep -E "(veto|blocked|Burst guard|too stale|Daily trade cap)" | tail -5)

    if [[ -n "$block_examples" ]]; then
        echo "$block_examples" | while IFS= read -r line; do
            echo -e "  ${YELLOW}$line${NC}"
        done
    else
        echo -e "  ${GREEN}No recent blocking examples found${NC}"
    fi
}

# Show near-miss analysis
show_near_misses() {
    log_section "Near-Miss Analysis"

    if ! check_file "$FVG_LOG" "FVG log file"; then
        log_warn "Cannot show near-miss analysis - log file missing"
        return
    fi

    local recent_logs
    recent_logs=$(get_recent_logs "$FVG_LOG" "$HOURS_BACK")

    if [[ -z "$recent_logs" ]]; then
        log_warn "No recent logs found"
        return
    fi

    echo -e "${WHITE}Looking for FVGs that almost qualified:${NC}"

    # Look for FVG detection attempts
    local fvg_candidates
    fvg_candidates=$(echo "$recent_logs" | grep -E "(displacement_pass|gap_pass|score_pass|FVG_TELEMETRY)" | tail -10)

    if [[ -n "$fvg_candidates" ]]; then
        echo "$fvg_candidates" | while IFS= read -r line; do
            echo -e "  ${CYAN}$line${NC}"
        done
    else
        echo -e "  ${YELLOW}No FVG candidate telemetry found${NC}"
    fi

    # Look for high-quality patterns that were filtered out
    local high_quality_filtered
    high_quality_filtered=$(echo "$recent_logs" | grep -E "(quality.*0\.[6-9]|confidence.*0\.[6-9])" | tail -5)

    if [[ -n "$high_quality_filtered" ]]; then
        echo -e "\n${WHITE}High-quality patterns that were filtered:${NC}"
        echo "$high_quality_filtered" | while IFS= read -r line; do
            echo -e "  ${YELLOW}$line${NC}"
        done
    fi
}

# Run sanity checks
run_sanity_checks() {
    log_section "Sanity Check & Replay Validation"

    echo -e "${WHITE}Environment Checks:${NC}"

    # Check if FVG bot is running
    local bot_running=false
    if pgrep -f "fvg_runner.py" >/dev/null; then
        log_success "FVG bot process is running"
        bot_running=true
    else
        log_warn "FVG bot process not found"
    fi

    # Check heartbeat file
    if check_file "$NQ_HEARTBEAT" "NQ bot heartbeat"; then
        local heartbeat_data
        heartbeat_data=$(cat "$NQ_HEARTBEAT" 2>/dev/null)

        if [[ -n "$heartbeat_data" ]]; then
            echo -e "${CYAN}  Heartbeat: $heartbeat_data${NC}"

            # Check if bot state is active
            if echo "$heartbeat_data" | grep -q '"state": "stopped"'; then
                log_warn "Bot state shows 'stopped'"
            elif echo "$heartbeat_data" | grep -q '"trading_enabled": false'; then
                log_warn "Trading is disabled"
            fi
        fi
    fi

    # Check log file sizes and timestamps
    echo -e "\n${WHITE}File Status:${NC}"
    if [[ -f "$FVG_LOG" ]]; then
        local log_size
        log_size=$(wc -c < "$FVG_LOG" 2>/dev/null || echo "0")
        local log_lines
        log_lines=$(wc -l < "$FVG_LOG" 2>/dev/null || echo "0")
        echo -e "  ${CYAN}FVG log: $log_lines lines, $log_size bytes${NC}"

        # Check last modification time
        local last_mod
        if command -v stat >/dev/null 2>&1; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                last_mod=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$FVG_LOG" 2>/dev/null || echo "unknown")
            else
                last_mod=$(stat -c "%y" "$FVG_LOG" 2>/dev/null | cut -d'.' -f1 || echo "unknown")
            fi
            echo -e "  ${CYAN}Last modified: $last_mod${NC}"
        fi
    fi

    if [[ -f "$FVG_TELEMETRY" ]]; then
        local tel_lines
        tel_lines=$(wc -l < "$FVG_TELEMETRY" 2>/dev/null || echo "0")
        echo -e "  ${CYAN}Telemetry CSV: $tel_lines lines${NC}"
    fi

    # Configuration sanity checks
    echo -e "\n${WHITE}Configuration Sanity:${NC}"

    # Check for common misconfigurations by looking at CONFIG_ECHO
    local recent_logs
    recent_logs=$(get_recent_logs "$FVG_LOG" "$HOURS_BACK")

    if [[ -n "$recent_logs" ]]; then
        # Check tick size
        local tick_size
        tick_size=$(echo "$recent_logs" | grep "CONFIG_ECHO: tick_size" | tail -1 | grep -o "[0-9.]*")
        if [[ "$tick_size" != "0.25" ]]; then
            log_warn "Tick size may be incorrect: $tick_size (should be 0.25 for NQ)"
        else
            log_success "Tick size correct: $tick_size"
        fi

        # Check min_gap_ticks
        local min_gap_ticks
        min_gap_ticks=$(echo "$recent_logs" | grep "CONFIG_ECHO: min_gap_ticks" | tail -1 | grep -o "[0-9]*")
        if [[ -n "$min_gap_ticks" && "$min_gap_ticks" -gt 2 ]]; then
            log_warn "min_gap_ticks may be too restrictive: $min_gap_ticks (consider 1-2)"
        elif [[ -n "$min_gap_ticks" ]]; then
            log_success "min_gap_ticks looks reasonable: $min_gap_ticks"
        fi

        # Check if live data is enabled appropriately
        local live_data
        live_data=$(echo "$recent_logs" | grep "CONFIG_ECHO: live_market_data" | tail -1 | grep -o "True\|False")
        if [[ "$live_data" == "True" ]]; then
            log_warn "Live market data enabled - ensure this is intended for your account type"
        elif [[ "$live_data" == "False" ]]; then
            log_success "Using practice/sim data feed"
        fi
    fi

    echo -e "\n${WHITE}Quick Diagnosis Guide:${NC}"
    echo -e "${CYAN}Case A: No FVGs detected at all${NC}"
    echo -e "  → Check min_gap_ticks (should be 1-2 for NQ)"
    echo -e "  → Check min_displacement settings"
    echo -e "  → Verify market data is flowing"

    echo -e "\n${CYAN}Case B: FVGs detected but none get armed${NC}"
    echo -e "  → Check quality threshold (min_quality)"
    echo -e "  → Review displacement and body fraction requirements"

    echo -e "\n${CYAN}Case C: FVGs armed but no orders placed${NC}"
    echo -e "  → Check RSI ranges (often too restrictive)"
    echo -e "  → Review burst guard and cooldown settings"
    echo -e "  → Check daily trade cap"

    echo -e "\n${CYAN}Case D: Orders placed but no fills${NC}"
    echo -e "  → Check entry percentages (50% vs 62%)"
    echo -e "  → Review TTL and cancel_if_runs settings"
    echo -e "  → Verify price levels are reasonable"

    echo -e "\n${CYAN}Case E: Data issues${NC}"
    echo -e "  → Check 'Data too stale' messages"
    echo -e "  → Verify contract_id is current"
    echo -e "  → Check broker connection"

    echo -e "\n${CYAN}Case F: No market activity${NC}"
    echo -e "  → Check if market is open"
    echo -e "  → Review time restrictions"
    echo -e "  → Consider market volatility requirements"
}

# Main execution
main() {
    log_info "FVG Triage Toolkit - Analyzing last $HOURS_BACK hours"
    log_info "Project root: $PROJECT_ROOT"

    # Show configuration if requested
    if [[ "$SHOW_CONFIG" == "true" ]]; then
        show_config
    fi

    # Show funnel stats
    if [[ "$SHOW_FUNNEL" == "true" ]]; then
        show_funnel_stats
    fi

    # Show top blocking reasons
    if [[ "$SHOW_TOP_BLOCKS" == "true" ]]; then
        show_top_blocks
    fi

    # Show near-miss analysis
    if [[ "$SHOW_NEAR_MISSES" == "true" ]]; then
        show_near_misses
    fi

    # Run sanity checks
    if [[ "$SANITY_CHECK" == "true" ]]; then
        run_sanity_checks
    fi

    log_info "Triage analysis complete"

    # Suggest next steps
    log_section "Suggested Next Steps"
    echo -e "${WHITE}Based on the analysis above:${NC}"
    echo -e "1. If no FVGs detected: Review gap/displacement requirements"
    echo -e "2. If FVGs detected but blocked: Focus on most frequent blocking reason"
    echo -e "3. If orders placed but no fills: Check entry levels and timeouts"
    echo -e "4. For detailed investigation: ${CYAN}python3 tools/fvg_triage.py${NC}"
    echo -e "5. For configuration tweaks: Edit ${CYAN}nq_bot/pattern_config.py${NC}"
}

# Run main function
main "$@"