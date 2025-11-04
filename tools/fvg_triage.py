#!/usr/bin/env python3

"""
FVG Triage Toolkit - Python Advanced Analysis
Comprehensive diagnosis of "no trades" issues with detailed statistical analysis
"""

import os
import sys
import json
import csv
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "nq_bot"))

try:
    from nq_bot.pattern_config import FVG, CONTRACT_ID, LIVE_MARKET_DATA
except ImportError:
    # Fallback if imports fail
    FVG = {}
    CONTRACT_ID = "CON.F.US.ENQ.U25"
    LIVE_MARKET_DATA = False

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[0;37m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

class FVGTriageAnalyzer:
    """Advanced FVG triage analysis with statistical insights"""

    def __init__(self, project_root: Path, hours_back: int = 4, verbose: bool = False):
        self.project_root = project_root
        self.hours_back = hours_back
        self.verbose = verbose

        # File paths
        self.logs_dir = project_root / "logs"
        self.fvg_log = self.logs_dir / "fvg_bot.log"
        self.fvg_telemetry = self.logs_dir / "fvg_telemetry.csv"
        self.nq_heartbeat = self.logs_dir / "nq_bot.heartbeat.json"

        # Analysis data
        self.log_data: List[str] = []
        self.telemetry_data: List[Dict] = []
        self.config_data: Dict = {}
        self.timeline_events: List[Dict] = []

    def log_info(self, message: str):
        """Log info message with color"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

    def log_warn(self, message: str):
        """Log warning message with color"""
        print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")

    def log_error(self, message: str):
        """Log error message with color"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

    def log_success(self, message: str):
        """Log success message with color"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

    def log_section(self, title: str):
        """Log section header"""
        print(f"\n{Colors.PURPLE}=== {title} ==={Colors.NC}")

    def load_log_data(self) -> bool:
        """Load and parse log data"""
        if not self.fvg_log.exists():
            self.log_error(f"FVG log file not found: {self.fvg_log}")
            return False

        try:
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=self.hours_back)

            with open(self.fvg_log, 'r') as f:
                lines = f.readlines()

            # Filter lines by timestamp
            self.log_data = []
            for line in lines:
                # Extract timestamp from log line
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    try:
                        line_time = datetime.fromisoformat(timestamp_match.group(1))
                        if line_time >= time_threshold:
                            self.log_data.append(line.strip())
                    except ValueError:
                        continue

            self.log_info(f"Loaded {len(self.log_data)} log lines from last {self.hours_back} hours")
            return True

        except Exception as e:
            self.log_error(f"Failed to load log data: {e}")
            return False

    def load_telemetry_data(self) -> bool:
        """Load and parse telemetry CSV data"""
        if not self.fvg_telemetry.exists():
            self.log_warn("Telemetry CSV file not found")
            return False

        try:
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=self.hours_back)

            with open(self.fvg_telemetry, 'r') as f:
                reader = csv.DictReader(f)
                self.telemetry_data = []

                for row in reader:
                    try:
                        # Parse timestamp
                        row_time = datetime.fromisoformat(row['timestamp'])
                        if row_time >= time_threshold:
                            self.telemetry_data.append(row)
                    except (ValueError, KeyError):
                        continue

            self.log_info(f"Loaded {len(self.telemetry_data)} telemetry records")
            return True

        except Exception as e:
            self.log_error(f"Failed to load telemetry data: {e}")
            return False

    def parse_config_echo(self):
        """Parse CONFIG_ECHO lines from logs"""
        self.config_data = {}

        for line in self.log_data:
            if "CONFIG_ECHO:" in line:
                # Extract config key-value pairs
                match = re.search(r'CONFIG_ECHO:\s*([^:]+):\s*(.+)', line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()

                    # Try to parse value as JSON/Python literal
                    try:
                        # Handle boolean values
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        # Handle numeric values
                        elif value.replace('.', '').replace('-', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                        # Handle list/dict values
                        elif value.startswith(('[', '{')):
                            value = eval(value)
                    except:
                        pass  # Keep as string if parsing fails

                    self.config_data[key] = value

    def analyze_pipeline_funnel(self) -> Dict[str, Any]:
        """Analyze the FVG detection pipeline funnel"""
        funnel_stats = {
            'bars_processed': 0,
            'fvg_candidates': 0,
            'displacement_pass': 0,
            'gap_pass': 0,
            'score_pass': 0,
            'fresh_fvgs': 0,
            'armed_fvgs': 0,
            'orders_placed': 0,
            'fills': 0,
            'blocking_reasons': defaultdict(int),
            'telemetry_snapshots': []
        }

        # Parse telemetry JSON lines
        for line in self.log_data:
            if "TELEMETRY:" in line:
                try:
                    # Extract JSON part
                    json_start = line.find('{')
                    if json_start >= 0:
                        json_data = json.loads(line[json_start:])
                        funnel_stats['telemetry_snapshots'].append(json_data)

                        # Update counters
                        funnel_stats['bars_processed'] = json_data.get('bars_seen', 0)
                        funnel_stats['fresh_fvgs'] = json_data.get('fresh', 0)
                        funnel_stats['armed_fvgs'] = json_data.get('armed', 0)
                        funnel_stats['orders_placed'] = json_data.get('orders_placed', 0)
                        funnel_stats['fills'] = json_data.get('fills', 0)

                        # Blocking reasons
                        blocked = json_data.get('blocked', {})
                        for reason, count in blocked.items():
                            funnel_stats['blocking_reasons'][reason] += count

                except json.JSONDecodeError:
                    continue

        # Parse individual event counts
        for line in self.log_data:
            if "displacement_pass" in line:
                match = re.search(r'displacement_pass=(\d+)', line)
                if match:
                    funnel_stats['displacement_pass'] = max(funnel_stats['displacement_pass'], int(match.group(1)))

            if "gap_pass" in line:
                match = re.search(r'gap_pass=(\d+)', line)
                if match:
                    funnel_stats['gap_pass'] = max(funnel_stats['gap_pass'], int(match.group(1)))

            if "score_pass" in line:
                match = re.search(r'score_pass=(\d+)', line)
                if match:
                    funnel_stats['score_pass'] = max(funnel_stats['score_pass'], int(match.group(1)))

        return funnel_stats

    def analyze_blocking_patterns(self) -> Dict[str, Any]:
        """Analyze what's blocking trades"""
        blocking_analysis = {
            'rsi_blocks': [],
            'cooldown_blocks': 0,
            'burst_guard_blocks': [],
            'data_stale_blocks': 0,
            'daily_cap_blocks': 0,
            'displacement_blocks': 0,
            'gap_blocks': 0,
            'quality_blocks': 0,
            'recent_examples': []
        }

        for line in self.log_data:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
            timestamp = timestamp_match.group(1) if timestamp_match else "unknown"

            if "RSI veto" in line:
                # Extract RSI details
                rsi_match = re.search(r'RSI (\d+\.?\d*)', line)
                range_match = re.search(r'range \[(\d+),\s*(\d+)\]', line)
                direction_match = re.search(r'(Long|Short)', line)

                blocking_analysis['rsi_blocks'].append({
                    'timestamp': timestamp,
                    'rsi_value': float(rsi_match.group(1)) if rsi_match else None,
                    'range': [int(range_match.group(1)), int(range_match.group(2))] if range_match else None,
                    'direction': direction_match.group(1).lower() if direction_match else None,
                    'line': line
                })

            elif "cooldown" in line.lower():
                blocking_analysis['cooldown_blocks'] += 1

            elif "Burst guard" in line:
                # Extract burst guard details
                time_match = re.search(r'only (\d+\.?\d*)s since last', line)
                need_match = re.search(r'need (\d+)s', line)
                direction_match = re.search(r'(long|short) trade blocked', line)

                blocking_analysis['burst_guard_blocks'].append({
                    'timestamp': timestamp,
                    'time_since_last': float(time_match.group(1)) if time_match else None,
                    'time_needed': int(need_match.group(1)) if need_match else None,
                    'direction': direction_match.group(1) if direction_match else None,
                    'line': line
                })

            elif "Data too stale" in line:
                blocking_analysis['data_stale_blocks'] += 1

            elif "Daily trade cap" in line:
                blocking_analysis['daily_cap_blocks'] += 1

            elif "displacement_body" in line:
                blocking_analysis['displacement_blocks'] += 1

            elif "gap_min" in line:
                blocking_analysis['gap_blocks'] += 1

            # Collect recent blocking examples
            if any(keyword in line.lower() for keyword in ['veto', 'blocked', 'too stale', 'daily trade cap']):
                blocking_analysis['recent_examples'].append({
                    'timestamp': timestamp,
                    'line': line
                })

        # Keep only recent examples (last 10)
        blocking_analysis['recent_examples'] = blocking_analysis['recent_examples'][-10:]

        return blocking_analysis

    def analyze_near_misses(self) -> Dict[str, Any]:
        """Analyze patterns that almost qualified"""
        near_miss_analysis = {
            'high_quality_filtered': [],
            'displacement_near_misses': [],
            'gap_near_misses': [],
            'rsi_near_misses': [],
            'timing_near_misses': []
        }

        for line in self.log_data:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
            timestamp = timestamp_match.group(1) if timestamp_match else "unknown"

            # Look for high-quality patterns that were filtered
            if "quality" in line and re.search(r'0\.[6-9]\d*', line):
                quality_match = re.search(r'quality.*?(\d+\.\d+)', line)
                near_miss_analysis['high_quality_filtered'].append({
                    'timestamp': timestamp,
                    'quality': float(quality_match.group(1)) if quality_match else None,
                    'line': line
                })

            # RSI near misses (within 5 points of range)
            if "RSI veto" in line:
                rsi_match = re.search(r'RSI (\d+\.?\d*)', line)
                range_match = re.search(r'range \[(\d+),\s*(\d+)\]', line)

                if rsi_match and range_match:
                    rsi_value = float(rsi_match.group(1))
                    range_min = int(range_match.group(1))
                    range_max = int(range_match.group(2))

                    # Check if within 5 points of acceptable range
                    if (range_min - 5 <= rsi_value < range_min) or (range_max < rsi_value <= range_max + 5):
                        near_miss_analysis['rsi_near_misses'].append({
                            'timestamp': timestamp,
                            'rsi_value': rsi_value,
                            'range': [range_min, range_max],
                            'miss_by': min(abs(rsi_value - range_min), abs(rsi_value - range_max)),
                            'line': line
                        })

        return near_miss_analysis

    def analyze_configuration_issues(self) -> Dict[str, Any]:
        """Analyze potential configuration issues"""
        config_issues = {
            'issues': [],
            'warnings': [],
            'recommendations': []
        }

        # Check tick size
        tick_size = self.config_data.get('tick_size', 0.25)
        if tick_size != 0.25:
            config_issues['issues'].append(f"Incorrect tick_size: {tick_size} (should be 0.25 for NQ)")

        # Check min_gap_ticks
        min_gap_ticks = self.config_data.get('min_gap_ticks', 1)
        if min_gap_ticks > 3:
            config_issues['warnings'].append(f"min_gap_ticks may be too restrictive: {min_gap_ticks}")

        # Check RSI ranges
        rsi_long_range = self.config_data.get('rsi_long_range', [50, 80])
        rsi_short_range = self.config_data.get('rsi_short_range', [20, 50])

        if isinstance(rsi_long_range, list) and len(rsi_long_range) == 2:
            if rsi_long_range[1] - rsi_long_range[0] < 20:
                config_issues['warnings'].append(f"RSI long range may be too narrow: {rsi_long_range}")

        if isinstance(rsi_short_range, list) and len(rsi_short_range) == 2:
            if rsi_short_range[1] - rsi_short_range[0] < 20:
                config_issues['warnings'].append(f"RSI short range may be too narrow: {rsi_short_range}")

        # Check quality threshold
        min_quality = self.config_data.get('min_quality', 0.55)
        if min_quality > 0.7:
            config_issues['warnings'].append(f"min_quality may be too high: {min_quality}")

        # Check daily trade cap
        daily_trade_cap = self.config_data.get('daily_trade_cap', 12)
        if daily_trade_cap < 5:
            config_issues['warnings'].append(f"daily_trade_cap may be too low: {daily_trade_cap}")

        # Check burst guard
        burst_guard_seconds = self.config_data.get('burst_guard_seconds', 120)
        if burst_guard_seconds > 300:
            config_issues['warnings'].append(f"burst_guard_seconds may be too long: {burst_guard_seconds}")

        # Generate recommendations based on analysis
        funnel_stats = self.analyze_pipeline_funnel()

        if funnel_stats['bars_processed'] == 0:
            config_issues['recommendations'].append("No bars processed - check data feed connection")
        elif funnel_stats['fresh_fvgs'] == 0:
            config_issues['recommendations'].append("No FVGs detected - consider lowering min_gap_ticks or displacement requirements")
        elif funnel_stats['armed_fvgs'] == 0:
            config_issues['recommendations'].append("FVGs detected but none armed - check quality threshold and lifecycle settings")
        elif funnel_stats['orders_placed'] == 0:
            config_issues['recommendations'].append("FVGs armed but no orders - check RSI ranges and blocking conditions")

        return config_issues

    def generate_timeline_analysis(self) -> List[Dict]:
        """Generate timeline of events for detailed analysis"""
        timeline = []

        for line in self.log_data:
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
            if not timestamp_match:
                continue

            timestamp = timestamp_match.group(1)

            # Categorize events
            event_type = "unknown"
            details = {}

            if "FVG_DETECTED" in line:
                event_type = "fvg_detected"
            elif "ENTRY_PLACED" in line:
                event_type = "entry_placed"
            elif "EXEC_FILL" in line:
                event_type = "exec_fill"
            elif "RSI veto" in line:
                event_type = "rsi_block"
            elif "Burst guard" in line:
                event_type = "burst_guard_block"
            elif "cooldown" in line.lower():
                event_type = "cooldown_block"
            elif "Data too stale" in line:
                event_type = "data_stale"
            elif "TELEMETRY:" in line:
                event_type = "telemetry"
                # Parse telemetry data
                json_start = line.find('{')
                if json_start >= 0:
                    try:
                        details = json.loads(line[json_start:])
                    except json.JSONDecodeError:
                        pass

            timeline.append({
                'timestamp': timestamp,
                'event_type': event_type,
                'details': details,
                'raw_line': line
            })

        return sorted(timeline, key=lambda x: x['timestamp'])

    def print_summary_report(self):
        """Print comprehensive summary report"""
        self.log_section("FVG Triage Analysis Summary")

        # Basic stats
        funnel_stats = self.analyze_pipeline_funnel()
        blocking_analysis = self.analyze_blocking_patterns()
        config_issues = self.analyze_configuration_issues()

        print(f"{Colors.WHITE}Analysis Period:{Colors.NC} Last {self.hours_back} hours")
        print(f"{Colors.WHITE}Log Lines Analyzed:{Colors.NC} {len(self.log_data)}")
        print(f"{Colors.WHITE}Telemetry Records:{Colors.NC} {len(self.telemetry_data)}")

        # Pipeline Funnel
        self.log_section("Pipeline Funnel Stats")
        print(f"{Colors.CYAN}  Bars Processed: {funnel_stats['bars_processed']}{Colors.NC}")
        print(f"{Colors.CYAN}  Displacement Pass: {funnel_stats['displacement_pass']}{Colors.NC}")
        print(f"{Colors.CYAN}  Gap Pass: {funnel_stats['gap_pass']}{Colors.NC}")
        print(f"{Colors.CYAN}  Score Pass: {funnel_stats['score_pass']}{Colors.NC}")
        print(f"{Colors.CYAN}  Fresh FVGs: {funnel_stats['fresh_fvgs']}{Colors.NC}")
        print(f"{Colors.CYAN}  Armed FVGs: {funnel_stats['armed_fvgs']}{Colors.NC}")
        print(f"{Colors.CYAN}  Orders Placed: {funnel_stats['orders_placed']}{Colors.NC}")
        print(f"{Colors.CYAN}  Fills: {funnel_stats['fills']}{Colors.NC}")

        # Blocking Analysis
        self.log_section("Top Blocking Reasons")
        for reason, count in sorted(funnel_stats['blocking_reasons'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"{Colors.YELLOW}  {reason}: {count}{Colors.NC}")

        total_rsi_blocks = len(blocking_analysis['rsi_blocks'])
        total_burst_blocks = len(blocking_analysis['burst_guard_blocks'])
        print(f"{Colors.YELLOW}  RSI blocks: {total_rsi_blocks}{Colors.NC}")
        print(f"{Colors.YELLOW}  Burst guard blocks: {total_burst_blocks}{Colors.NC}")
        print(f"{Colors.YELLOW}  Data stale blocks: {blocking_analysis['data_stale_blocks']}{Colors.NC}")

        # Configuration Issues
        if config_issues['issues'] or config_issues['warnings']:
            self.log_section("Configuration Issues")

            for issue in config_issues['issues']:
                print(f"{Colors.RED}  ISSUE: {issue}{Colors.NC}")

            for warning in config_issues['warnings']:
                print(f"{Colors.YELLOW}  WARNING: {warning}{Colors.NC}")

        # Recommendations
        if config_issues['recommendations']:
            self.log_section("Recommendations")
            for i, rec in enumerate(config_issues['recommendations'], 1):
                print(f"{Colors.GREEN}  {i}. {rec}{Colors.NC}")

    def print_detailed_analysis(self):
        """Print detailed analysis with specific insights"""

        # Near-miss analysis
        near_misses = self.analyze_near_misses()
        if near_misses['rsi_near_misses']:
            self.log_section("RSI Near Misses")
            for miss in near_misses['rsi_near_misses'][-5:]:  # Last 5
                print(f"{Colors.CYAN}  {miss['timestamp']}: RSI {miss['rsi_value']} (range {miss['range']}, miss by {miss['miss_by']:.1f}){Colors.NC}")

        # Configuration echo
        if self.config_data:
            self.log_section("Current Configuration")
            key_configs = [
                'min_gap_ticks', 'min_quality', 'tick_size', 'daily_trade_cap',
                'burst_guard_seconds', 'rsi_long_range', 'rsi_short_range'
            ]

            for key in key_configs:
                if key in self.config_data:
                    print(f"{Colors.CYAN}  {key}: {self.config_data[key]}{Colors.NC}")

        # Timeline analysis (last 10 events)
        timeline = self.generate_timeline_analysis()
        if timeline:
            self.log_section("Recent Event Timeline")
            for event in timeline[-10:]:
                color = Colors.WHITE
                if event['event_type'] == 'fvg_detected':
                    color = Colors.GREEN
                elif 'block' in event['event_type']:
                    color = Colors.YELLOW
                elif event['event_type'] == 'entry_placed':
                    color = Colors.BLUE

                print(f"{color}  {event['timestamp']}: {event['event_type']}{Colors.NC}")
                if self.verbose and event['details']:
                    print(f"    {event['details']}")

    def export_analysis(self, output_file: str):
        """Export analysis to JSON file"""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'hours_back': self.hours_back,
            'pipeline_funnel': self.analyze_pipeline_funnel(),
            'blocking_analysis': self.analyze_blocking_patterns(),
            'near_misses': self.analyze_near_misses(),
            'config_issues': self.analyze_configuration_issues(),
            'config_data': self.config_data,
            'timeline': self.generate_timeline_analysis()
        }

        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        self.log_success(f"Analysis exported to {output_file}")

    def run_analysis(self):
        """Run complete analysis"""
        self.log_info("Starting FVG triage analysis...")

        # Load data
        if not self.load_log_data():
            return False

        self.load_telemetry_data()  # Optional
        self.parse_config_echo()

        # Print analysis
        self.print_summary_report()

        if self.verbose:
            self.print_detailed_analysis()

        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FVG Triage Toolkit - Advanced Python Analysis")
    parser.add_argument('-t', '--hours', type=int, default=4, help='Hours to look back (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-o', '--output', help='Export analysis to JSON file')
    parser.add_argument('--project-root', help='Project root directory (auto-detected)')

    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Auto-detect based on script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent

    if not project_root.exists():
        print(f"{Colors.RED}[ERROR]{Colors.NC} Project root not found: {project_root}")
        return 1

    # Create analyzer
    analyzer = FVGTriageAnalyzer(project_root, args.hours, args.verbose)

    # Run analysis
    if not analyzer.run_analysis():
        return 1

    # Export if requested
    if args.output:
        analyzer.export_analysis(args.output)

    # Final guidance
    print(f"\n{Colors.PURPLE}=== Diagnostic Guidance ==={Colors.NC}")
    print(f"{Colors.WHITE}For quick triage, also run:{Colors.NC} {Colors.CYAN}bin/fvg_triage.sh{Colors.NC}")
    print(f"{Colors.WHITE}For configuration changes:{Colors.NC} {Colors.CYAN}nq_bot/pattern_config.py{Colors.NC}")
    print(f"{Colors.WHITE}For Makefile shortcuts:{Colors.NC} {Colors.CYAN}make fvg-triage{Colors.NC}")

    return 0

if __name__ == "__main__":
    sys.exit(main())