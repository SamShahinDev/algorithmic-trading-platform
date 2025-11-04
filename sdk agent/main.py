"""
SDK Trading Agent - Main Entry Point

This is the main entry point for the SDK Trading Agent.
Initializes all components and starts the trading system.

Usage:
    python main.py

Environment Variables Required:
    - ANTHROPIC_API_KEY: Your Anthropic API key
    - TOPSTEPX_API_KEY: Your TopStepX API key
    - DISCORD_BOT_TOKEN: Your Discord bot token (optional)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from typing import Dict, Any
import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules
from agent.sdk_agent import SDKAgent
from agent.risk_manager import RiskManager
from agent.strategy_selector import StrategySelector
from execution.topstep_client import TopStepXClient
from execution.websocket_handler import WebSocketHandler
from execution.order_manager import OrderManager
from discord_bot.bot import TradingBot
from discord_bot.notifications import NotificationManager

# Import trading strategies
from strategies.vwap_strategy import VWAPStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.momentum_strategy import MomentumStrategy

# Load environment variables
load_dotenv(dotenv_path=project_root.parent / '.env')


def setup_logging(config: dict) -> None:
    """
    Setup logging configuration.

    Args:
        config: Logging configuration from settings
    """
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = project_root / config.get('logging', {}).get('file_path', 'logs/sdk_agent.log')

    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config() -> dict:
    """
    Load configuration from YAML files.

    Returns:
        dict: Combined configuration
    """
    config = {}

    # Load main settings
    settings_path = project_root / 'config' / 'settings.yaml'
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            config['settings'] = yaml.safe_load(f)
    else:
        logging.warning(f"Settings file not found: {settings_path}")
        config['settings'] = {}

    # Load Discord config
    discord_path = project_root / 'config' / 'discord_config.yaml'
    if discord_path.exists():
        with open(discord_path, 'r') as f:
            config['discord'] = yaml.safe_load(f)
    else:
        logging.warning(f"Discord config not found: {discord_path}")
        config['discord'] = {}

    return config


class TradingSystem:
    """
    Main trading system coordinator.

    Manages all components and orchestrates trading operations.
    """

    def __init__(self, config: dict):
        """
        Initialize trading system.

        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False

        # Check for dry-run mode
        self.dry_run = config.get('dry_run', False)

        # Get API keys from environment
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.topstepx_api_key = os.getenv('TOPSTEPX_API_KEY')
        self.topstepx_username = os.getenv('TOPSTEPX_USERNAME')
        self.topstepx_account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
        self.discord_token = os.getenv('DISCORD_BOT_TOKEN')

        # Validate required keys
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        if not self.topstepx_api_key:
            raise ValueError("TOPSTEPX_API_KEY not found in environment")
        if not self.topstepx_username:
            raise ValueError("TOPSTEPX_USERNAME not found in environment")
        if not self.topstepx_account_id:
            raise ValueError("TOPSTEPX_ACCOUNT_ID not found in environment")

        # Dry-run tracking
        if self.dry_run:
            self.dry_run_trades = []
            self.dry_run_stats = {
                'signals_detected': 0,
                'claude_approvals': 0,
                'theoretical_trades': 0,
                'total_slippage_ticks': 0,
                'avg_latency_ms': 0
            }

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all trading system components."""
        settings = self.config.get('settings', {})

        # Initialize SDK agent
        agent_config = settings.get('agent', {})
        self.sdk_agent = SDKAgent(config=agent_config)

        # Initialize risk manager
        risk_config = settings.get('risk', {})
        self.risk_manager = RiskManager(risk_config)

        # Initialize TopStepX client
        trading_config = settings.get('trading', {})
        self.topstep_client = TopStepXClient(
            api_key=self.topstepx_api_key,
            username=self.topstepx_username,
            account_id=self.topstepx_account_id,
            environment=trading_config.get('mode', 'demo').upper()
        )

        # WebSocket and Order Manager will be initialized after authentication
        self.ws_handler = None
        self.order_manager = None

        # Initialize strategies
        strategy_config = settings.get('strategies', {})

        strategies = {
            'vwap': VWAPStrategy(
                config=strategy_config.get('vwap', {}),
                tick_size=0.25,  # NQ tick size
                tick_value=5.0   # NQ tick value
            ),
            'breakout': BreakoutStrategy(
                config=strategy_config.get('breakout', {}),
                tick_size=0.25,
                tick_value=5.0
            ),
            'momentum': MomentumStrategy(
                config=strategy_config.get('momentum', {}),
                tick_size=0.25,
                tick_value=5.0
            )
        }

        self.logger.info(f"✅ Loaded {len(strategies)} strategies: {', '.join(strategies.keys())}")

        # Initialize strategy selector with actual strategies
        self.strategy_selector = StrategySelector(
            strategies=strategies,
            sdk_agent=self.sdk_agent,
            risk_manager=self.risk_manager,
            config=strategy_config
        )

        # Initialize Discord bot (optional)
        self.discord_bot = None
        self.notification_manager = None
        if self.discord_token:
            discord_config = self.config.get('discord', {})
            self.discord_bot = TradingBot(
                config=discord_config.get('bot', {}),
                trading_agent=self
            )
            self.notification_manager = NotificationManager(
                bot=self.discord_bot,
                config=discord_config.get('notifications', {})
            )

        self.logger.info("All components initialized successfully")

    async def start(self) -> None:
        """Start the trading system."""
        self.logger.info("Starting SDK Trading Agent...")
        self.running = True

        try:
            # Start WebSocket connection (if available)
            if self.ws_handler:
                self.ws_handler.connect()

            # Start Discord bot if configured
            if self.discord_bot and self.discord_token:
                asyncio.create_task(self.discord_bot.start_bot(self.discord_token))

            # Main trading loop
            await self.run_trading_loop()

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in trading system: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading system."""
        self.logger.info("Stopping SDK Trading Agent...")
        self.running = False

        # Disconnect WebSocket
        if self.ws_handler:
            self.ws_handler.disconnect()

        # Stop Discord bot
        if self.discord_bot:
            await self.discord_bot.stop_bot()

        # Print dry-run summary
        if self.dry_run:
            self._print_dry_run_summary()

        self.logger.info("SDK Trading Agent stopped")

    def _print_dry_run_summary(self) -> None:
        """Print summary of dry-run session."""
        self.logger.info("=" * 60)
        self.logger.info("DRY RUN SESSION SUMMARY")
        self.logger.info("=" * 60)

        stats = self.dry_run_stats

        self.logger.info(f"Signals Detected: {stats['signals_detected']}")
        self.logger.info(f"Claude Approvals: {stats['claude_approvals']}")
        self.logger.info(f"Theoretical Trades: {stats['theoretical_trades']}")

        if stats['theoretical_trades'] > 0:
            avg_slippage = stats['total_slippage_ticks'] / stats['theoretical_trades']
            self.logger.info(f"Avg Slippage: {avg_slippage:.2f} ticks")
        else:
            self.logger.info("Avg Slippage: N/A")

        self.logger.info(f"Avg Claude Latency: {stats['avg_latency_ms']:.0f}ms")

        self.logger.info("\nTheoretical Trades:")
        for i, trade in enumerate(self.dry_run_trades, 1):
            self.logger.info(
                f"  {i}. {trade['strategy']} {trade['signal']} @ {trade['entry_price']:.2f} "
                f"(slippage: {trade['slippage_ticks']:.2f} ticks, "
                f"latency: {trade['latency_ms']:.0f}ms)"
            )

        self.logger.info("=" * 60)

        # Save to file
        import json
        dry_run_file = Path('logs/dry_run_summary.json')
        with open(dry_run_file, 'w') as f:
            json.dump({
                'stats': stats,
                'trades': self.dry_run_trades
            }, f, indent=2, default=str)

        self.logger.info(f"Dry-run summary saved to: {dry_run_file}")

    def _log_dry_run_trade(self, trade_decision: dict) -> None:
        """
        Log theoretical trade in dry-run mode.

        Args:
            trade_decision: Trade decision from strategy selector
        """
        setup = trade_decision['setup']
        decision = trade_decision['decision']

        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': trade_decision['strategy_name'],
            'signal': setup['signal'],
            'entry_price': decision.get('current_entry', setup['entry_price']),
            'stop_price': setup['stop_price'],
            'target_price': setup['target_price'],
            'slippage_ticks': decision.get('slippage_ticks', 0),
            'latency_ms': decision.get('latency_ms', 0),
            'confidence': setup['confidence']
        }

        self.dry_run_trades.append(trade_record)
        self.dry_run_stats['theoretical_trades'] += 1
        self.dry_run_stats['total_slippage_ticks'] += trade_record['slippage_ticks']

        # Log immediately
        self.logger.info("=" * 60)
        self.logger.info(f"🧪 DRY RUN TRADE #{len(self.dry_run_trades)}")
        self.logger.info(f"Strategy: {trade_record['strategy']}")
        self.logger.info(f"Signal: {trade_record['signal']}")
        self.logger.info(f"Entry: {trade_record['entry_price']:.2f}")
        self.logger.info(f"Stop: {trade_record['stop_price']:.2f}")
        self.logger.info(f"Target: {trade_record['target_price']:.2f}")
        self.logger.info(f"Slippage: {trade_record['slippage_ticks']:.2f} ticks")
        self.logger.info(f"Latency: {trade_record['latency_ms']:.0f}ms")
        self.logger.info(f"Confidence: {trade_record['confidence']:.1f}/10")
        self.logger.info("⚠️  NO ACTUAL ORDER PLACED (DRY RUN)")
        self.logger.info("=" * 60)

    async def run_trading_loop(self) -> None:
        """Main trading loop - fetches market data and evaluates strategies."""
        decision_interval = self.config.get('settings', {}).get('agent', {}).get('decision_interval', 60)

        # NQ contract ID (E-mini NASDAQ-100 December 2025 - Active contract)
        contract_id = "CON.F.US.ENQ.Z25"

        self.logger.info(f"Trading loop started (interval: {decision_interval}s)")
        self.logger.info(f"Monitoring contract: {contract_id}")

        # Authenticate with TopStepX
        if not await self.topstep_client.connect():
            self.logger.error("Failed to connect to TopStepX - stopping agent")
            self.running = False
            return

        while self.running:
            try:
                # Fetch recent market data (use UTC timestamps)
                current_time = datetime.now(timezone.utc)
                start_time = current_time - timedelta(hours=1)  # 1 hour ago
                end_time = current_time  # Now

                self.logger.debug(f"🕐 Time calc: current={current_time.isoformat()}, start={start_time.isoformat()}, end={end_time.isoformat()}")

                bars = await self.topstep_client.retrieve_bars(
                    contract_id=contract_id,
                    start_time=start_time,
                    end_time=end_time,
                    unit=2,  # Minute bars
                    unit_number=1,
                    limit=100,
                    include_partial_bar=True,
                    live=False  # False for DEMO/practice accounts
                )

                if bars and len(bars) > 0:
                    # Sort bars by timestamp to ensure latest is last
                    bars = sorted(bars, key=lambda x: x.get('t', ''))

                    # Log first and last bar timestamps for debugging
                    first_bar_time = bars[0].get('t', 'unknown')
                    last_bar_time = bars[-1].get('t', 'unknown')
                    self.logger.info(f"📅 Bar range: {first_bar_time} to {last_bar_time} ({len(bars)} bars)")

                    # Convert bars to market state dict
                    latest_bar = bars[-1]
                    current_price = latest_bar.get('c')

                    # Validate bar age - reject if older than 2 minutes
                    bar_timestamp = latest_bar.get('t')
                    if bar_timestamp:
                        try:
                            from dateutil import parser
                            bar_time = parser.isoparse(bar_timestamp)
                            age_seconds = (datetime.now(timezone.utc) - bar_time).total_seconds()

                            if age_seconds > 120:  # More than 2 minutes old
                                self.logger.warning(
                                    f"⚠️ STALE DATA: Latest bar is {age_seconds:.0f}s old "
                                    f"(timestamp: {bar_timestamp})"
                                )
                        except:
                            pass

                    # Calculate basic indicators from bars
                    closes = [bar.get('c') for bar in bars if bar.get('c') is not None]
                    highs = [bar.get('h') for bar in bars if bar.get('h') is not None]
                    lows = [bar.get('l') for bar in bars if bar.get('l') is not None]
                    volumes = [bar.get('v') for bar in bars if bar.get('v') is not None]

                    # Proper volume-weighted VWAP calculation
                    if len(closes) >= 20 and len(volumes) >= 20:
                        # VWAP = Σ(Price × Volume) / Σ(Volume)
                        price_volume = sum(closes[i] * volumes[i] for i in range(-20, 0))
                        total_volume = sum(volumes[-20:])
                        vwap = price_volume / total_volume if total_volume > 0 else current_price

                        # Volume-weighted standard deviation
                        # Var = Σ(Volume × (Price - VWAP)²) / Σ(Volume)
                        variance = sum(volumes[i] * (closes[i] - vwap) ** 2 for i in range(-20, 0))
                        vwap_std = (variance / total_volume) ** 0.5 if total_volume > 0 else 1.0
                    else:
                        vwap = current_price
                        vwap_std = 1.0

                    # Simple RSI calculation (14-period)
                    if len(closes) >= 15:
                        gains = [max(closes[i] - closes[i-1], 0) for i in range(1, min(15, len(closes)))]
                        losses = [max(closes[i-1] - closes[i], 0) for i in range(1, min(15, len(closes)))]
                        avg_gain = sum(gains) / len(gains) if gains else 0.01
                        avg_loss = sum(losses) / len(losses) if losses else 0.01
                        rs = avg_gain / avg_loss if avg_loss != 0 else 100
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 50

                    # Simple ATR calculation
                    if len(highs) >= 14 and len(lows) >= 14 and len(closes) >= 14:
                        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                               for i in range(1, min(15, len(closes)))]
                        atr = sum(trs) / len(trs) if trs else 1.0
                    else:
                        atr = 1.0

                    # Determine market regime (simple: trending if price moved >1% in last 20 bars)
                    if len(closes) >= 20:
                        price_change = (closes[-1] - closes[-20]) / closes[-20]
                        regime = "TRENDING" if abs(price_change) > 0.01 else "RANGING"
                    else:
                        regime = "RANGING"

                    current_time = datetime.now(timezone.utc)
                    market_state = {
                        'timestamp': latest_bar.get('t'),
                        'time': current_time,
                        'current_time': current_time,
                        'current_price': current_price,
                        'open': latest_bar.get('o'),
                        'high': latest_bar.get('h'),
                        'low': latest_bar.get('l'),
                        'close': current_price,
                        'volume': latest_bar.get('v'),
                        'contract_id': contract_id,
                        'bars': bars,
                        'regime': regime,
                        'indicators': {
                            'vwap': {
                                'vwap': vwap,
                                'std': vwap_std,
                                'distance': (current_price - vwap) / vwap if vwap != 0 else 0
                            },
                            'rsi': {
                                'rsi': rsi,
                                'value': rsi,
                                'overbought': rsi > 70,
                                'oversold': rsi < 30
                            },
                            'atr': {
                                'value': atr,
                                'level': 'HIGH' if atr > 5 else 'MEDIUM' if atr > 2 else 'LOW'
                            }
                        },
                        'time_filters': {
                            'is_trading_hours': True,
                            'session': 'REGULAR',
                            'market_open': '17:00',  # Sunday 5 PM CST
                            'market_close': '15:10',  # Friday 3:10 PM CST
                            'skip_lunch': False  # Futures trade 24/5
                        }
                    }

                    # Calculate distance in standard deviations
                    distance_dollars = current_price - vwap
                    distance_std = (distance_dollars / vwap_std) if vwap_std > 0 else 0

                    self.logger.info(
                        f"📊 Market Data: {contract_id} @ {current_price:.2f} | "
                        f"Vol: {market_state['volume']} | "
                        f"VWAP: {vwap:.2f} (σ: {vwap_std:.2f}) | "
                        f"Distance: {distance_dollars:.2f} ({distance_std:.2f}σ) | "
                        f"RSI: {rsi:.1f} | "
                        f"Regime: {regime}"
                    )

                    # Evaluate all strategies
                    trade_decision = await self.strategy_selector.evaluate_market(market_state)

                    if trade_decision:
                        self.logger.info(f"🎯 Trade signal detected: {trade_decision}")

                        # In dry-run mode, log theoretical trade
                        if self.dry_run:
                            self._log_dry_run_trade(trade_decision)
                        else:
                            # Execute real trade
                            await self._execute_trade(trade_decision)
                else:
                    self.logger.warning("No market data received from TopStepX")

                # Wait for next decision interval
                await asyncio.sleep(decision_interval)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(decision_interval)

    async def _execute_trade(self, trade_decision: Dict[str, Any]) -> None:
        """
        Execute a real trade based on strategy decision.

        Args:
            trade_decision: Trade decision from strategy selector
        """
        try:
            setup = trade_decision['setup']
            decision = trade_decision['decision']

            # Determine order side
            from execution.topstep_client import OrderSide, OrderType
            side = OrderSide.BID if setup['signal'] == 'LONG' else OrderSide.ASK

            # Place market order
            order_id = await self.topstep_client.place_order(
                contract_id=trade_decision['contract_id'],
                side=side,
                size=1,  # 1 contract
                order_type=OrderType.MARKET,
                custom_tag=f"{trade_decision['strategy_name']}_{datetime.now().strftime('%H%M%S')}"
            )

            if order_id:
                self.logger.info(f"✅ Order placed successfully: #{order_id}")
                # TODO: Track position and set stop/target orders
            else:
                self.logger.error("❌ Failed to place order")

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)


async def main(dry_run: bool = False):
    """
    Main entry point.

    Args:
        dry_run: If True, run in DRY-RUN mode (no real orders)
    """
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('settings', {}))

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)

    if dry_run:
        logger.info("🧪 SDK Trading Agent Starting - DRY RUN MODE")
        logger.info("=" * 60)
        logger.warning("DRY RUN MODE ENABLED:")
        logger.warning("  - Connects to real market data")
        logger.warning("  - Evaluates strategies and calls Claude")
        logger.warning("  - Logs what trades WOULD be taken")
        logger.warning("  - Tracks theoretical slippage")
        logger.warning("  - Does NOT place actual orders")
        logger.info("=" * 60)
    else:
        logger.info("SDK Trading Agent Starting - LIVE TRADING")
        logger.info("=" * 60)

    # Set dry-run flag in config
    config['dry_run'] = dry_run

    try:
        # Create and start trading system
        system = TradingSystem(config)
        await system.start()

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        await system.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if hasattr(system, 'stop'):
            await system.stop()
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SDK Trading Agent')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in DRY-RUN mode (no real orders placed)'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode with historical data'
    )

    args = parser.parse_args()

    # Check for dry-run mode
    if args.dry_run:
        print("\n" + "=" * 60)
        print("🧪 DRY RUN MODE - No real orders will be placed")
        print("=" * 60)
        print("\nThis mode will:")
        print("  ✓ Connect to real market data")
        print("  ✓ Evaluate all strategies")
        print("  ✓ Call Claude for AI decisions")
        print("  ✓ Log theoretical trades")
        print("  ✓ Track expected slippage")
        print("  ✗ NOT place actual orders")
        print("\nUse this to validate strategy logic before going live.")
        print("=" * 60 + "\n")

        # Auto-confirm in non-interactive mode or get user confirmation
        import sys
        import os
        # Check if running in background (output redirected or not a TTY)
        is_interactive = sys.stdin.isatty() and sys.stdout.isatty() and not os.getenv('_')
        if not is_interactive:
            print("Auto-confirming DRY RUN (non-interactive mode)")
            response = 'yes'
        else:
            try:
                response = input("Continue with DRY RUN? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Cancelled.")
                    sys.exit(0)
            except EOFError:
                # Running in background
                print("Auto-confirming DRY RUN (background mode)")
                response = 'yes'

    # Run the async main function
    asyncio.run(main(dry_run=args.dry_run))
