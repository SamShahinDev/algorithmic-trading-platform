"""
Discord notifications with rich embeds.

This module sends formatted notifications to Discord channel:
- Market analysis updates (throttled)
- Trade signals with latency info
- Order fills with slippage data
- Position exits with P&L
- Daily summaries with comprehensive stats

All notifications use rich embeds with emojis for visual clarity.
"""

import discord
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Manages Discord notifications for trading events.

    Handles throttling, formatting, and sending all trading notifications
    with comprehensive latency and slippage tracking.
    """

    def __init__(self, bot):
        """
        Initialize notification manager.

        Args:
            bot: Discord bot instance
        """
        self.bot = bot

        # Throttling
        self.last_analysis_time = None
        self.analysis_throttle_seconds = 300  # 5 minutes

        logger.info("NotificationManager initialized")

    async def send_analysis(
        self,
        market_state: Dict[str, Any],
        regime: str,
        force: bool = False
    ) -> None:
        """
        Send market analysis update (throttled).

        Args:
            market_state: Current market conditions
            regime: Market regime (TRENDING_UP, RANGING, etc.)
            force: Skip throttle check
        """
        # Check throttle
        now = datetime.now()
        if not force and self.last_analysis_time:
            elapsed = (now - self.last_analysis_time).total_seconds()
            if elapsed < self.analysis_throttle_seconds:
                logger.debug(f"Analysis throttled ({elapsed:.0f}s < {self.analysis_throttle_seconds}s)")
                return

        self.last_analysis_time = now

        try:
            current_price = market_state.get('current_price', 0)
            indicators = market_state.get('indicators', {})

            # Build embed
            embed = discord.Embed(
                title="📊 Market Analysis Update",
                description=f"Current Price: **{current_price:.2f}**",
                color=discord.Color.blue(),
                timestamp=discord.utils.utcnow()
            )

            # Regime
            regime_emoji = {
                'TRENDING_UP': '📈',
                'TRENDING_DOWN': '📉',
                'RANGING': '↔️',
                'UNKNOWN': '❓'
            }.get(regime, '❓')

            embed.add_field(
                name="Market Regime",
                value=f"{regime_emoji} {regime}",
                inline=True
            )

            # VWAP
            vwap_data = indicators.get('vwap', {})
            if vwap_data:
                vwap_text = (
                    f"Price: {vwap_data.get('vwap', 0):.2f}\n"
                    f"Distance: {vwap_data.get('distance_from_vwap', 0):.2f}\n"
                    f"Std Dev: {vwap_data.get('std_dev_distance', 0):.2f}σ"
                )
                embed.add_field(name="VWAP", value=vwap_text, inline=True)

            # RSI
            rsi_data = indicators.get('rsi', {})
            if rsi_data:
                rsi_value = rsi_data.get('rsi', 50)
                rsi_signal = rsi_data.get('signal', 'NEUTRAL')

                rsi_emoji = '🔴' if rsi_value > 70 else '🟢' if rsi_value < 30 else '⚪'
                rsi_text = f"{rsi_emoji} {rsi_value:.1f} ({rsi_signal})"
                embed.add_field(name="RSI", value=rsi_text, inline=True)

            # EMA
            ema_data = indicators.get('ema', {})
            if ema_data:
                ema20 = ema_data.get('ema20', 0)
                ema50 = ema_data.get('ema50', 0)
                alignment = ema_data.get('alignment', 'NEUTRAL')

                ema_text = f"EMA20: {ema20:.2f}\nEMA50: {ema50:.2f}\n{alignment}"
                embed.add_field(name="EMAs", value=ema_text, inline=True)

            # MACD
            macd_data = indicators.get('macd', {})
            if macd_data:
                histogram = macd_data.get('histogram', 0)
                signal_line = macd_data.get('signal', 'NEUTRAL')

                macd_emoji = '🟢' if histogram > 0 else '🔴'
                macd_text = f"{macd_emoji} Histogram: {histogram:.2f}\n{signal_line}"
                embed.add_field(name="MACD", value=macd_text, inline=True)

            # ATR
            atr_data = indicators.get('atr', {})
            if atr_data:
                atr_value = atr_data.get('atr', 0)
                volatility = atr_data.get('volatility_level', 'NORMAL')

                atr_text = f"{atr_value:.2f} ({volatility})"
                embed.add_field(name="ATR (Volatility)", value=atr_text, inline=True)

            embed.set_footer(text="Market analysis updates every 5 minutes")

            await self.bot.send_to_channel(embed=embed)
            logger.info("Market analysis sent to Discord")

        except Exception as e:
            logger.error(f"Error sending analysis: {e}", exc_info=True)

    async def send_signal(
        self,
        strategy_name: str,
        setup: Dict[str, Any],
        confidence: float,
        latency_ms: Optional[float] = None,
        claude_confidence: Optional[float] = None
    ) -> None:
        """
        Send trade signal notification with latency info.

        Args:
            strategy_name: Strategy that generated signal
            setup: Setup details
            confidence: Setup confidence score (0-10)
            latency_ms: Claude API latency in milliseconds
            claude_confidence: Claude's confidence (0-1)
        """
        try:
            signal = setup.get('signal', 'NONE')
            entry_price = setup.get('entry_price', 0)
            stop_price = setup.get('stop_price', 0)
            target_price = setup.get('target_price', 0)

            # Calculate R/R
            if signal == 'LONG':
                risk = entry_price - stop_price
                reward = target_price - entry_price
            else:
                risk = stop_price - entry_price
                reward = entry_price - target_price

            rr_ratio = reward / risk if risk > 0 else 0

            # Build embed
            embed = discord.Embed(
                title=f"🎯 Trade Signal: {signal}",
                description=f"**{strategy_name}** strategy detected high-confidence setup",
                color=discord.Color.green() if signal == 'LONG' else discord.Color.red(),
                timestamp=discord.utils.utcnow()
            )

            # Setup info
            setup_text = (
                f"**Confidence:** {confidence:.1f}/10\n"
                f"**Entry:** {entry_price:.2f}\n"
                f"**Stop:** {stop_price:.2f}\n"
                f"**Target:** {target_price:.2f}\n"
                f"**R/R:** {rr_ratio:.2f}"
            )
            embed.add_field(name="Setup Details", value=setup_text, inline=False)

            # Claude decision
            if latency_ms is not None and claude_confidence is not None:
                latency_emoji = '🟢' if latency_ms < 500 else '🟡' if latency_ms < 1000 else '🔴'
                confidence_emoji = '🟢' if claude_confidence > 0.8 else '🟡' if claude_confidence > 0.6 else '🔴'

                claude_text = (
                    f"{latency_emoji} **Latency:** {latency_ms:.0f}ms\n"
                    f"{confidence_emoji} **Claude Confidence:** {claude_confidence:.2f}\n"
                    f"**Decision:** ENTER"
                )
                embed.add_field(name="🤖 SDK Agent", value=claude_text, inline=False)

            # Reasoning
            reasoning = setup.get('reasoning', {})
            if reasoning:
                conditions_met = reasoning.get('conditions_met', [])
                if conditions_met:
                    conditions_text = '\n'.join(f"✅ {c}" for c in conditions_met[:5])
                    embed.add_field(name="Conditions Met", value=conditions_text, inline=False)

            embed.set_footer(text=f"Strategy: {strategy_name}")

            await self.bot.send_to_channel(embed=embed)
            logger.info(f"Signal notification sent: {strategy_name} {signal}")

        except Exception as e:
            logger.error(f"Error sending signal: {e}", exc_info=True)

    async def send_validation(
        self,
        passed: bool,
        slippage_ticks: float,
        original_entry: float,
        current_entry: float,
        reason: Optional[str] = None
    ) -> None:
        """
        Send post-validation result notification.

        Args:
            passed: Whether validation passed
            slippage_ticks: Slippage in ticks
            original_entry: Original entry price
            current_entry: Current entry price
            reason: Reason if failed
        """
        try:
            if passed:
                slippage_emoji = '🟢' if slippage_ticks < 1 else '🟡' if slippage_ticks < 2 else '🟠'

                embed = discord.Embed(
                    title="✅ Validation: PASSED",
                    description="Setup still valid after Claude decision",
                    color=discord.Color.green(),
                    timestamp=discord.utils.utcnow()
                )

                validation_text = (
                    f"{slippage_emoji} **Slippage:** {slippage_ticks:.2f} ticks\n"
                    f"**Original Entry:** {original_entry:.2f}\n"
                    f"**Current Entry:** {current_entry:.2f}\n"
                    f"**Status:** Proceeding to execution"
                )
                embed.add_field(name="Validation Details", value=validation_text, inline=False)
            else:
                embed = discord.Embed(
                    title="❌ Validation: FAILED",
                    description=reason or "Setup degraded during latency",
                    color=discord.Color.orange(),
                    timestamp=discord.utils.utcnow()
                )

                validation_text = (
                    f"**Slippage:** {slippage_ticks:.2f} ticks\n"
                    f"**Original Entry:** {original_entry:.2f}\n"
                    f"**Current Entry:** {current_entry:.2f}\n"
                    f"**Status:** Trade cancelled"
                )
                embed.add_field(name="Validation Details", value=validation_text, inline=False)

            await self.bot.send_to_channel(embed=embed)
            logger.info(f"Validation notification sent: {'PASSED' if passed else 'FAILED'}")

        except Exception as e:
            logger.error(f"Error sending validation: {e}", exc_info=True)

    async def send_fill(
        self,
        strategy_name: str,
        side: str,
        fill_price: float,
        intended_price: float,
        slippage_ticks: float,
        order_id: str,
        stop_price: float,
        target_price: float
    ) -> None:
        """
        Send order fill notification with slippage data.

        Args:
            strategy_name: Strategy name
            side: LONG or SHORT
            fill_price: Actual fill price
            intended_price: Intended entry price
            slippage_ticks: Slippage in ticks
            order_id: Order ID
            stop_price: Stop loss price
            target_price: Target price
        """
        try:
            # Slippage color coding
            slippage_emoji = '🟢' if slippage_ticks < 1 else '🟡' if slippage_ticks < 2 else '🟠' if slippage_ticks < 3 else '🔴'

            embed = discord.Embed(
                title=f"✅ Order Filled: {side}",
                description=f"**{strategy_name}** position opened",
                color=discord.Color.green() if side == 'LONG' else discord.Color.red(),
                timestamp=discord.utils.utcnow()
            )

            # Fill details
            fill_text = (
                f"**Fill Price:** {fill_price:.2f}\n"
                f"**Intended Price:** {intended_price:.2f}\n"
                f"{slippage_emoji} **Slippage:** {slippage_ticks:.2f} ticks (${slippage_ticks * 5:.2f})\n"
                f"**Order ID:** {order_id}"
            )
            embed.add_field(name="Fill Details", value=fill_text, inline=False)

            # Bracket orders
            bracket_text = (
                f"**Stop Loss:** {stop_price:.2f}\n"
                f"**Take Profit:** {target_price:.2f}"
            )
            embed.add_field(name="Exit Orders", value=bracket_text, inline=False)

            # Calculate potential
            if side == 'LONG':
                risk = fill_price - stop_price
                reward = target_price - fill_price
            else:
                risk = stop_price - fill_price
                reward = fill_price - target_price

            risk_dollars = risk * 20  # NQ = $20/point
            reward_dollars = reward * 20

            potential_text = (
                f"**Risk:** ${abs(risk_dollars):.2f} ({abs(risk):.2f} pts)\n"
                f"**Reward:** ${reward_dollars:.2f} ({reward:.2f} pts)\n"
                f"**R/R:** {reward/risk if risk != 0 else 0:.2f}"
            )
            embed.add_field(name="Risk/Reward", value=potential_text, inline=False)

            embed.set_footer(text=f"Strategy: {strategy_name}")

            await self.bot.send_to_channel(embed=embed)
            logger.info(f"Fill notification sent: {side} @ {fill_price:.2f}, slippage {slippage_ticks:.2f} ticks")

        except Exception as e:
            logger.error(f"Error sending fill: {e}", exc_info=True)

    async def send_exit(
        self,
        strategy_name: str,
        side: str,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_ticks: float,
        daily_pnl: float,
        trade_number: int
    ) -> None:
        """
        Send position exit notification with P&L.

        Args:
            strategy_name: Strategy name
            side: LONG or SHORT
            entry_price: Entry fill price
            exit_price: Exit fill price
            exit_reason: Why position closed (TARGET, STOP, MANUAL)
            pnl: Realized P&L in dollars
            pnl_ticks: P&L in ticks
            daily_pnl: Updated daily P&L
            trade_number: Trade number for the day
        """
        try:
            # Color based on P&L
            if pnl > 0:
                color = discord.Color.green()
                pnl_emoji = '🟢'
                result = 'WIN'
            elif pnl < 0:
                color = discord.Color.red()
                pnl_emoji = '🔴'
                result = 'LOSS'
            else:
                color = discord.Color.greyple()
                pnl_emoji = '⚪'
                result = 'BREAKEVEN'

            # Exit reason emoji
            exit_emoji = {
                'TARGET': '🎯',
                'STOP': '🛑',
                'MANUAL': '✋',
                'EOD': '🕐'
            }.get(exit_reason, '❓')

            embed = discord.Embed(
                title=f"{pnl_emoji} Position Closed: {result}",
                description=f"**{strategy_name}** - {exit_emoji} {exit_reason}",
                color=color,
                timestamp=discord.utils.utcnow()
            )

            # Trade details
            trade_text = (
                f"**Side:** {side}\n"
                f"**Entry:** {entry_price:.2f}\n"
                f"**Exit:** {exit_price:.2f}\n"
                f"**Trade #{trade_number}**"
            )
            embed.add_field(name="Trade Details", value=trade_text, inline=True)

            # P&L
            pnl_text = (
                f"{pnl_emoji} **${pnl:.2f}**\n"
                f"**{pnl_ticks:+.2f} ticks**\n"
                f"**Daily P&L:** ${daily_pnl:.2f}"
            )
            embed.add_field(name="Profit/Loss", value=pnl_text, inline=True)

            embed.set_footer(text=f"Strategy: {strategy_name}")

            await self.bot.send_to_channel(embed=embed)
            logger.info(f"Exit notification sent: {result} ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Error sending exit: {e}", exc_info=True)

    async def send_daily_summary(
        self,
        summary: Dict[str, Any],
        sdk_stats: Dict[str, Any],
        slippage_stats: Dict[str, Any],
        strategy_stats: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Send end-of-day summary with comprehensive stats.

        Args:
            summary: Daily summary from risk manager
            sdk_stats: SDK agent statistics
            slippage_stats: Slippage statistics
            strategy_stats: Per-strategy performance
        """
        try:
            daily_pnl = summary.get('daily_pnl', 0)

            # Color based on daily result
            if daily_pnl >= summary.get('target_profit', 250):
                color = discord.Color.gold()
                title_emoji = '🏆'
                status = 'TARGET HIT'
            elif daily_pnl > 0:
                color = discord.Color.green()
                title_emoji = '🟢'
                status = 'PROFITABLE'
            elif daily_pnl < 0:
                color = discord.Color.red()
                title_emoji = '🔴'
                status = 'LOSS'
            else:
                color = discord.Color.greyple()
                title_emoji = '⚪'
                status = 'BREAKEVEN'

            embed = discord.Embed(
                title=f"{title_emoji} Daily Summary: {status}",
                description=f"**{summary.get('date')}**",
                color=color,
                timestamp=discord.utils.utcnow()
            )

            # Performance
            pnl_emoji = '🟢' if daily_pnl >= 0 else '🔴'
            perf_text = (
                f"{pnl_emoji} **Daily P&L:** ${daily_pnl:.2f}\n"
                f"**Target:** ${summary.get('target_profit', 250)}\n"
                f"**Max Loss:** -${abs(summary.get('max_loss', 150))}\n"
                f"**Win Rate:** {summary.get('win_rate', 0):.1f}%\n"
                f"**Trades:** {summary.get('total_trades', 0)}/{summary.get('max_trades', 8)}"
            )
            embed.add_field(name="📊 Performance", value=perf_text, inline=False)

            # Strategy breakdown
            if strategy_stats:
                strat_text = ""
                for strat_name, stats in strategy_stats.items():
                    if stats.get('total_trades', 0) > 0:
                        strat_pnl = stats.get('total_pnl', 0)
                        strat_emoji = '🟢' if strat_pnl > 0 else '🔴'
                        strat_text += (
                            f"{strat_emoji} **{strat_name}:** "
                            f"{stats['total_trades']} trades, "
                            f"${strat_pnl:.2f}, "
                            f"{stats.get('win_rate', 0):.1f}% WR\n"
                        )

                if strat_text:
                    embed.add_field(name="🎯 Strategy Breakdown", value=strat_text, inline=False)

            # SDK Agent stats
            sdk_text = (
                f"**Evaluations:** {sdk_stats.get('total_evaluations', 0)}\n"
                f"**Pre-Filtered:** {sdk_stats.get('pre_filtered', 0)} "
                f"({sdk_stats.get('pre_filter_rate', 0)*100:.1f}%)\n"
                f"**Claude Calls:** {sdk_stats.get('claude_calls', 0)}\n"
                f"**Avg Latency:** {sdk_stats.get('avg_latency_ms', 0):.0f}ms\n"
                f"**Validation Rate:** {sdk_stats.get('validation_success_rate', 0)*100:.1f}%"
            )
            embed.add_field(name="🤖 SDK Agent", value=sdk_text, inline=False)

            # Slippage stats
            slip_text = (
                f"**Avg Slippage:** {slippage_stats.get('avg_slippage_ticks', 0):.2f} ticks\n"
                f"**Max Slippage:** {slippage_stats.get('max_slippage_ticks', 0):.2f} ticks\n"
                f"**Successful Entries:** {slippage_stats.get('successful_entries', 0)}\n"
                f"**Validation Failures:** {slippage_stats.get('validation_failures', 0)}"
            )
            embed.add_field(name="⚡ Latency & Slippage", value=slip_text, inline=False)

            embed.set_footer(text="End of Day Report")

            await self.bot.send_to_channel(embed=embed)
            logger.info(f"Daily summary sent: ${daily_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}", exc_info=True)

    async def send_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send error notification.

        Args:
            error_type: Type of error
            message: Error message
            details: Optional error details
        """
        try:
            embed = discord.Embed(
                title=f"⚠️ Error: {error_type}",
                description=message,
                color=discord.Color.orange(),
                timestamp=discord.utils.utcnow()
            )

            if details:
                details_text = '\n'.join(f"**{k}:** {v}" for k, v in details.items())
                embed.add_field(name="Details", value=details_text, inline=False)

            await self.bot.send_to_channel(embed=embed)
            logger.warning(f"Error notification sent: {error_type}")

        except Exception as e:
            logger.error(f"Error sending error notification: {e}", exc_info=True)

    async def send_risk_limit(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        reason: str
    ) -> None:
        """
        Send risk limit notification.

        Args:
            limit_type: Type of limit (PROFIT_TARGET, MAX_LOSS, TRADE_LIMIT, etc.)
            current_value: Current value
            limit_value: Limit value
            reason: Reason message
        """
        try:
            if limit_type == 'PROFIT_TARGET':
                color = discord.Color.gold()
                emoji = '🎯'
            elif limit_type == 'MAX_LOSS':
                color = discord.Color.red()
                emoji = '🛑'
            else:
                color = discord.Color.orange()
                emoji = '⚠️'

            embed = discord.Embed(
                title=f"{emoji} Risk Limit: {limit_type}",
                description=reason,
                color=color,
                timestamp=discord.utils.utcnow()
            )

            limit_text = (
                f"**Current:** {current_value}\n"
                f"**Limit:** {limit_value}\n"
                f"**Status:** Trading stopped"
            )
            embed.add_field(name="Limit Details", value=limit_text, inline=False)

            await self.bot.send_to_channel(embed=embed)
            logger.info(f"Risk limit notification sent: {limit_type}")

        except Exception as e:
            logger.error(f"Error sending risk limit: {e}", exc_info=True)
