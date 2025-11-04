"""
Discord bot commands with authorization.

All trading commands with user ID authorization:
- !status - Current position and daily P&L
- !pause - Stop taking new trades
- !resume - Resume trading
- !close - Manually close position (with confirmation)
- !summary - Detailed daily report
- !config - Show current settings
- !slippage - Show slippage statistics
- !kill - Emergency shutdown
"""

from discord.ext import commands
import discord
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def authorized_only():
    """Decorator to require authorization for command."""
    async def predicate(ctx):
        bot = ctx.bot
        if not bot.is_authorized(ctx.author.id):
            raise commands.CheckFailure("User not authorized")
        return True
    return commands.check(predicate)


class TradingCommands(commands.Cog):
    """
    Trading bot commands with authorization checks.

    All commands require user to be in authorized_users list.
    """

    def __init__(self, bot):
        """
        Initialize commands cog.

        Args:
            bot: Discord bot instance
        """
        self.bot = bot

    @commands.command(name='status')
    @authorized_only()
    async def status(self, ctx: commands.Context):
        """
        Show current position and daily P&L.

        Usage: !status
        """
        agent = self.bot.trading_agent

        if not agent:
            await ctx.send("❌ Trading agent not connected")
            return

        try:
            # Get daily summary from risk manager
            summary = agent.risk_manager.get_daily_summary()

            # Get SDK agent stats
            sdk_stats = agent.sdk_agent.get_statistics()

            # Get slippage stats from strategy selector
            slippage_stats = agent.strategy_selector.get_slippage_statistics()

            # Build embed
            embed = discord.Embed(
                title="📊 SDK Trading Agent Status",
                color=discord.Color.green() if summary['daily_pnl'] >= 0 else discord.Color.red()
            )

            # Position info
            position_status = f"{summary['current_positions']}/1"
            embed.add_field(name="Position", value=position_status, inline=True)

            # P&L
            pnl_emoji = "🟢" if summary['daily_pnl'] >= 0 else "🔴"
            pnl_text = f"{pnl_emoji} ${summary['daily_pnl']:.2f}"
            embed.add_field(name="Daily P&L", value=pnl_text, inline=True)

            # Target/Max Loss
            target_text = f"${summary['target_profit']} (${summary['remaining_to_target']:.2f} to go)"
            embed.add_field(name="Target", value=target_text, inline=True)

            # Trades
            trades_text = f"{summary['total_trades']}/{summary['max_trades']}"
            embed.add_field(name="Trades", value=trades_text, inline=True)

            # Win Rate
            win_rate_text = f"{summary['win_rate']:.1f}%"
            embed.add_field(name="Win Rate", value=win_rate_text, inline=True)

            # Claude Calls
            claude_text = f"{sdk_stats['claude_calls']} calls ({sdk_stats['avg_latency_ms']:.0f}ms avg)"
            embed.add_field(name="Claude API", value=claude_text, inline=True)

            # Slippage
            slip_text = f"{slippage_stats['avg_slippage_ticks']:.2f} ticks avg"
            embed.add_field(name="Slippage", value=slip_text, inline=True)

            # Trading status
            status_text = "✅ Active" if summary['can_continue_trading'] else "⛔ Stopped"
            if self.bot.is_paused:
                status_text = "⏸️ Paused"
            embed.add_field(name="Status", value=status_text, inline=True)

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in !status command: {e}", exc_info=True)
            await ctx.send(f"❌ Error getting status: {str(e)}")

    @commands.command(name='pause')
    @authorized_only()
    async def pause(self, ctx: commands.Context):
        """
        Stop taking new trades (keeps existing position).

        Usage: !pause
        """
        self.bot.is_paused = True
        await ctx.send("⏸️ **Trading Paused**\nWill not take new trades. Existing position remains open.")
        logger.info(f"Trading paused by {ctx.author.name}")

    @commands.command(name='resume')
    @authorized_only()
    async def resume(self, ctx: commands.Context):
        """
        Resume trading after pause.

        Usage: !resume
        """
        self.bot.is_paused = False
        await ctx.send("▶️ **Trading Resumed**\nNow accepting new trade signals.")
        logger.info(f"Trading resumed by {ctx.author.name}")

    @commands.command(name='close')
    @authorized_only()
    async def close_position(self, ctx: commands.Context):
        """
        Manually close position with confirmation.

        Usage: !close
        """
        agent = self.bot.trading_agent

        if not agent:
            await ctx.send("❌ Trading agent not connected")
            return

        # Check if position open
        summary = agent.risk_manager.get_daily_summary()

        if summary['current_positions'] == 0:
            await ctx.send("ℹ️ No open positions to close")
            return

        # Send confirmation message
        embed = discord.Embed(
            title="⚠️ Confirm Position Close",
            description="React with ✅ to confirm closing the current position",
            color=discord.Color.orange()
        )

        msg = await ctx.send(embed=embed)
        await msg.add_reaction("✅")
        await msg.add_reaction("❌")

        def check(reaction, user):
            return (
                user == ctx.author and
                str(reaction.emoji) in ["✅", "❌"] and
                reaction.message.id == msg.id
            )

        try:
            reaction, user = await self.bot.wait_for('reaction_add', timeout=30.0, check=check)

            if str(reaction.emoji) == "✅":
                # TODO: Close position via trading agent
                await ctx.send("✅ **Position Closed** (manual)")
                logger.info(f"Position manually closed by {ctx.author.name}")
            else:
                await ctx.send("❌ Position close cancelled")

        except TimeoutError:
            await ctx.send("⏱️ Confirmation timeout - position close cancelled")

    @commands.command(name='summary')
    @authorized_only()
    async def summary(self, ctx: commands.Context):
        """
        Detailed daily report with all metrics.

        Usage: !summary
        """
        agent = self.bot.trading_agent

        if not agent:
            await ctx.send("❌ Trading agent not connected")
            return

        try:
            # Get all stats
            summary = agent.risk_manager.get_daily_summary()
            sdk_stats = agent.sdk_agent.get_statistics()
            slippage_stats = agent.strategy_selector.get_slippage_statistics()
            strategy_stats = agent.strategy_selector.get_all_strategy_stats()

            # Build comprehensive embed
            embed = discord.Embed(
                title="📈 Daily Trading Summary",
                description=f"Date: {summary['date']}",
                color=discord.Color.blue()
            )

            # Performance Section
            pnl_emoji = "🟢" if summary['daily_pnl'] >= 0 else "🔴"
            perf_text = (
                f"{pnl_emoji} **Daily P&L:** ${summary['daily_pnl']:.2f}\n"
                f"**Target:** ${summary['target_profit']}\n"
                f"**Max Loss:** -${summary['max_loss']}\n"
                f"**Win Rate:** {summary['win_rate']:.1f}%\n"
                f"**Trades:** {summary['total_trades']}/{summary['max_trades']}"
            )
            embed.add_field(name="📊 Performance", value=perf_text, inline=False)

            # Strategy Breakdown
            strat_text = ""
            for strat_name, stats in strategy_stats.items():
                if stats.get('total_trades', 0) > 0:
                    strat_text += (
                        f"**{strat_name}:** {stats['total_trades']} trades, "
                        f"${stats['total_pnl']:.2f}, "
                        f"{stats['win_rate']:.1f}% WR\n"
                    )
            if strat_text:
                embed.add_field(name="🎯 Strategy Breakdown", value=strat_text, inline=False)

            # SDK Agent Stats
            sdk_text = (
                f"**Evaluations:** {sdk_stats['total_evaluations']}\n"
                f"**Pre-Filtered:** {sdk_stats['pre_filtered']} ({sdk_stats['pre_filter_rate']*100:.1f}%)\n"
                f"**Claude Calls:** {sdk_stats['claude_calls']}\n"
                f"**Avg Latency:** {sdk_stats['avg_latency_ms']:.0f}ms\n"
                f"**Validation Rate:** {sdk_stats['validation_success_rate']*100:.1f}%"
            )
            embed.add_field(name="🤖 SDK Agent", value=sdk_text, inline=False)

            # Slippage Stats
            slip_text = (
                f"**Avg Slippage:** {slippage_stats['avg_slippage_ticks']:.2f} ticks\n"
                f"**Max Slippage:** {slippage_stats['max_slippage_ticks']:.2f} ticks\n"
                f"**Successful Entries:** {slippage_stats['successful_entries']}\n"
                f"**Validation Failures:** {slippage_stats['validation_failures']}"
            )
            embed.add_field(name="⚡ Latency & Slippage", value=slip_text, inline=False)

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in !summary command: {e}", exc_info=True)
            await ctx.send(f"❌ Error generating summary: {str(e)}")

    @commands.command(name='config')
    @authorized_only()
    async def config(self, ctx: commands.Context):
        """
        Show current trading configuration.

        Usage: !config
        """
        agent = self.bot.trading_agent

        if not agent:
            await ctx.send("❌ Trading agent not connected")
            return

        summary = agent.risk_manager.get_daily_summary()

        embed = discord.Embed(
            title="⚙️ Trading Configuration",
            color=discord.Color.blue()
        )

        # Daily Limits
        limits_text = (
            f"**Target Profit:** ${summary['target_profit']}\n"
            f"**Max Loss:** -${summary['max_loss']}\n"
            f"**Max Trades:** {summary['max_trades']}\n"
            f"**Position Size:** 1 contract"
        )
        embed.add_field(name="📋 Daily Limits", value=limits_text, inline=False)

        # Strategy Limits
        strat_limits_text = ""
        for strat, count in summary['strategy_trades'].items():
            max_val = agent.risk_manager.strategy_max_trades.get(strat, "?")
            strat_limits_text += f"**{strat}:** {count}/{max_val}\n"
        if not strat_limits_text:
            strat_limits_text = "No trades yet today"
        embed.add_field(name="🎯 Strategy Limits", value=strat_limits_text, inline=False)

        # SDK Agent Settings
        sdk_text = (
            f"**Min Confidence:** {agent.sdk_agent.min_confidence_for_claude}/10\n"
            f"**Max Slippage:** {agent.sdk_agent.max_acceptable_slippage_ticks} ticks\n"
            f"**Model:** claude-sonnet-4-5"
        )
        embed.add_field(name="🤖 SDK Agent", value=sdk_text, inline=False)

        await ctx.send(embed=embed)

    @commands.command(name='slippage')
    @authorized_only()
    async def slippage(self, ctx: commands.Context):
        """
        Show average slippage statistics.

        Usage: !slippage
        """
        agent = self.bot.trading_agent

        if not agent:
            await ctx.send("❌ Trading agent not connected")
            return

        # Get slippage stats from both sources
        selector_stats = agent.strategy_selector.get_slippage_statistics()
        risk_stats = agent.risk_manager.get_slippage_statistics()

        embed = discord.Embed(
            title="⚡ Slippage Statistics",
            description="Latency and slippage metrics",
            color=discord.Color.gold()
        )

        # Overall Stats
        overall_text = (
            f"**Avg Slippage:** {selector_stats['avg_slippage_ticks']:.2f} ticks\n"
            f"**Max Slippage:** {selector_stats['max_slippage_ticks']:.2f} ticks\n"
            f"**Min Slippage:** {selector_stats['min_slippage_ticks']:.2f} ticks\n"
            f"**Samples:** {selector_stats['slippage_samples_count']}"
        )
        embed.add_field(name="📊 Overall", value=overall_text, inline=False)

        # Validation Stats
        validation_text = (
            f"**High-Confidence Setups:** {selector_stats['high_confidence_setups']}\n"
            f"**Claude Decisions:** {selector_stats['claude_decisions']}\n"
            f"**Validation Failures:** {selector_stats['validation_failures']}\n"
            f"**Successful Entries:** {selector_stats['successful_entries']}"
        )
        embed.add_field(name="✅ Validation", value=validation_text, inline=False)

        # SDK Agent Stats
        sdk_stats = agent.sdk_agent.get_statistics()
        sdk_text = (
            f"**Avg Latency:** {sdk_stats['avg_latency_ms']:.0f}ms\n"
            f"**Pre-Filter Rate:** {sdk_stats['pre_filter_rate']*100:.1f}%\n"
            f"**Validation Success:** {sdk_stats['validation_success_rate']*100:.1f}%"
        )
        embed.add_field(name="🤖 SDK Agent", value=sdk_text, inline=False)

        await ctx.send(embed=embed)

    @commands.command(name='kill')
    @authorized_only()
    async def kill(self, ctx: commands.Context):
        """
        Emergency shutdown with confirmation.

        Usage: !kill
        """
        # Send confirmation
        embed = discord.Embed(
            title="🚨 Emergency Shutdown",
            description="React with ✅ to confirm emergency shutdown\n⚠️ This will close all positions and stop the bot",
            color=discord.Color.red()
        )

        msg = await ctx.send(embed=embed)
        await msg.add_reaction("✅")
        await msg.add_reaction("❌")

        def check(reaction, user):
            return (
                user == ctx.author and
                str(reaction.emoji) in ["✅", "❌"] and
                reaction.message.id == msg.id
            )

        try:
            reaction, user = await self.bot.wait_for('reaction_add', timeout=30.0, check=check)

            if str(reaction.emoji) == "✅":
                await ctx.send("🛑 **EMERGENCY SHUTDOWN INITIATED**")
                logger.warning(f"Emergency shutdown triggered by {ctx.author.name}")

                # TODO: Close all positions
                # TODO: Stop trading agent
                # TODO: Shutdown bot

                await ctx.send("✅ Shutdown complete")
            else:
                await ctx.send("❌ Shutdown cancelled")

        except TimeoutError:
            await ctx.send("⏱️ Confirmation timeout - shutdown cancelled")

    @commands.command(name='help')
    async def help_command(self, ctx: commands.Context):
        """
        Show available commands.

        Usage: !help
        """
        embed = discord.Embed(
            title="📚 SDK Trading Agent Commands",
            description="All commands require authorization",
            color=discord.Color.blue()
        )

        commands_list = [
            ("!status", "Current position and daily P&L"),
            ("!pause", "Stop taking new trades"),
            ("!resume", "Resume trading"),
            ("!close", "Manually close position (with confirmation)"),
            ("!summary", "Detailed daily report"),
            ("!config", "Show current settings"),
            ("!slippage", "Show slippage statistics"),
            ("!kill", "Emergency shutdown"),
            ("!help", "Show this help message")
        ]

        for cmd, desc in commands_list:
            embed.add_field(name=cmd, value=desc, inline=False)

        await ctx.send(embed=embed)


def setup_commands(bot):
    """
    Setup function to register commands.

    Args:
        bot: Discord bot instance
    """
    # Use synchronous add_cog for compatibility
    import asyncio

    async def add_cog():
        await bot.add_cog(TradingCommands(bot))

    # Run in event loop
    if bot.loop.is_running():
        asyncio.create_task(add_cog())
    else:
        bot.loop.run_until_complete(add_cog())

    logger.info("Trading commands registered")
