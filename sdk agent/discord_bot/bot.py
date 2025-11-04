"""
Discord bot for monitoring and control.

This module implements the Discord bot client:
- Bot initialization and connection
- Event handling (on_ready, on_message, etc.)
- Command routing with authorization
- Reference to trading agent
- Keep-alive mechanisms
"""

import discord
from discord.ext import commands
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class TradingBot(commands.Bot):
    """
    Discord bot for SDK trading agent.

    Provides Discord interface for monitoring and controlling the agent
    with command authorization and rich embed support.
    """

    def __init__(self, config: Dict[str, Any], trading_agent: Any = None):
        """
        Initialize Discord bot.

        Args:
            config: Bot configuration from discord_config.yaml
            trading_agent: Reference to main trading agent instance
        """
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix='!',
            intents=intents,
            description='SDK Trading Agent Bot',
            help_command=None  # Custom help command
        )

        self.config = config.get('discord', {})
        self.trading_agent = trading_agent

        # Configuration
        self.trading_channel_id = self.config.get('channel_id')
        self.authorized_users = self.config.get('authorized_users', [])
        self.notification_settings = self.config.get('notifications', {})

        # State
        self.is_paused = False
        self.bot_started_at = None

        logger.info("TradingBot initialized")

    async def on_ready(self):
        """Called when bot successfully connects to Discord."""
        self.bot_started_at = discord.utils.utcnow()

        logger.info(f'Bot connected as {self.user.name} (ID: {self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')

        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="NQ futures 📈"
            )
        )

        # Send startup notification
        if self.trading_channel_id:
            channel = self.get_channel(self.trading_channel_id)
            if channel:
                embed = discord.Embed(
                    title="🤖 SDK Trading Agent Online",
                    description="Bot connected and ready for trading",
                    color=discord.Color.green()
                )
                embed.add_field(name="Status", value="✅ Active", inline=True)
                embed.add_field(name="Mode", value="Live Trading", inline=True)
                await channel.send(embed=embed)

    async def on_message(self, message: discord.Message):
        """
        Handle incoming messages.

        Args:
            message: Discord message object
        """
        # Ignore bot's own messages
        if message.author == self.user:
            return

        # Process commands
        await self.process_commands(message)

    async def on_command_error(self, ctx: commands.Context, error: Exception):
        """
        Handle command errors.

        Args:
            ctx: Command context
            error: Error that occurred
        """
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("❌ Command not found. Use `!help` to see available commands.")
        elif isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ You don't have permission to use this command.")
        elif isinstance(error, commands.CheckFailure):
            await ctx.send("❌ You are not authorized to use this command.")
        else:
            await ctx.send(f"❌ Error: {str(error)}")
            logger.error(f"Command error: {error}", exc_info=True)

    def is_authorized(self, user_id: int) -> bool:
        """
        Check if user is authorized to use bot.

        Args:
            user_id: Discord user ID

        Returns:
            bool: True if authorized
        """
        if not self.authorized_users:
            return True  # No restrictions if list is empty

        return user_id in self.authorized_users

    async def send_to_channel(
        self,
        content: Optional[str] = None,
        embed: Optional[discord.Embed] = None
    ) -> None:
        """
        Send message to trading channel.

        Args:
            content: Text message
            embed: Optional embed object
        """
        if not self.trading_channel_id:
            logger.warning("No trading channel ID configured")
            return

        try:
            channel = self.get_channel(self.trading_channel_id)
            if channel:
                await channel.send(content=content, embed=embed)
            else:
                logger.error(f"Channel {self.trading_channel_id} not found")
        except Exception as e:
            logger.error(f"Error sending to channel: {e}")

    def get_trading_channel(self) -> Optional[discord.TextChannel]:
        """
        Get trading channel object.

        Returns:
            TextChannel or None
        """
        if not self.trading_channel_id:
            return None

        return self.get_channel(self.trading_channel_id)


def create_embed(
    title: str,
    description: str = "",
    color: discord.Color = discord.Color.blue(),
    fields: Optional[Dict[str, str]] = None
) -> discord.Embed:
    """
    Create a Discord embed.

    Args:
        title: Embed title
        description: Embed description
        color: Embed color
        fields: Optional dictionary of field names and values

    Returns:
        discord.Embed: Formatted embed
    """
    embed = discord.Embed(
        title=title,
        description=description,
        color=color
    )

    if fields:
        for name, value in fields.items():
            # Auto-detect inline based on content length
            inline = len(str(value)) < 30
            embed.add_field(name=name, value=value, inline=inline)

    embed.timestamp = discord.utils.utcnow()

    return embed


async def run_bot(config: Dict[str, Any], trading_agent: Any = None):
    """
    Run Discord bot.

    Args:
        config: Discord configuration
        trading_agent: Trading agent reference
    """
    # Get token from config or environment
    token = config.get('discord', {}).get('token')

    if not token or token == "YOUR_DISCORD_BOT_TOKEN":
        # Try environment variable
        token = os.getenv('DISCORD_BOT_TOKEN')

    if not token:
        logger.error("No Discord bot token provided")
        return

    # Create and run bot
    bot = TradingBot(config, trading_agent)

    # Import and register commands
    try:
        from .commands import setup_commands
        setup_commands(bot)
        logger.info("Discord commands registered")
    except Exception as e:
        logger.error(f"Failed to register commands: {e}")

    # Run bot
    try:
        await bot.start(token)
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
