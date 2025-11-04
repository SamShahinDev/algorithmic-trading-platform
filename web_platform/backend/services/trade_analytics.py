"""
Trade Analytics Service
Analyzes historical trade data for insights and performance metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class TradeAnalyticsService:
    """Service for analyzing trade performance and generating insights"""
    
    def __init__(self, topstepx_client):
        """Initialize the trade analytics service
        
        Args:
            topstepx_client: TopStepX API client instance
        """
        self.topstepx = topstepx_client
        self.cached_analytics = {}
        self.cache_expiry = 300  # 5 minutes
    
    async def get_performance_summary(self, days: int = 7) -> Dict:
        """Get comprehensive performance summary
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with detailed performance metrics
        """
        try:
            # Get trade statistics
            stats = await self.topstepx.get_trade_statistics(days)
            
            # Get time-based analysis
            time_analysis = await self.analyze_by_time_of_day(days)
            
            # Get pattern performance
            pattern_analysis = await self.analyze_by_pattern(days)
            
            # Get streak analysis
            streak_info = await self.analyze_streaks(days)
            
            # Get risk metrics
            risk_metrics = await self.calculate_risk_metrics(days)
            
            return {
                "basic_stats": stats,
                "time_analysis": time_analysis,
                "pattern_performance": pattern_analysis,
                "streak_analysis": streak_info,
                "risk_metrics": risk_metrics,
                "period_days": days,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance summary error: {e}")
            return {}
    
    async def analyze_by_time_of_day(self, days: int) -> Dict:
        """Analyze performance by time of day
        
        Returns:
            Dict with hourly performance breakdown
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            result = await self.topstepx.search_trades(start_date)
            trades = result.get("trades", [])
            
            hourly_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})
            
            for trade in trades:
                pnl = trade.get("profitAndLoss")
                if pnl is not None:
                    timestamp = datetime.fromisoformat(trade["creationTimestamp"].replace("+00:00", ""))
                    hour = timestamp.hour
                    
                    if float(pnl) > 0:
                        hourly_stats[hour]["wins"] += 1
                    else:
                        hourly_stats[hour]["losses"] += 1
                    
                    hourly_stats[hour]["total_pnl"] += float(pnl)
            
            # Calculate win rates and format results
            best_hours = []
            worst_hours = []
            
            for hour, stats in hourly_stats.items():
                total_trades = stats["wins"] + stats["losses"]
                if total_trades > 0:
                    win_rate = (stats["wins"] / total_trades) * 100
                    avg_pnl = stats["total_pnl"] / total_trades
                    
                    hour_data = {
                        "hour": hour,
                        "win_rate": round(win_rate, 2),
                        "total_trades": total_trades,
                        "avg_pnl": round(avg_pnl, 2)
                    }
                    
                    if win_rate >= 60:
                        best_hours.append(hour_data)
                    elif win_rate <= 40:
                        worst_hours.append(hour_data)
            
            # Sort by win rate
            best_hours.sort(key=lambda x: x["win_rate"], reverse=True)
            worst_hours.sort(key=lambda x: x["win_rate"])
            
            return {
                "best_hours": best_hours[:3],
                "worst_hours": worst_hours[:3],
                "all_hours": dict(hourly_stats)
            }
            
        except Exception as e:
            logger.error(f"Time analysis error: {e}")
            return {}
    
    async def analyze_by_pattern(self, days: int) -> Dict:
        """Analyze performance by trading pattern
        
        Returns:
            Dict with pattern-specific performance
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            result = await self.topstepx.search_trades(start_date)
            trades = result.get("trades", [])
            
            pattern_stats = defaultdict(lambda: {
                "wins": 0, 
                "losses": 0, 
                "total_pnl": 0,
                "trades": []
            })
            
            for trade in trades:
                # Extract pattern from custom tag if available
                custom_tag = trade.get("customTag", "unknown")
                pattern_name = self._extract_pattern_name(custom_tag)
                
                pnl = trade.get("profitAndLoss")
                if pnl is not None:
                    pnl_value = float(pnl)
                    
                    if pnl_value > 0:
                        pattern_stats[pattern_name]["wins"] += 1
                    else:
                        pattern_stats[pattern_name]["losses"] += 1
                    
                    pattern_stats[pattern_name]["total_pnl"] += pnl_value
                    pattern_stats[pattern_name]["trades"].append({
                        "timestamp": trade["creationTimestamp"],
                        "pnl": pnl_value
                    })
            
            # Calculate metrics for each pattern
            pattern_performance = []
            
            for pattern, stats in pattern_stats.items():
                total_trades = stats["wins"] + stats["losses"]
                if total_trades > 0:
                    win_rate = (stats["wins"] / total_trades) * 100
                    avg_pnl = stats["total_pnl"] / total_trades
                    
                    # Calculate consistency (standard deviation of P&L)
                    pnls = [t["pnl"] for t in stats["trades"]]
                    consistency = np.std(pnls) if len(pnls) > 1 else 0
                    
                    pattern_performance.append({
                        "pattern": pattern,
                        "total_trades": total_trades,
                        "win_rate": round(win_rate, 2),
                        "total_pnl": round(stats["total_pnl"], 2),
                        "avg_pnl": round(avg_pnl, 2),
                        "consistency": round(consistency, 2),
                        "profit_factor": self._calculate_profit_factor(stats)
                    })
            
            # Sort by total P&L
            pattern_performance.sort(key=lambda x: x["total_pnl"], reverse=True)
            
            return {
                "top_patterns": pattern_performance[:5],
                "worst_patterns": pattern_performance[-3:] if len(pattern_performance) > 3 else [],
                "total_patterns": len(pattern_performance)
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return {}
    
    async def analyze_streaks(self, days: int) -> Dict:
        """Analyze winning and losing streaks
        
        Returns:
            Dict with streak information
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            result = await self.topstepx.search_trades(start_date)
            trades = result.get("trades", [])
            
            # Sort trades by timestamp
            trades.sort(key=lambda x: x["creationTimestamp"])
            
            current_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            streaks = []
            
            for trade in trades:
                pnl = trade.get("profitAndLoss")
                if pnl is not None:
                    pnl_value = float(pnl)
                    
                    if pnl_value > 0:  # Win
                        if current_streak >= 0:
                            current_streak += 1
                        else:
                            # End of losing streak
                            if abs(current_streak) > max_loss_streak:
                                max_loss_streak = abs(current_streak)
                            current_streak = 1
                    else:  # Loss
                        if current_streak <= 0:
                            current_streak -= 1
                        else:
                            # End of winning streak
                            if current_streak > max_win_streak:
                                max_win_streak = current_streak
                            current_streak = -1
                    
                    streaks.append(current_streak)
            
            # Check final streak
            if current_streak > 0 and current_streak > max_win_streak:
                max_win_streak = current_streak
            elif current_streak < 0 and abs(current_streak) > max_loss_streak:
                max_loss_streak = abs(current_streak)
            
            # Calculate average streak length
            avg_win_streak = self._calculate_avg_streak(streaks, positive=True)
            avg_loss_streak = self._calculate_avg_streak(streaks, positive=False)
            
            return {
                "current_streak": current_streak,
                "current_streak_type": "winning" if current_streak > 0 else "losing" if current_streak < 0 else "none",
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
                "avg_win_streak": round(avg_win_streak, 1),
                "avg_loss_streak": round(avg_loss_streak, 1),
                "streak_history": streaks[-20:] if len(streaks) > 20 else streaks  # Last 20 trades
            }
            
        except Exception as e:
            logger.error(f"Streak analysis error: {e}")
            return {}
    
    async def calculate_risk_metrics(self, days: int) -> Dict:
        """Calculate risk-related metrics
        
        Returns:
            Dict with risk metrics
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            result = await self.topstepx.search_trades(start_date)
            trades = result.get("trades", [])
            
            if not trades:
                return {}
            
            # Extract P&L values
            pnls = []
            daily_pnls = defaultdict(float)
            
            for trade in trades:
                pnl = trade.get("profitAndLoss")
                if pnl is not None:
                    pnl_value = float(pnl)
                    pnls.append(pnl_value)
                    
                    # Group by day for daily P&L
                    timestamp = datetime.fromisoformat(trade["creationTimestamp"].replace("+00:00", ""))
                    date_key = timestamp.date()
                    daily_pnls[date_key] += pnl_value
            
            if not pnls:
                return {}
            
            # Calculate metrics
            pnls_array = np.array(pnls)
            daily_pnls_array = np.array(list(daily_pnls.values()))
            
            # Sharpe Ratio (simplified - using daily returns)
            if len(daily_pnls_array) > 1:
                daily_returns = np.diff(daily_pnls_array)
                sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            else:
                sharpe = 0
            
            # Maximum Drawdown
            cumsum = np.cumsum(pnls_array)
            running_max = np.maximum.accumulate(cumsum)
            drawdown = cumsum - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(pnls_array, 5) if len(pnls_array) > 0 else 0
            
            # Average Risk/Reward
            wins = pnls_array[pnls_array > 0]
            losses = np.abs(pnls_array[pnls_array < 0])
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Recovery Factor (Total Profit / Max Drawdown)
            total_profit = np.sum(pnls_array)
            recovery_factor = abs(total_profit / max_drawdown) if max_drawdown < 0 else float('inf')
            
            return {
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_drawdown, 2),
                "value_at_risk_95": round(var_95, 2),
                "avg_risk_reward": round(risk_reward, 2),
                "recovery_factor": round(recovery_factor, 2),
                "daily_volatility": round(np.std(daily_pnls_array), 2) if len(daily_pnls_array) > 0 else 0,
                "best_day": round(np.max(daily_pnls_array), 2) if len(daily_pnls_array) > 0 else 0,
                "worst_day": round(np.min(daily_pnls_array), 2) if len(daily_pnls_array) > 0 else 0,
                "profitable_days": sum(1 for pnl in daily_pnls_array if pnl > 0),
                "losing_days": sum(1 for pnl in daily_pnls_array if pnl < 0)
            }
            
        except Exception as e:
            logger.error(f"Risk metrics error: {e}")
            return {}
    
    async def generate_insights(self, days: int = 7) -> List[str]:
        """Generate actionable insights from trade analysis
        
        Returns:
            List of insight strings
        """
        try:
            insights = []
            
            # Get comprehensive analysis
            summary = await self.get_performance_summary(days)
            
            # Basic performance insights
            stats = summary.get("basic_stats", {})
            if stats:
                win_rate = stats.get("win_rate", 0)
                if win_rate > 70:
                    insights.append(f"🎯 Excellent win rate of {win_rate}% - current strategy is highly effective")
                elif win_rate < 50:
                    insights.append(f"⚠️ Win rate of {win_rate}% is below 50% - consider adjusting entry criteria")
                
                profit_factor = stats.get("profit_factor", 0)
                if profit_factor > 2:
                    insights.append(f"💪 Strong profit factor of {profit_factor} - wins significantly outweigh losses")
                elif profit_factor < 1:
                    insights.append(f"🔴 Profit factor below 1 - losses exceed wins, review risk management")
            
            # Time-based insights
            time_analysis = summary.get("time_analysis", {})
            best_hours = time_analysis.get("best_hours", [])
            if best_hours:
                best_hour = best_hours[0]
                insights.append(f"⏰ Best performance at {best_hour['hour']}:00 with {best_hour['win_rate']}% win rate")
            
            worst_hours = time_analysis.get("worst_hours", [])
            if worst_hours:
                worst_hour = worst_hours[0]
                insights.append(f"🚫 Avoid trading at {worst_hour['hour']}:00 - only {worst_hour['win_rate']}% win rate")
            
            # Pattern insights
            pattern_perf = summary.get("pattern_performance", {})
            top_patterns = pattern_perf.get("top_patterns", [])
            if top_patterns:
                best_pattern = top_patterns[0]
                insights.append(f"🏆 '{best_pattern['pattern']}' pattern generated ${best_pattern['total_pnl']:.2f} profit")
            
            # Streak insights
            streaks = summary.get("streak_analysis", {})
            current_streak = streaks.get("current_streak", 0)
            if current_streak > 3:
                insights.append(f"🔥 On a {current_streak}-trade winning streak! Consider taking profits")
            elif current_streak < -2:
                insights.append(f"💔 {abs(current_streak)} losses in a row - consider reducing position size")
            
            # Risk insights
            risk = summary.get("risk_metrics", {})
            max_dd = risk.get("max_drawdown", 0)
            if max_dd < -500:
                insights.append(f"📉 Maximum drawdown of ${abs(max_dd):.2f} - tighten risk controls")
            
            sharpe = risk.get("sharpe_ratio", 0)
            if sharpe > 2:
                insights.append(f"⭐ Excellent Sharpe ratio of {sharpe} - strong risk-adjusted returns")
            elif sharpe < 0.5:
                insights.append(f"📊 Low Sharpe ratio of {sharpe} - returns not compensating for risk")
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return ["Unable to generate insights at this time"]
    
    # Helper methods
    def _extract_pattern_name(self, custom_tag: str) -> str:
        """Extract pattern name from custom tag"""
        if not custom_tag or custom_tag == "unknown":
            return "Manual"
        
        # Parse pattern from tag format: "pattern_name_123"
        parts = custom_tag.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:-1])  # Remove trailing ID
        return custom_tag
    
    def _calculate_profit_factor(self, stats: Dict) -> float:
        """Calculate profit factor from stats"""
        total_wins = 0
        total_losses = 0
        
        for trade in stats.get("trades", []):
            pnl = trade["pnl"]
            if pnl > 0:
                total_wins += pnl
            else:
                total_losses += abs(pnl)
        
        return round(total_wins / total_losses, 2) if total_losses > 0 else float('inf') if total_wins > 0 else 0
    
    def _calculate_avg_streak(self, streaks: List[int], positive: bool) -> float:
        """Calculate average streak length"""
        if positive:
            win_streaks = [s for s in streaks if s > 0]
            return np.mean(win_streaks) if win_streaks else 0
        else:
            loss_streaks = [abs(s) for s in streaks if s < 0]
            return np.mean(loss_streaks) if loss_streaks else 0


# Example usage
async def test_analytics():
    """Test the analytics service"""
    from brokers.topstepx_client import topstepx_client
    
    # Initialize service
    analytics = TradeAnalyticsService(topstepx_client)
    
    # Get performance summary
    summary = await analytics.get_performance_summary(days=30)
    print("Performance Summary:", summary)
    
    # Generate insights
    insights = await analytics.generate_insights(days=30)
    print("\nInsights:")
    for insight in insights:
        print(f"  {insight}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_analytics())