"""
Trade Search and Analytics API Endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("/search")
async def search_trades(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    days: Optional[int] = Query(7, description="Number of days to search (if dates not provided)")
) -> Dict:
    """Search for trades within a date range
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        days: Number of days to search backward from today (default: 7)
    
    Returns:
        Dict containing trades and metadata
    """
    try:
        from brokers.topstepx_client import topstepx_client
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        # Parse dates
        if start_date:
            start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        else:
            start = datetime.now(timezone.utc) - timedelta(days=days)
        
        end = None
        if end_date:
            end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        
        # Search trades
        result = await topstepx_client.search_trades(start, end)
        
        # Add summary statistics
        trades = result.get("trades", [])
        total_pnl = sum(float(t.get("profitAndLoss", 0)) for t in trades if t.get("profitAndLoss") is not None)
        total_fees = sum(float(t.get("fees", 0)) for t in trades)
        
        result["summary"] = {
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(total_pnl - total_fees, 2)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Trade search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily")
async def get_daily_trades() -> Dict:
    """Get today's trades
    
    Returns:
        Dict containing today's trades and daily statistics
    """
    try:
        from brokers.topstepx_client import topstepx_client
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        # Get daily trades
        trades = await topstepx_client.get_daily_trades()
        
        # Calculate daily stats
        daily_pnl = await topstepx_client.calculate_daily_pnl()
        
        # Group trades by hour for intraday analysis
        hourly_pnl = {}
        for trade in trades:
            if trade.get("profitAndLoss") is not None:
                timestamp = datetime.fromisoformat(trade["creationTimestamp"].replace("+00:00", ""))
                hour = timestamp.hour
                
                if hour not in hourly_pnl:
                    hourly_pnl[hour] = {"count": 0, "pnl": 0}
                
                hourly_pnl[hour]["count"] += 1
                hourly_pnl[hour]["pnl"] += float(trade["profitAndLoss"])
        
        return {
            "trades": trades,
            "daily_stats": {
                "total_trades": len(trades),
                "net_pnl": daily_pnl,
                "hourly_breakdown": hourly_pnl
            }
        }
        
    except Exception as e:
        logger.error(f"Daily trades error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{days}")
async def get_trade_statistics(days: int = 7) -> Dict:
    """Get trade statistics for specified number of days
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Dict with comprehensive trade statistics
    """
    try:
        from brokers.topstepx_client import topstepx_client
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        stats = await topstepx_client.get_trade_statistics(days)
        
        return {
            "period_days": days,
            "statistics": stats,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trade statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{days}")
async def get_performance_analysis(days: int = 7) -> Dict:
    """Get comprehensive performance analysis
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Dict with detailed performance metrics and insights
    """
    try:
        from brokers.topstepx_client import topstepx_client
        from services.trade_analytics import TradeAnalyticsService
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        # Initialize analytics service
        analytics = TradeAnalyticsService(topstepx_client)
        
        # Get comprehensive analysis
        performance = await analytics.get_performance_summary(days)
        
        # Generate insights
        insights = await analytics.generate_insights(days)
        
        return {
            "performance": performance,
            "insights": insights,
            "recommendation": _generate_recommendation(performance)
        }
        
    except Exception as e:
        logger.error(f"Performance analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/time")
async def analyze_by_time(days: int = Query(7, description="Number of days to analyze")) -> Dict:
    """Analyze trading performance by time of day
    
    Returns:
        Dict with time-based performance analysis
    """
    try:
        from brokers.topstepx_client import topstepx_client
        from services.trade_analytics import TradeAnalyticsService
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        analytics = TradeAnalyticsService(topstepx_client)
        time_analysis = await analytics.analyze_by_time_of_day(days)
        
        return {
            "analysis": time_analysis,
            "recommendation": _generate_time_recommendation(time_analysis)
        }
        
    except Exception as e:
        logger.error(f"Time analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/patterns")
async def analyze_by_pattern(days: int = Query(7, description="Number of days to analyze")) -> Dict:
    """Analyze trading performance by pattern
    
    Returns:
        Dict with pattern-specific performance analysis
    """
    try:
        from brokers.topstepx_client import topstepx_client
        from services.trade_analytics import TradeAnalyticsService
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        analytics = TradeAnalyticsService(topstepx_client)
        pattern_analysis = await analytics.analyze_by_pattern(days)
        
        return {
            "analysis": pattern_analysis,
            "recommendation": _generate_pattern_recommendation(pattern_analysis)
        }
        
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/streaks")
async def analyze_streaks(days: int = Query(7, description="Number of days to analyze")) -> Dict:
    """Analyze winning and losing streaks
    
    Returns:
        Dict with streak analysis
    """
    try:
        from brokers.topstepx_client import topstepx_client
        from services.trade_analytics import TradeAnalyticsService
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        analytics = TradeAnalyticsService(topstepx_client)
        streak_analysis = await analytics.analyze_streaks(days)
        
        return {
            "analysis": streak_analysis,
            "recommendation": _generate_streak_recommendation(streak_analysis)
        }
        
    except Exception as e:
        logger.error(f"Streak analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/risk")
async def analyze_risk_metrics(days: int = Query(7, description="Number of days to analyze")) -> Dict:
    """Calculate risk metrics for trading performance
    
    Returns:
        Dict with risk metrics and recommendations
    """
    try:
        from brokers.topstepx_client import topstepx_client
        from services.trade_analytics import TradeAnalyticsService
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        analytics = TradeAnalyticsService(topstepx_client)
        risk_metrics = await analytics.calculate_risk_metrics(days)
        
        return {
            "metrics": risk_metrics,
            "risk_level": _calculate_risk_level(risk_metrics),
            "recommendation": _generate_risk_recommendation(risk_metrics)
        }
        
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_trading_insights(days: int = Query(7, description="Number of days to analyze")) -> Dict:
    """Get AI-generated trading insights
    
    Returns:
        Dict with actionable insights
    """
    try:
        from brokers.topstepx_client import topstepx_client
        from services.trade_analytics import TradeAnalyticsService
        
        if not topstepx_client.connected:
            await topstepx_client.connect()
        
        analytics = TradeAnalyticsService(topstepx_client)
        insights = await analytics.generate_insights(days)
        
        return {
            "insights": insights,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for recommendations
def _generate_recommendation(performance: Dict) -> str:
    """Generate overall recommendation based on performance"""
    stats = performance.get("basic_stats", {})
    win_rate = stats.get("win_rate", 0)
    profit_factor = stats.get("profit_factor", 0)
    
    if win_rate > 70 and profit_factor > 2:
        return "Excellent performance! Consider increasing position size gradually."
    elif win_rate > 60 and profit_factor > 1.5:
        return "Good performance. Maintain current strategy and risk parameters."
    elif win_rate < 50 or profit_factor < 1:
        return "Performance needs improvement. Review entry criteria and risk management."
    else:
        return "Moderate performance. Focus on consistency and risk control."


def _generate_time_recommendation(analysis: Dict) -> str:
    """Generate recommendation based on time analysis"""
    best_hours = analysis.get("best_hours", [])
    worst_hours = analysis.get("worst_hours", [])
    
    if best_hours and worst_hours:
        best = best_hours[0]["hour"]
        worst = worst_hours[0]["hour"]
        return f"Focus trading during hour {best}:00 and avoid hour {worst}:00 for optimal results."
    elif best_hours:
        best = best_hours[0]["hour"]
        return f"Best performance observed during hour {best}:00. Consider concentrating trades in this period."
    else:
        return "Insufficient data for time-based recommendations. Continue trading to build history."


def _generate_pattern_recommendation(analysis: Dict) -> str:
    """Generate recommendation based on pattern analysis"""
    top_patterns = analysis.get("top_patterns", [])
    
    if top_patterns:
        best = top_patterns[0]
        return f"'{best['pattern']}' pattern shows best results with {best['win_rate']}% win rate. Prioritize this setup."
    else:
        return "No clear pattern advantage detected. Continue monitoring pattern performance."


def _generate_streak_recommendation(analysis: Dict) -> str:
    """Generate recommendation based on streak analysis"""
    current = analysis.get("current_streak", 0)
    
    if current > 3:
        return f"On a {current}-trade winning streak! Consider taking some profits and reducing position size."
    elif current < -2:
        return f"After {abs(current)} consecutive losses, consider taking a break or reducing position size."
    else:
        return "Streak patterns are normal. Maintain disciplined approach."


def _calculate_risk_level(metrics: Dict) -> str:
    """Calculate overall risk level"""
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)
    
    if sharpe > 1.5 and max_dd > -300:
        return "LOW"
    elif sharpe > 1 and max_dd > -500:
        return "MODERATE"
    elif sharpe > 0.5 and max_dd > -750:
        return "ELEVATED"
    else:
        return "HIGH"


def _generate_risk_recommendation(metrics: Dict) -> str:
    """Generate recommendation based on risk metrics"""
    risk_level = _calculate_risk_level(metrics)
    
    if risk_level == "LOW":
        return "Risk metrics are excellent. Current strategy provides good risk-adjusted returns."
    elif risk_level == "MODERATE":
        return "Risk levels are acceptable. Monitor drawdowns and maintain stop losses."
    elif risk_level == "ELEVATED":
        return "Risk levels are elevated. Consider tightening stop losses and reducing position sizes."
    else:
        return "High risk detected. Immediate review of risk management required."