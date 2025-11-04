"""
FastAPI Backend for NQ Trading Platform
Following NovaGent Design System principles
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import sys

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.smart_scalper_enhanced import enhanced_scalper
try:
    from agents.scalping_patterns import ScalpingPatternAgent
except ImportError:
    ScalpingPatternAgent = None
from utils.logger import setup_logger
from static_files import setup_static_files
from services.pattern_statistics import pattern_stats_service
from services.ai_thoughts_generator import ai_thoughts_generator

# Database imports
from database.connection import init_db, get_db_session, db_manager
from database.models import Trade, Pattern, ShadowTrade, PatternStatus, BacktestResult

# Shadow trading import
from shadow_trading.shadow_manager import shadow_manager

# Strategy and compliance imports
from strategies.orchestrator import strategy_orchestrator
from strategies.strategy_manager import strategy_manager
from topstepx.compliance import topstepx_compliance
from manus.strategy_importer import manus_importer
from risk_management.risk_manager import risk_manager
from brokers.topstepx_client import topstepx_client
from utils.market_hours import market_hours

# Real-time and analytics imports
from brokers.topstepx_realtime import TopStepXRealTimeClient
from services.trade_analytics import TradeAnalyticsService
from api.trades import router as trades_router

# Logger
logger = setup_logger('WebPlatform')

# Global instances
realtime_client = None
trade_analytics = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.smart_scalper = None
        self.pattern_agent = None
        self.monitoring = False
        self.realtime_client = None
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, data: Dict):
        """Send data to all connected clients"""
        message = json.dumps(data)
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)
                
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

# Initialize connection manager
manager = ConnectionManager()

# Background task for monitoring
async def monitor_loop():
    """Background monitoring loop"""
    while manager.monitoring:
        try:
            if manager.smart_scalper:
                # Get current status
                status = await manager.smart_scalper.monitor_and_trade()
                
                # Broadcast to all clients
                await manager.broadcast({
                    'type': 'status_update',
                    'data': status,
                    'timestamp': datetime.now().isoformat()
                })
                
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Monitor loop error: {e}")
            await asyncio.sleep(5)

# Background task for market condition tracking
async def market_condition_tracker():
    """Track market conditions periodically - only when trading is active"""
    while True:
        try:
            # Only update market conditions when actively trading
            if manager.monitoring:
                from ai.market_tracker import market_tracker
                from database.connection import get_db_session
                from database.models import MarketCondition
                
                # Update market conditions every 5 minutes
                conditions = await market_tracker.track_market_conditions()
                
                # Store in database
                with get_db_session() as session:
                    market_record = MarketCondition(
                        timestamp=datetime.now(),
                        symbol='NQ',
                        price=conditions.get('support_resistance', {}).get('current_price', 0),
                        atr=conditions.get('volatility', {}).get('atr', 0),
                        std_dev=conditions.get('volatility', {}).get('std_dev', 0),
                        volatility_level=conditions.get('volatility', {}).get('volatility_level', 'unknown'),
                        volatility_percentage=conditions.get('volatility', {}).get('volatility_percentage', 0),
                        trend_direction=conditions.get('trend', {}).get('direction', 'unknown'),
                        trend_strength=conditions.get('trend', {}).get('strength', 0),
                        sma_5=conditions.get('trend', {}).get('sma_5', 0),
                        sma_10=conditions.get('trend', {}).get('sma_10', 0),
                        sma_20=conditions.get('trend', {}).get('sma_20', 0),
                        support_levels=conditions.get('support_resistance', {}).get('support', []),
                        resistance_levels=conditions.get('support_resistance', {}).get('resistance', []),
                        volume_trend=conditions.get('volume', {}).get('volume_trend', 'unknown'),
                        vwap=conditions.get('volume', {}).get('current_vwap', 0),
                        high_volume_nodes=conditions.get('volume', {}).get('high_volume_nodes', []),
                        volume_ratio=conditions.get('volume', {}).get('volume_ratio', 1),
                        market_regime=conditions.get('market_regime', 'unknown'),
                        raw_data=conditions
                    )
                    session.add(market_record)
                    session.commit()
                    
                logger.info(f"Market conditions updated: {conditions.get('market_regime')}")
            
        except Exception as e:
            logger.error(f"Market condition tracker error: {e}")
            
        await asyncio.sleep(300)  # Update every 5 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting NQ Trading Platform Backend")
    
    # Initialize database
    init_db()
    logger.info("✅ Database initialized")
    
    # Initialize enhanced scalper
    manager.smart_scalper = enhanced_scalper
    manager.pattern_agent = None  # Not needed with enhanced scalper
    
    # Initialize smart scalper
    initialized = await manager.smart_scalper.initialize()
    if initialized:
        logger.info("✅ Smart Scalper initialized")
        # Don't start monitoring automatically - wait for user to click Start Trading
        manager.monitoring = False
        logger.info("⏸️ Monitoring paused - click 'Start Trading' to begin")
    else:
        logger.error("❌ Failed to initialize Smart Scalper")
    
    # Initialize TopStepX broker connection
    from brokers.topstepx_client import topstepx_client
    connected = await topstepx_client.connect()
    if connected:
        logger.info("✅ Connected to TopStepX broker")
        
        # Initialize real-time SignalR client if we have a session token
        if topstepx_client.session_token and topstepx_client.account_id:
            try:
                global realtime_client, trade_analytics
                realtime_client = TopStepXRealTimeClient(
                    jwt_token=topstepx_client.session_token,
                    account_id=topstepx_client.account_id
                )
                
                # Register callbacks for real-time updates
                async def on_order_update(data):
                    await manager.broadcast({
                        'type': 'order_update',
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    })
                
                async def on_position_update(data):
                    await manager.broadcast({
                        'type': 'position_update',
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    })
                
                async def on_trade_update(data):
                    await manager.broadcast({
                        'type': 'trade_update',
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    })
                
                async def on_quote_update(data):
                    await manager.broadcast({
                        'type': 'quote_update',
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    })
                
                realtime_client.on_order(on_order_update)
                realtime_client.on_position(on_position_update)
                realtime_client.on_trade(on_trade_update)
                realtime_client.on_quote(on_quote_update)
                
                # Connect to SignalR hubs
                if await realtime_client.connect():
                    logger.info("✅ Connected to TopStepX real-time feeds")
                    manager.realtime_client = realtime_client
                else:
                    logger.warning("⚠️ SignalR connection failed - will use polling")
                    
                # Initialize trade analytics service
                trade_analytics = TradeAnalyticsService(topstepx_client)
                logger.info("✅ Trade analytics service initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize real-time client: {e}")
                
    else:
        logger.warning("⚠️ TopStepX connection failed - will retry when trading starts")
    
    # Start market condition tracker (will only run when monitoring is active)
    asyncio.create_task(market_condition_tracker())
    logger.info("📊 Market condition tracker ready (will activate with trading)")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down NQ Trading Platform Backend")
    manager.monitoring = False
    if manager.smart_scalper:
        await manager.smart_scalper.cleanup()
    
    # Disconnect SignalR if connected
    if realtime_client and realtime_client.connected:
        await realtime_client.disconnect()
        logger.info("🔌 Disconnected from real-time feeds")

# Create FastAPI app
app = FastAPI(
    title="NQ Trading Platform",
    description="Smart Scalping Platform with S/R Bounce Strategy",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static file serving for frontend
app = setup_static_files(app)

# Include trade API routes
app.include_router(trades_router)

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "NQ Trading Platform",
        "status": "running",
        "monitoring": manager.monitoring,
        "connected_clients": len(manager.active_connections)
    }

@app.get("/api/status")
async def get_status():
    """Get current system status with real broker data"""
    if not manager.smart_scalper:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Get real account info from TopStepX
    from brokers.topstepx_client import topstepx_client
    from utils.market_hours import market_hours
    
    # Get market hours info
    market_info = market_hours.get_trading_hours_info()
    
    # Get account data if connected
    account_info = {}
    if topstepx_client.connected:
        account_info = await topstepx_client.get_account_info()
        current_price = await topstepx_client.get_market_price()
    else:
        current_price = await manager.smart_scalper.get_current_price()
    
    return {
        "current_price": current_price,
        "support_levels": manager.smart_scalper.support_levels,
        "resistance_levels": manager.smart_scalper.resistance_levels,
        "current_position": manager.smart_scalper.current_position,
        "daily_pnl": account_info.get('dailyPnL', manager.smart_scalper.daily_pnl),
        "trades_today": len(manager.smart_scalper.trades_today),
        "max_trades": manager.smart_scalper.max_trades_today,
        "can_trade": manager.smart_scalper.can_trade(),
        "monitoring": manager.monitoring,
        "broker_connected": topstepx_client.connected,
        "account_balance": account_info.get('balance', 25000),
        "open_positions": len(topstepx_client.open_positions),
        "market_open": market_info['is_open'],
        "market_session": market_info['session'],
        "next_open": market_info['next_open'],
        "next_close": market_info['next_close']
    }

@app.get("/api/patterns")
async def get_patterns():
    """Get all patterns with statistics"""
    patterns = await pattern_stats_service.get_all_patterns()
    
    return {
        "patterns": patterns,
        "count": len(patterns),
        "best_pattern": patterns[0] if patterns else None
    }

@app.get("/api/patterns/{pattern_id}")
async def get_pattern_detail(pattern_id: str):
    """Get detailed pattern statistics"""
    pattern = await pattern_stats_service.get_pattern(pattern_id)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    return pattern

@app.get("/api/patterns/{pattern_id}/performance")
async def get_pattern_performance(pattern_id: str, timeframe: str = "7d"):
    """Get pattern performance over timeframe"""
    performance = await pattern_stats_service.get_pattern_performance(pattern_id, timeframe)
    if not performance:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    return performance

@app.get("/api/patterns/top/{metric}")
async def get_top_patterns(metric: str = "win_rate", limit: int = 5):
    """Get top performing patterns by metric"""
    patterns = await pattern_stats_service.get_top_patterns(metric, limit)
    return {
        "metric": metric,
        "patterns": patterns
    }

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    trades = []
    
    if manager.smart_scalper and manager.smart_scalper.trades_today:
        for trade in manager.smart_scalper.trades_today:
            trades.append({
                "entry_time": trade.get('entry_time', '').isoformat() if trade.get('entry_time') else None,
                "direction": trade.get('direction'),
                "entry_price": trade.get('entry_price'),
                "stop_loss": trade.get('stop_loss'),
                "take_profit": trade.get('take_profit'),
                "level": trade.get('level')
            })
    
    return {
        "trades": trades,
        "count": len(trades),
        "daily_pnl": manager.smart_scalper.daily_pnl if manager.smart_scalper else 0
    }

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    if not manager.smart_scalper:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    total_trades = len(manager.smart_scalper.trades_today)
    
    # Calculate win rate (simplified - would need actual results)
    wins = sum(1 for t in manager.smart_scalper.trades_today if t.get('pnl', 0) > 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        "daily_pnl": manager.smart_scalper.daily_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "consecutive_losses": manager.smart_scalper.consecutive_losses,
        "target_pattern": "Support_Resistance_Bounce",
        "expected_win_rate": 89.5,
        "risk_reward_ratio": "1:1"
    }

@app.post("/api/control/start")
async def start_monitoring():
    """Start monitoring and trading with strategy orchestration"""
    if not manager.smart_scalper:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Check if market is open
    from utils.market_hours import is_market_open, get_market_info
    if not is_market_open():
        market_info = get_market_info()
        raise HTTPException(
            status_code=400, 
            detail=f"Market is closed. Opens at {market_info['next_open']}"
        )
    
    # Check TopStepX compliance status
    compliance_status = await topstepx_compliance.get_compliance_status()
    if compliance_status['recovery_mode']:
        logger.warning("⚠️ Starting in recovery mode - limited trading")
    
    manager.monitoring = True
    asyncio.create_task(monitor_loop())
    
    # Start pattern discovery
    from patterns.pattern_discovery import pattern_discovery
    pattern_discovery.monitoring = True
    asyncio.create_task(pattern_discovery.continuous_discovery())
    logger.info("🔍 Pattern discovery started")
    
    # Start strategy orchestrator
    await strategy_orchestrator.start()
    logger.info("🎭 Strategy orchestrator started")
    
    return {
        "status": "started",
        "message": "Monitoring, pattern discovery, and strategy orchestration started",
        "compliance": compliance_status
    }

@app.post("/api/control/stop")
async def stop_monitoring():
    """Stop monitoring and trading"""
    manager.monitoring = False
    
    # Stop pattern discovery
    from patterns.pattern_discovery import pattern_discovery
    pattern_discovery.monitoring = False
    logger.info("⏸️ Pattern discovery stopped")
    
    # Stop strategy orchestrator
    await strategy_orchestrator.stop()
    logger.info("🛑 Strategy orchestrator stopped")
    
    return {"status": "stopped", "message": "All trading systems stopped"}

@app.post("/api/control/reset")
async def reset_daily_stats():
    """Reset daily statistics"""
    if not manager.smart_scalper:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    manager.smart_scalper.daily_pnl = 0
    manager.smart_scalper.max_trades_today = 0
    manager.smart_scalper.consecutive_losses = 0
    manager.smart_scalper.trades_today = []
    
    return {"status": "reset", "message": "Daily stats reset"}

# Shadow Trading Endpoints
@app.get("/api/shadow/trades/{pattern_id}")
async def get_shadow_trades(pattern_id: str, limit: int = 100):
    """Get shadow trades for a specific pattern"""
    with get_db_session() as session:
        shadow_trades = db_manager.get_recent_shadow_trades(session, pattern_id, limit)
        return {
            "pattern_id": pattern_id,
            "shadow_trades": [
                {
                    "id": st.id,
                    "entry_price": st.entry_price,
                    "exit_price": st.exit_price,
                    "pnl": st.pnl,
                    "confidence": st.confidence,
                    "would_have_traded": st.would_have_traded,
                    "created_at": st.created_at.isoformat() if st.created_at else None
                }
                for st in shadow_trades
            ],
            "count": len(shadow_trades)
        }

@app.get("/api/shadow/stats/{pattern_id}")
async def get_shadow_stats(pattern_id: str):
    """Get shadow trading statistics for a pattern"""
    stats = await shadow_manager.get_shadow_performance(pattern_id)
    return stats

@app.get("/api/shadow/compare/{pattern_id}")
async def compare_shadow_vs_live(pattern_id: str):
    """Compare shadow vs live performance"""
    comparison = await shadow_manager.get_shadow_vs_live_comparison(pattern_id)
    return comparison

@app.post("/api/patterns/deploy/{pattern_id}")
async def deploy_pattern(pattern_id: str):
    """Deploy a pattern for live trading"""
    with get_db_session() as session:
        pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Check if pattern meets deployment criteria
        if pattern.shadow_trades < 50:
            raise HTTPException(
                status_code=400, 
                detail=f"Pattern needs at least 50 shadow trades (current: {pattern.shadow_trades})"
            )
        
        if pattern.shadow_win_rate < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Pattern win rate too low: {pattern.shadow_win_rate}% (minimum: 60%)"
            )
        
        # Deploy pattern
        pattern.is_deployed = True
        pattern.status = PatternStatus.ACTIVE
        session.commit()
        
        # Broadcast update
        await manager.broadcast({
            "type": "pattern_deployed",
            "data": {
                "pattern_id": pattern_id,
                "pattern_name": pattern.name,
                "win_rate": pattern.shadow_win_rate
            }
        })
        
        return {
            "status": "deployed",
            "pattern_id": pattern_id,
            "message": f"Pattern {pattern.name} deployed successfully"
        }

@app.post("/api/patterns/pause/{pattern_id}")
async def pause_pattern(pattern_id: str):
    """Pause a pattern from trading"""
    with get_db_session() as session:
        pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        pattern.is_deployed = False
        pattern.status = PatternStatus.PAUSED
        session.commit()
        
        return {
            "status": "paused",
            "pattern_id": pattern_id,
            "message": f"Pattern {pattern.name} paused"
        }

@app.put("/api/patterns/{pattern_id}/config")
async def update_pattern_config(pattern_id: str, config: dict):
    """Update pattern configuration"""
    with get_db_session() as session:
        pattern = session.query(Pattern).filter_by(pattern_id=pattern_id).first()
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Update risk parameters
        if "risk_params" in config:
            pattern.risk_params = config["risk_params"]
        
        # Update entry/exit rules
        if "entry_rules" in config:
            pattern.entry_rules = config["entry_rules"]
        if "exit_rules" in config:
            pattern.exit_rules = config["exit_rules"]
        
        session.commit()
        
        return {
            "status": "updated",
            "pattern_id": pattern_id,
            "config": config
        }

@app.get("/api/patterns/deployable")
async def get_deployable_patterns():
    """Get patterns ready for deployment"""
    with get_db_session() as session:
        patterns = db_manager.get_deployable_patterns(session)
        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "name": p.name,
                    "shadow_win_rate": p.shadow_win_rate,
                    "shadow_trades": p.shadow_trades,
                    "expected_value": p.expected_value
                }
                for p in patterns
            ],
            "count": len(patterns)
        }

@app.get("/api/discoveries/recent")
async def get_recent_discoveries(limit: int = 10):
    """Get recently discovered patterns"""
    with get_db_session() as session:
        from database.models import PatternDiscovery
        discoveries = session.query(PatternDiscovery)\
            .order_by(PatternDiscovery.created_at.desc())\
            .limit(limit)\
            .all()
        
        return {
            "discoveries": [
                {
                    "id": d.id,
                    "pattern_type": d.pattern_type,
                    "first_detected": d.first_detected.isoformat() if d.first_detected else None,
                    "shadow_test_count": d.shadow_test_count,
                    "shadow_win_rate": d.shadow_win_rate,
                    "promoted_to_testing": d.promoted_to_testing
                }
                for d in discoveries
            ],
            "count": len(discoveries)
        }

# Backtesting endpoints
@app.post("/api/backtest/pattern/{pattern_name}")
async def run_pattern_backtest(pattern_name: str, days: int = 30):
    """Run backtest for a specific pattern"""
    from backtesting.backtest_engine import backtest_engine
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        results = await backtest_engine.run_pattern_backtest(pattern_name, start_date, end_date)
        
        return {
            "pattern_name": results.pattern_name,
            "total_trades": results.total_trades,
            "win_rate": results.win_rate,
            "total_pnl": results.total_pnl,
            "sharpe_ratio": results.sharpe_ratio,
            "profit_factor": results.profit_factor,
            "avg_win": results.avg_win,
            "avg_loss": results.avg_loss,
            "max_drawdown": results.max_drawdown,
            "expectancy": results.expectancy,
            "equity_curve": [(str(date), equity) for date, equity in results.equity_curve]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.post("/api/backtest/optimize/{pattern_name}")
async def optimize_pattern(pattern_name: str):
    """Run parameter optimization for a pattern"""
    from backtesting.backtest_engine import backtest_engine
    
    try:
        optimization = await backtest_engine.run_pattern_optimization(pattern_name)
        
        return {
            "pattern_name": optimization['pattern_name'],
            "best_params": optimization['best_params'],
            "best_results": {
                "win_rate": optimization['best_results'].win_rate,
                "total_pnl": optimization['best_results'].total_pnl,
                "sharpe_ratio": optimization['best_results'].sharpe_ratio,
                "profit_factor": optimization['best_results'].profit_factor
            },
            "optimization_results": optimization['all_results']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/api/backtest/walkforward/{pattern_name}")
async def run_walkforward_analysis(pattern_name: str, periods: int = 6):
    """Run walk-forward analysis for robustness testing"""
    from backtesting.backtest_engine import backtest_engine
    
    try:
        analysis = await backtest_engine.run_walk_forward_analysis(pattern_name, periods)
        
        return {
            "pattern_name": analysis['pattern_name'],
            "metrics": analysis['metrics'],
            "periods": len(analysis['periods'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Walk-forward analysis failed: {str(e)}")

@app.get("/api/backtest/results/{pattern_name}")
async def get_backtest_results(pattern_name: str, limit: int = 10):
    """Get historical backtest results for a pattern"""
    with get_db_session() as session:
        results = session.query(BacktestResult).filter_by(
            pattern_name=pattern_name
        ).order_by(BacktestResult.created_at.desc()).limit(limit).all()
        
        return {
            "pattern_name": pattern_name,
            "results": [
                {
                    "id": r.id,
                    "total_trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "total_pnl": r.total_pnl,
                    "sharpe_ratio": r.sharpe_ratio,
                    "profit_factor": r.profit_factor,
                    "expectancy": r.expectancy,
                    "test_period_start": r.test_period_start.isoformat() if r.test_period_start else None,
                    "test_period_end": r.test_period_end.isoformat() if r.test_period_end else None,
                    "created_at": r.created_at.isoformat()
                }
                for r in results
            ],
            "count": len(results)
        }

# Risk Management endpoints
@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics"""
    from risk_management.risk_manager import risk_manager
    
    metrics = await risk_manager.get_current_risk_metrics()
    
    return {
        "risk_score": metrics.risk_score,
        "total_exposure": metrics.total_exposure,
        "max_exposure": metrics.max_exposure,
        "current_drawdown": metrics.current_drawdown,
        "max_drawdown": metrics.max_drawdown,
        "daily_loss": metrics.daily_loss,
        "max_daily_loss": metrics.max_daily_loss,
        "open_positions": metrics.open_positions,
        "max_positions": metrics.max_positions,
        "var_95": metrics.var_95,
        "cvar_95": metrics.cvar_95,
        "kelly_fraction": metrics.kelly_fraction
    }

@app.post("/api/risk/check-trade")
async def check_trade_permission(
    pattern_name: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float
):
    """Check if a trade is allowed by risk manager"""
    from risk_management.risk_manager import risk_manager
    
    result = await risk_manager.check_trade_permission(
        pattern_name, entry_price, stop_loss, take_profit
    )
    
    return result

@app.get("/api/risk/report")
async def get_risk_report():
    """Get comprehensive risk report"""
    from risk_management.risk_manager import risk_manager
    
    report = await risk_manager.generate_risk_report()
    return report

@app.post("/api/risk/reset-daily")
async def reset_daily_metrics():
    """Reset daily risk metrics"""
    from risk_management.risk_manager import risk_manager
    
    await risk_manager.reset_daily_metrics()
    
    return {"status": "success", "message": "Daily metrics reset"}

@app.post("/api/risk/emergency-stop")
async def emergency_stop():
    """Trigger emergency stop - close all positions"""
    from risk_management.risk_manager import risk_manager
    
    await risk_manager.emergency_stop()
    
    return {"status": "success", "message": "Emergency stop executed"}

@app.post("/api/risk/position-size")
async def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    pattern_name: str
):
    """Calculate optimal position size"""
    from risk_management.risk_manager import risk_manager
    
    position_size = await risk_manager.calculate_position_size(
        entry_price, stop_loss, pattern_name
    )
    
    return {
        "shares": position_size.shares,
        "dollar_amount": position_size.dollar_amount,
        "risk_amount": position_size.risk_amount,
        "stop_distance": position_size.stop_distance,
        "risk_reward_ratio": position_size.risk_reward_ratio,
        "kelly_size": position_size.kelly_size,
        "recommended_size": position_size.recommended_size
    }

# AI Assistant endpoints
@app.post("/api/ai/chat")
async def ai_chat(request: dict):
    """Process AI chat messages"""
    try:
        from ai.chat_handler import ai_assistant
        
        user_query = request.get('query', '')
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Process query and get response
        result = await ai_assistant.process_query(user_query)
        
        return result
        
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return {
            'success': False,
            'error': str(e),
            'response': 'Sorry, I encountered an error processing your request.'
        }

@app.get("/api/ai/market/conditions")
async def get_market_conditions():
    """Get current market conditions"""
    try:
        from ai.market_tracker import market_tracker
        
        conditions = await market_tracker.get_current_conditions()
        
        return {
            'success': True,
            'data': conditions,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.get("/api/ai/analyze/performance/{days}")
async def analyze_performance(days: int = 7):
    """Analyze trading performance over specified days"""
    try:
        from ai.strategy_analyzer import strategy_analyzer
        
        performance = await strategy_analyzer.analyze_pattern_performance(days)
        summary = await strategy_analyzer.get_recent_performance()
        
        return {
            'success': True,
            'data': {
                'pattern_performance': performance,
                'summary': summary,
                'timeframe_days': days
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance analysis error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.post("/api/ai/strategy/suggest")
async def suggest_strategy():
    """Get AI strategy suggestions based on current conditions"""
    try:
        from ai.strategy_analyzer import strategy_analyzer
        from ai.market_tracker import market_tracker
        
        # Get current market conditions
        market = await market_tracker.get_current_conditions()
        
        # Get optimal parameters
        recommendations = await strategy_analyzer.calculate_optimal_parameters()
        
        # Get pattern correlations
        correlations = strategy_analyzer.find_pattern_correlations()
        
        return {
            'success': True,
            'data': {
                'market_conditions': market,
                'recommendations': recommendations,
                'pattern_correlations': correlations
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Strategy suggestion error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.get("/api/ai/suggestions")
async def get_suggested_questions():
    """Get suggested questions based on current context"""
    try:
        from ai.chat_handler import ai_assistant
        
        suggestions = await ai_assistant.get_suggested_questions()
        
        return {
            'success': True,
            'suggestions': suggestions,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return {
            'success': False,
            'error': str(e),
            'suggestions': [
                "What's the best strategy for today?",
                "Analyze my last 7 days performance",
                "Which patterns are most profitable?",
                "Should I adjust my risk parameters?"
            ]
        }

# Settings Management Endpoints
@app.get("/api/settings")
async def get_settings():
    """Get current trading settings"""
    if manager.smart_scalper:
        # Load settings from file if exists
        import json
        import os
        
        settings = {}
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        else:
            # Return current values
            settings = {
                "target_points": manager.smart_scalper.target_points,
                "stop_points": manager.smart_scalper.stop_points,
                "max_daily_trades": manager.smart_scalper.max_daily_trades,
                "max_consecutive_losses": manager.smart_scalper.max_consecutive_losses,
                "max_daily_loss": abs(manager.smart_scalper.max_daily_loss),
                "account_balance": 25000,
                "max_risk_per_trade": 2,
                "max_daily_loss_pct": 6,
                "max_open_positions": 3,
                "auto_stop_on_limit": True,
                "min_pattern_confidence": 75,
                "shadow_trading": True,
                "pattern_discovery": True,
                "market_hours_only": True,
                "extended_hours": False,
                "trade_alerts": True,
                "pattern_alerts": True,
                "risk_alerts": True,
                "alert_sound": True,
                "alert_volume": 70
            }
        return settings
    return {}

@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update trading settings"""
    try:
        if manager.smart_scalper:
            # Update scalper settings
            if 'target_points' in settings:
                manager.smart_scalper.target_points = int(settings['target_points'])
            if 'stop_points' in settings:
                manager.smart_scalper.stop_points = int(settings['stop_points'])
            if 'max_daily_trades' in settings:
                manager.smart_scalper.max_daily_trades = int(settings['max_daily_trades'])
            if 'max_consecutive_losses' in settings:
                manager.smart_scalper.max_consecutive_losses = int(settings['max_consecutive_losses'])
            if 'max_daily_loss' in settings:
                manager.smart_scalper.max_daily_loss = -abs(float(settings['max_daily_loss']))
            
            # Update risk manager settings
            from risk_management.risk_manager import risk_manager
            if 'account_balance' in settings:
                risk_manager.account_balance = float(settings['account_balance'])
            if 'max_risk_per_trade' in settings:
                risk_manager.max_risk_per_trade = float(settings['max_risk_per_trade']) / 100
            if 'max_daily_loss_pct' in settings:
                risk_manager.max_daily_loss = float(settings['max_daily_loss_pct']) / 100
            if 'max_open_positions' in settings:
                risk_manager.max_positions = int(settings['max_open_positions'])
            
            # Save settings to file for persistence
            import json
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            logger.info(f"Settings updated: {settings}")
            
            # Broadcast settings update to all connected clients
            await manager.broadcast({
                'type': 'settings_updated',
                'data': settings,
                'timestamp': datetime.now().isoformat()
            })
            
            return {"success": True, "message": "Settings updated successfully"}
        
        return {"success": False, "message": "Trading system not initialized"}
        
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        return {"success": False, "message": str(e)}

# Emergency Stop Endpoint
@app.post("/api/emergency/stop")
async def emergency_stop():
    """Emergency stop - close all positions and halt trading"""
    try:
        from brokers.topstepx_client import topstepx_client
        
        # Stop all trading
        manager.monitoring = False
        
        # Close all positions at TopStepX
        if topstepx_client.connected:
            await topstepx_client.close_all_positions()
        
        # Clear local position tracking
        if manager.smart_scalper:
            manager.smart_scalper.current_position = None
            manager.smart_scalper.monitoring = False
        
        logger.warning("🛑 EMERGENCY STOP ACTIVATED")
        
        # Broadcast emergency stop
        await manager.broadcast({
            'type': 'emergency_stop',
            'message': 'Emergency stop activated - all positions closed',
            'timestamp': datetime.now().isoformat()
        })
        
        return {"success": True, "message": "Emergency stop executed"}
        
    except Exception as e:
        logger.error(f"Emergency stop error: {e}")
        return {"success": False, "message": str(e)}

# Strategy Management Endpoints
@app.get("/api/strategies")
async def get_strategies():
    """Get strategy information"""
    report = await strategy_manager.get_strategy_report()
    orchestrator_status = await strategy_orchestrator.get_orchestrator_status()
    
    return {
        "strategy_report": report,
        "orchestrator": orchestrator_status
    }

@app.post("/api/strategies/rotate")
async def rotate_strategies():
    """Manually trigger strategy rotation"""
    if not manager.monitoring:
        raise HTTPException(status_code=400, detail="Trading not active")
    
    await strategy_orchestrator.rotate_strategies()
    
    return {
        "status": "rotated",
        "active_strategies": [s.name for s in strategy_manager.active_strategies]
    }

# Compliance Endpoint
@app.get("/api/compliance")
async def get_compliance():
    """Get TopStepX compliance status"""
    compliance = await topstepx_compliance.get_compliance_status()
    
    return compliance

# Manus AI Integration Endpoints
@app.post("/api/manus/import")
async def import_manus_strategy(strategy_data: dict):
    """Import a strategy from Manus AI"""
    result = await manus_importer.import_strategy(strategy_data)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('errors', ['Import failed']))
    
    return result

@app.get("/api/manus/status")
async def get_manus_status():
    """Get Manus import status"""
    return await manus_importer.get_import_status()

@app.post("/api/manus/deploy/{strategy_id}")
async def deploy_manus_strategy(strategy_id: str):
    """Deploy an imported Manus strategy"""
    result = await manus_importer.deploy_strategy(strategy_id)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Deployment failed'))
    
    return result

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        if manager.smart_scalper:
            current_price = await manager.smart_scalper.get_current_price()
            await websocket.send_json({
                "type": "connected",
                "data": {
                    "current_price": current_price,
                    "support_levels": manager.smart_scalper.support_levels,
                    "resistance_levels": manager.smart_scalper.resistance_levels
                }
            })
        
        # Keep connection alive
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Handle client messages
            try:
                message = json.loads(data)
                
                if message.get('type') == 'ping':
                    await websocket.send_json({"type": "pong"})
                    
                elif message.get('type') == 'get_status':
                    status = await get_status()
                    await websocket.send_json({
                        "type": "status",
                        "data": status
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# AI Thoughts WebSocket
@app.websocket("/ws/ai-thoughts")
async def ai_thoughts_websocket(websocket: WebSocket):
    """WebSocket for AI thoughts streaming - only when trading is active"""
    await websocket.accept()
    
    try:
        # Send initial greeting
        greeting = await ai_thoughts_generator.generate_specific_thought(
            'session_start', 
            {'time': datetime.now().strftime('%H:%M')}
        )
        await websocket.send_json(greeting)
        
        # Start thought generation loop
        while True:
            # Only generate thoughts when trading is active
            if manager.monitoring:
                # Reset the inactive message flag when trading starts
                if hasattr(websocket, '_inactive_message_sent'):
                    delattr(websocket, '_inactive_message_sent')
                
                # Generate a thought
                thought = await ai_thoughts_generator.generate_thought()
                await websocket.send_json(thought)
                
                # Wait before next thought
                await asyncio.sleep(random.randint(15, 45))
            else:
                # Send status update only once, then stay quiet
                if not hasattr(websocket, '_inactive_message_sent'):
                    await websocket.send_json({
                        'type': 'status',
                        'message': '⏸️ Trading inactive - click "Start Trading" to begin monitoring',
                        'category': 'info',
                        'timestamp': datetime.now().isoformat()
                    })
                    websocket._inactive_message_sent = True
                # Wait much longer when not trading (5 minutes)
                await asyncio.sleep(300)
            
    except WebSocketDisconnect:
        logger.info("AI thoughts client disconnected")
    except Exception as e:
        logger.error(f"AI thoughts WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )