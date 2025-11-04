# File: trading_bot/deployment/monitoring_dashboard.py
"""
Real-time Monitoring Dashboard - Phase 7.2
Provides live monitoring and control interface
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
from aiohttp import web
import aiohttp_cors
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """
    Web-based monitoring dashboard for trading bot
    Provides real-time metrics, controls, and alerts
    """
    
    def __init__(self, bot_manager, port: int = 8080):
        self.bot_manager = bot_manager
        self.port = port
        self.app = web.Application()
        self.routes = web.RouteTableDef()
        self.websocket_clients = []
        
        # Setup routes
        self._setup_routes()
        
        # Metrics cache
        self.metrics_cache = {}
        self.last_update = datetime.now()
    
    def _setup_routes(self):
        """Setup web routes"""
        
        @self.routes.get('/')
        async def index(request):
            """Serve dashboard HTML"""
            return web.Response(text=self._get_dashboard_html(), content_type='text/html')
        
        @self.routes.get('/api/status')
        async def get_status(request):
            """Get current bot status"""
            status = self.bot_manager.get_status() if self.bot_manager else {}
            return web.json_response(status)
        
        @self.routes.get('/api/health')
        async def get_health(request):
            """Get health metrics"""
            if self.bot_manager and self.bot_manager.health_monitor:
                health = self.bot_manager.health_monitor.get_health_report()
            else:
                health = {'status': 'unknown'}
            return web.json_response(health)
        
        @self.routes.get('/api/risk')
        async def get_risk(request):
            """Get risk metrics"""
            if self.bot_manager and self.bot_manager.risk_manager:
                risk = self.bot_manager.risk_manager.get_risk_summary()
            else:
                risk = {}
            return web.json_response(risk)
        
        @self.routes.get('/api/positions')
        async def get_positions(request):
            """Get current positions"""
            position = None
            if self.bot_manager and self.bot_manager.bot.current_position:
                pos = self.bot_manager.bot.current_position
                position = {
                    'symbol': pos.symbol,
                    'side': 'LONG' if pos.position_type == 1 else 'SHORT',
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'pattern': pos.pattern,
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else None
                }
            return web.json_response({'position': position})
        
        @self.routes.get('/api/performance')
        async def get_performance(request):
            """Get performance metrics"""
            if self.bot_manager:
                perf = {
                    'session_stats': self.bot_manager.session_stats,
                    'order_gate': self.bot_manager.order_gate.get_stats() if self.bot_manager.order_gate else {},
                    'atomic_orders': self.bot_manager.atomic_order_manager.get_metrics() if self.bot_manager.atomic_order_manager else {},
                    'position_tracker': self.bot_manager.position_tracker.get_metrics() if self.bot_manager.position_tracker else {}
                }
            else:
                perf = {}
            return web.json_response(perf)
        
        @self.routes.get('/api/errors')
        async def get_errors(request):
            """Get recent errors"""
            if self.bot_manager and self.bot_manager.error_recovery:
                errors = self.bot_manager.error_recovery.get_error_stats()
            else:
                errors = {}
            return web.json_response(errors)
        
        @self.routes.post('/api/control/pause')
        async def pause_trading(request):
            """Pause trading"""
            if self.bot_manager:
                self.bot_manager.bot.trading_paused = True
                return web.json_response({'status': 'paused'})
            return web.json_response({'error': 'Bot not initialized'}, status=400)
        
        @self.routes.post('/api/control/resume')
        async def resume_trading(request):
            """Resume trading"""
            if self.bot_manager:
                self.bot_manager.bot.trading_paused = False
                return web.json_response({'status': 'resumed'})
            return web.json_response({'error': 'Bot not initialized'}, status=400)
        
        @self.routes.post('/api/control/flatten')
        async def flatten_position(request):
            """Flatten all positions"""
            if self.bot_manager and self.bot_manager.atomic_order_manager:
                result = await self.bot_manager.atomic_order_manager.flatten_position("manual_dashboard")
                return web.json_response({
                    'status': result.state.value,
                    'message': result.rejection_reason if result.rejection_reason else 'Success'
                })
            return web.json_response({'error': 'Bot not initialized'}, status=400)
        
        @self.routes.get('/ws')
        async def websocket_handler(request):
            """WebSocket for real-time updates"""
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self.websocket_clients.append(ws)
            
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Handle incoming messages if needed
                        pass
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f'WebSocket error: {ws.exception()}')
            finally:
                self.websocket_clients.remove(ws)
            
            return ws
        
        # Add routes to app
        self.app.router.add_routes(self.routes)
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Monitor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .card h2 {
            margin-top: 0;
            color: #4CAF50;
            font-size: 18px;
            border-bottom: 1px solid #444;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid #333;
        }
        .metric-label {
            color: #999;
        }
        .metric-value {
            font-weight: bold;
        }
        .status-healthy { color: #4CAF50; }
        .status-degraded { color: #FFC107; }
        .status-critical { color: #F44336; }
        .position-long { color: #4CAF50; }
        .position-short { color: #F44336; }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: #45a049;
        }
        button.danger {
            background: #F44336;
        }
        button.danger:hover {
            background: #da190b;
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .alert {
            background: #F44336;
            color: white;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #4CAF50;
            margin: 0;
        }
        .timestamp {
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Trading Bot Monitor</h1>
        <div class="timestamp" id="lastUpdate">Last Update: Never</div>
    </div>
    
    <div class="dashboard">
        <!-- Status Card -->
        <div class="card">
            <h2>System Status</h2>
            <div class="metric">
                <span class="metric-label">Health:</span>
                <span class="metric-value" id="health-status">Unknown</span>
            </div>
            <div class="metric">
                <span class="metric-label">Bot State:</span>
                <span class="metric-value" id="bot-state">Unknown</span>
            </div>
            <div class="metric">
                <span class="metric-label">Trading:</span>
                <span class="metric-value" id="trading-status">Unknown</span>
            </div>
            <div class="metric">
                <span class="metric-label">Risk Level:</span>
                <span class="metric-value" id="risk-level">Unknown</span>
            </div>
        </div>
        
        <!-- Position Card -->
        <div class="card">
            <h2>Current Position</h2>
            <div id="position-info">
                <div class="metric">
                    <span class="metric-label">No Position</span>
                </div>
            </div>
        </div>
        
        <!-- Performance Card -->
        <div class="card">
            <h2>Performance</h2>
            <div class="metric">
                <span class="metric-label">Daily P&L:</span>
                <span class="metric-value" id="daily-pnl">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">Trades Today:</span>
                <span class="metric-value" id="trades-today">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Win Rate:</span>
                <span class="metric-value" id="win-rate">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Fill Rate:</span>
                <span class="metric-value" id="fill-rate">0%</span>
            </div>
        </div>
        
        <!-- Risk Management Card -->
        <div class="card">
            <h2>Risk Management</h2>
            <div class="metric">
                <span class="metric-label">Daily Loss Limit:</span>
                <span class="metric-value" id="daily-loss-limit">$0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Trades Remaining:</span>
                <span class="metric-value" id="trades-remaining">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Consecutive Losses:</span>
                <span class="metric-value" id="consecutive-losses">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Circuit Breaker:</span>
                <span class="metric-value" id="circuit-breaker">Inactive</span>
            </div>
        </div>
        
        <!-- System Health Card -->
        <div class="card">
            <h2>System Health</h2>
            <div class="metric">
                <span class="metric-label">CPU Usage:</span>
                <span class="metric-value" id="cpu-usage">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Memory Usage:</span>
                <span class="metric-value" id="memory-usage">0%</span>
            </div>
            <div class="metric">
                <span class="metric-label">API Status:</span>
                <span class="metric-value" id="api-status">Unknown</span>
            </div>
            <div class="metric">
                <span class="metric-label">Uptime:</span>
                <span class="metric-value" id="uptime">0h</span>
            </div>
        </div>
        
        <!-- Controls Card -->
        <div class="card">
            <h2>Controls</h2>
            <div class="controls">
                <button id="btn-pause" onclick="pauseTrading()">⏸️ Pause</button>
                <button id="btn-resume" onclick="resumeTrading()" disabled>▶️ Resume</button>
                <button class="danger" onclick="flattenPosition()">🚨 Flatten</button>
            </div>
            <div id="control-message" style="margin-top: 10px;"></div>
        </div>
        
        <!-- Alerts Card -->
        <div class="card" style="grid-column: span 2;">
            <h2>Active Alerts</h2>
            <div id="alerts-container">
                <div class="metric">
                    <span class="metric-label">No active alerts</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let updateInterval = null;
        
        // Initialize WebSocket
        function initWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(initWebSocket, 5000);
            };
        }
        
        // Fetch and update all data
        async function fetchData() {
            try {
                const [status, health, risk, positions, performance] = await Promise.all([
                    fetch('/api/status').then(r => r.json()),
                    fetch('/api/health').then(r => r.json()),
                    fetch('/api/risk').then(r => r.json()),
                    fetch('/api/positions').then(r => r.json()),
                    fetch('/api/performance').then(r => r.json())
                ]);
                
                updateStatus(status);
                updateHealth(health);
                updateRisk(risk);
                updatePosition(positions);
                updatePerformance(performance);
                
                document.getElementById('lastUpdate').textContent = 
                    `Last Update: ${new Date().toLocaleTimeString()}`;
                    
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        // Update status display
        function updateStatus(data) {
            const setState = (id, value, colorClass) => {
                const elem = document.getElementById(id);
                if (elem) {
                    elem.textContent = value;
                    elem.className = 'metric-value ' + (colorClass || '');
                }
            };
            
            setState('bot-state', data.bot_state || 'Unknown');
            setState('trading-status', data.running ? 'Active' : 'Stopped',
                     data.running ? 'status-healthy' : 'status-critical');
        }
        
        // Update health display
        function updateHealth(data) {
            const healthElem = document.getElementById('health-status');
            if (healthElem) {
                healthElem.textContent = data.status || 'Unknown';
                healthElem.className = 'metric-value status-' + (data.status || 'unknown').toLowerCase();
            }
            
            // Update system metrics
            if (data.metrics) {
                setMetric('cpu-usage', data.metrics.cpu_usage?.message);
                setMetric('memory-usage', data.metrics.memory_usage?.message);
                setMetric('api-status', data.metrics.api_connectivity?.message);
            }
            
            if (data.uptime_hours) {
                setMetric('uptime', `${data.uptime_hours.toFixed(1)}h`);
            }
            
            // Update alerts
            if (data.alerts && data.alerts.length > 0) {
                const alertsContainer = document.getElementById('alerts-container');
                alertsContainer.innerHTML = data.alerts.map(alert => 
                    `<div class="alert">${alert}</div>`
                ).join('');
            }
        }
        
        // Update risk display
        function updateRisk(data) {
            if (data.risk_level) {
                const elem = document.getElementById('risk-level');
                elem.textContent = data.risk_level;
                elem.className = 'metric-value ' + 
                    (data.risk_level === 'critical' ? 'status-critical' : 
                     data.risk_level === 'high' || data.risk_level === 'elevated' ? 'status-degraded' : 
                     'status-healthy');
            }
            
            if (data.metrics) {
                setMetric('daily-pnl', `$${data.metrics.daily_pnl?.toFixed(2) || '0.00'}`,
                         data.metrics.daily_pnl >= 0 ? 'status-healthy' : 'status-critical');
                setMetric('consecutive-losses', data.metrics.consecutive_losses || 0);
            }
            
            if (data.limits) {
                setMetric('daily-loss-limit', `$${data.limits.max_daily_loss?.toFixed(2) || '0'}`);
                setMetric('trades-remaining', data.limits.trades_remaining || 0);
            }
            
            if (data.circuit_breaker) {
                setMetric('circuit-breaker', 
                         data.circuit_breaker.active ? 'ACTIVE' : 'Inactive',
                         data.circuit_breaker.active ? 'status-critical' : '');
            }
        }
        
        // Update position display
        function updatePosition(data) {
            const container = document.getElementById('position-info');
            
            if (data.position) {
                const pos = data.position;
                container.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Side:</span>
                        <span class="metric-value position-${pos.side.toLowerCase()}">${pos.side}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Size:</span>
                        <span class="metric-value">${pos.size}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Entry:</span>
                        <span class="metric-value">$${pos.entry_price.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Stop Loss:</span>
                        <span class="metric-value">$${pos.stop_loss.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Take Profit:</span>
                        <span class="metric-value">$${pos.take_profit.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Pattern:</span>
                        <span class="metric-value">${pos.pattern || 'N/A'}</span>
                    </div>
                `;
            } else {
                container.innerHTML = '<div class="metric"><span class="metric-label">No Position</span></div>';
            }
        }
        
        // Update performance display
        function updatePerformance(data) {
            if (data.session_stats) {
                setMetric('trades-today', data.session_stats.trades_executed || 0);
            }
            
            if (data.atomic_orders) {
                const fillRate = data.atomic_orders.fill_rate || 0;
                setMetric('fill-rate', `${fillRate.toFixed(1)}%`);
            }
            
            // Calculate win rate from risk data
            if (data.risk && data.risk.metrics) {
                const wins = data.risk.metrics.winning_trades || 0;
                const losses = data.risk.metrics.losing_trades || 0;
                const total = wins + losses;
                const winRate = total > 0 ? (wins / total * 100) : 0;
                setMetric('win-rate', `${winRate.toFixed(1)}%`);
            }
        }
        
        // Helper function to set metric value
        function setMetric(id, value, className) {
            const elem = document.getElementById(id);
            if (elem) {
                elem.textContent = value;
                if (className) {
                    elem.className = 'metric-value ' + className;
                }
            }
        }
        
        // Control functions
        async function pauseTrading() {
            try {
                const response = await fetch('/api/control/pause', { method: 'POST' });
                const data = await response.json();
                
                document.getElementById('btn-pause').disabled = true;
                document.getElementById('btn-resume').disabled = false;
                document.getElementById('control-message').textContent = 'Trading paused';
                
                fetchData();
            } catch (error) {
                console.error('Error pausing:', error);
            }
        }
        
        async function resumeTrading() {
            try {
                const response = await fetch('/api/control/resume', { method: 'POST' });
                const data = await response.json();
                
                document.getElementById('btn-pause').disabled = false;
                document.getElementById('btn-resume').disabled = true;
                document.getElementById('control-message').textContent = 'Trading resumed';
                
                fetchData();
            } catch (error) {
                console.error('Error resuming:', error);
            }
        }
        
        async function flattenPosition() {
            if (!confirm('Are you sure you want to flatten all positions?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/control/flatten', { method: 'POST' });
                const data = await response.json();
                
                document.getElementById('control-message').textContent = 
                    `Flatten ${data.status}: ${data.message || ''}`;
                
                fetchData();
            } catch (error) {
                console.error('Error flattening:', error);
            }
        }
        
        // Initialize on load
        window.onload = () => {
            initWebSocket();
            fetchData();
            
            // Update every 2 seconds
            updateInterval = setInterval(fetchData, 2000);
        };
        
        // Cleanup on unload
        window.onbeforeunload = () => {
            if (ws) ws.close();
            if (updateInterval) clearInterval(updateInterval);
        };
    </script>
</body>
</html>
        '''
    
    async def start(self):
        """Start the monitoring dashboard"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"📊 Monitoring dashboard started at http://localhost:{self.port}")
        
        # Start metrics broadcast loop
        asyncio.create_task(self._broadcast_metrics())
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to connected WebSocket clients"""
        while True:
            try:
                await asyncio.sleep(2)  # Update every 2 seconds
                
                if not self.websocket_clients:
                    continue
                
                # Gather metrics
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'status': self.bot_manager.get_status() if self.bot_manager else {},
                    'health': self.bot_manager.health_monitor.get_health_report() if self.bot_manager and self.bot_manager.health_monitor else {},
                    'risk': self.bot_manager.risk_manager.get_risk_summary() if self.bot_manager and self.bot_manager.risk_manager else {}
                }
                
                # Broadcast to all clients
                disconnected = []
                for ws in self.websocket_clients:
                    try:
                        await ws.send_str(json.dumps(metrics))
                    except ConnectionResetError:
                        disconnected.append(ws)
                
                # Remove disconnected clients
                for ws in disconnected:
                    self.websocket_clients.remove(ws)
                    
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

# Standalone dashboard runner
async def run_dashboard(bot_manager=None, port=8080):
    """Run dashboard standalone"""
    dashboard = MonitoringDashboard(bot_manager, port)
    await dashboard.start()
    
    # Keep running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    # Can run dashboard standalone for testing
    asyncio.run(run_dashboard())