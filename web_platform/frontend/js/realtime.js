/**
 * Real-Time Update Handler for TopStepX Integration
 * Handles WebSocket messages for order, position, trade, and quote updates
 */

class RealtimeUpdateHandler {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        
        // Callbacks for different update types
        this.callbacks = {
            order: [],
            position: [],
            trade: [],
            quote: [],
            status: []
        };
        
        // Cache for latest data
        this.latestData = {
            orders: {},
            positions: {},
            quotes: {},
            trades: []
        };
        
        // UI elements
        this.elements = {
            orderNotifications: null,
            positionDisplay: null,
            priceDisplay: null,
            tradeHistory: null
        };
    }
    
    /**
     * Initialize the real-time handler
     */
    init() {
        this.connectWebSocket();
        this.setupUIElements();
        this.registerDefaultHandlers();
    }
    
    /**
     * Connect to WebSocket
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ Real-time connection established');
                this.reconnectAttempts = 0;
                this.showNotification('Connected to real-time feed', 'success');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Failed to parse message:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('Real-time connection error', 'error');
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.handleReconnect();
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.handleReconnect();
        }
    }
    
    /**
     * Handle reconnection logic
     */
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectInterval * Math.min(this.reconnectAttempts, 3);
            
            console.log(`Reconnecting in ${delay/1000} seconds... (Attempt ${this.reconnectAttempts})`);
            this.showNotification(`Reconnecting... (Attempt ${this.reconnectAttempts})`, 'warning');
            
            setTimeout(() => {
                this.connectWebSocket();
            }, delay);
        } else {
            this.showNotification('Unable to connect to real-time feed', 'error');
        }
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(message) {
        const { type, data, timestamp } = message;
        
        switch (type) {
            case 'order_update':
                this.handleOrderUpdate(data, timestamp);
                break;
                
            case 'position_update':
                this.handlePositionUpdate(data, timestamp);
                break;
                
            case 'trade_update':
                this.handleTradeUpdate(data, timestamp);
                break;
                
            case 'quote_update':
                this.handleQuoteUpdate(data, timestamp);
                break;
                
            case 'status_update':
                this.handleStatusUpdate(data, timestamp);
                break;
                
            default:
                console.log(`Unknown message type: ${type}`, data);
        }
        
        // Call registered callbacks
        const callbacks = this.callbacks[type.replace('_update', '')] || [];
        callbacks.forEach(callback => callback(data, timestamp));
    }
    
    /**
     * Handle order updates
     */
    handleOrderUpdate(data, timestamp) {
        const orderId = data.id;
        const status = this.getOrderStatus(data.status);
        
        // Update cache
        if (status === 'Open' || status === 'Pending') {
            this.latestData.orders[orderId] = data;
        } else {
            delete this.latestData.orders[orderId];
        }
        
        // Show notification for important status changes
        if (status === 'Filled') {
            const side = data.side === 0 ? 'BUY' : 'SELL';
            const price = data.filledPrice || data.limitPrice;
            this.showNotification(
                `✅ Order FILLED: ${side} ${data.size} @ $${price}`,
                'success',
                true
            );
            this.playSound('fill');
        } else if (status === 'Rejected') {
            this.showNotification(
                `❌ Order REJECTED: ${data.customTag || 'Unknown reason'}`,
                'error',
                true
            );
            this.playSound('error');
        }
        
        // Update UI
        this.updateOrderDisplay();
    }
    
    /**
     * Handle position updates
     */
    handlePositionUpdate(data, timestamp) {
        const contractId = data.contractId;
        const size = data.size;
        
        // Update cache
        if (size > 0) {
            this.latestData.positions[contractId] = data;
        } else {
            delete this.latestData.positions[contractId];
        }
        
        // Update position display
        this.updatePositionDisplay(data);
        
        // Show P&L if available
        if (data.unrealizedPnl !== undefined) {
            const pnlClass = data.unrealizedPnl >= 0 ? 'profit' : 'loss';
            const pnlSign = data.unrealizedPnl >= 0 ? '+' : '';
            
            if (this.elements.positionDisplay) {
                const pnlElement = this.elements.positionDisplay.querySelector('.unrealized-pnl');
                if (pnlElement) {
                    pnlElement.textContent = `${pnlSign}$${data.unrealizedPnl.toFixed(2)}`;
                    pnlElement.className = `unrealized-pnl ${pnlClass}`;
                }
            }
        }
    }
    
    /**
     * Handle trade updates
     */
    handleTradeUpdate(data, timestamp) {
        // Add to trade history
        this.latestData.trades.unshift({
            ...data,
            timestamp
        });
        
        // Keep only last 100 trades
        if (this.latestData.trades.length > 100) {
            this.latestData.trades = this.latestData.trades.slice(0, 100);
        }
        
        // Show trade notification
        const side = data.side === 0 ? 'BUY' : 'SELL';
        const pnl = data.profitAndLoss;
        
        if (pnl !== null && pnl !== undefined) {
            // Closing trade with P&L
            const pnlClass = pnl >= 0 ? 'success' : 'error';
            const pnlSign = pnl >= 0 ? '+' : '';
            this.showNotification(
                `💰 Trade Closed: ${side} ${data.size} @ $${data.price} | P&L: ${pnlSign}$${pnl.toFixed(2)}`,
                pnlClass,
                true
            );
            
            // Update daily P&L
            this.updateDailyPnL(pnl);
            
            // Play sound based on P&L
            this.playSound(pnl >= 0 ? 'profit' : 'loss');
        } else {
            // Opening trade
            this.showNotification(
                `🎯 Trade Opened: ${side} ${data.size} @ $${data.price}`,
                'info'
            );
        }
        
        // Update trade history display
        this.updateTradeHistory();
    }
    
    /**
     * Handle quote updates
     */
    handleQuoteUpdate(data, timestamp) {
        const symbol = data.symbol;
        
        // Update cache
        this.latestData.quotes[symbol] = data;
        
        // Update price displays
        this.updatePriceDisplays(data);
        
        // Flash price on significant moves
        const change = data.change || 0;
        if (Math.abs(change) > 5) {
            this.flashPriceUpdate(change > 0 ? 'up' : 'down');
        }
    }
    
    /**
     * Handle status updates
     */
    handleStatusUpdate(data, timestamp) {
        // Update general status displays
        if (data.current_price) {
            this.updatePriceDisplays({
                lastPrice: data.current_price,
                change: data.price_change || 0
            });
        }
        
        if (data.daily_pnl !== undefined) {
            this.updateDailyPnL(data.daily_pnl);
        }
    }
    
    /**
     * Register a callback for specific update type
     */
    on(type, callback) {
        if (this.callbacks[type]) {
            this.callbacks[type].push(callback);
        }
    }
    
    /**
     * Setup UI elements
     */
    setupUIElements() {
        this.elements.orderNotifications = document.getElementById('order-notifications');
        this.elements.positionDisplay = document.getElementById('position-display');
        this.elements.priceDisplay = document.getElementById('current-price');
        this.elements.tradeHistory = document.getElementById('trade-history');
    }
    
    /**
     * Register default UI update handlers
     */
    registerDefaultHandlers() {
        // Update price card on quote updates
        this.on('quote', (data) => {
            const priceCard = document.querySelector('.price-card');
            if (priceCard) {
                const priceElement = priceCard.querySelector('.price-value');
                const changeElement = priceCard.querySelector('.price-change');
                
                if (priceElement) {
                    priceElement.textContent = `$${data.lastPrice.toFixed(2)}`;
                }
                
                if (changeElement) {
                    const changeClass = data.change >= 0 ? 'positive' : 'negative';
                    changeElement.className = `price-change ${changeClass}`;
                    changeElement.textContent = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)} (${data.changePercent.toFixed(2)}%)`;
                }
            }
        });
    }
    
    /**
     * Update order display
     */
    updateOrderDisplay() {
        const ordersContainer = document.getElementById('active-orders');
        if (!ordersContainer) return;
        
        const orders = Object.values(this.latestData.orders);
        
        if (orders.length === 0) {
            ordersContainer.innerHTML = '<p class="no-data">No active orders</p>';
            return;
        }
        
        const ordersHtml = orders.map(order => {
            const side = order.side === 0 ? 'BUY' : 'SELL';
            const status = this.getOrderStatus(order.status);
            const type = this.getOrderType(order.type);
            
            return `
                <div class="order-item">
                    <div class="order-header">
                        <span class="order-side ${side.toLowerCase()}">${side}</span>
                        <span class="order-status">${status}</span>
                    </div>
                    <div class="order-details">
                        <span>${order.size} contracts</span>
                        <span>${type} @ $${order.limitPrice || order.stopPrice || 'Market'}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        ordersContainer.innerHTML = ordersHtml;
    }
    
    /**
     * Update position display
     */
    updatePositionDisplay(position) {
        const positionCard = document.getElementById('position-card');
        if (!positionCard) return;
        
        if (!position || position.size === 0) {
            positionCard.style.display = 'none';
            return;
        }
        
        positionCard.style.display = 'block';
        
        const direction = position.type === 1 ? 'LONG' : 'SHORT';
        const directionClass = position.type === 1 ? 'long' : 'short';
        
        positionCard.innerHTML = `
            <h3>Active Position</h3>
            <div class="position-details">
                <div class="position-direction ${directionClass}">${direction}</div>
                <div class="position-info">
                    <div>Size: ${position.size} contracts</div>
                    <div>Avg Price: $${position.averagePrice.toFixed(2)}</div>
                    <div class="unrealized-pnl">P&L: Calculating...</div>
                </div>
            </div>
        `;
    }
    
    /**
     * Update trade history
     */
    updateTradeHistory() {
        const historyContainer = document.getElementById('trade-history-list');
        if (!historyContainer) return;
        
        const trades = this.latestData.trades.slice(0, 10);
        
        if (trades.length === 0) {
            historyContainer.innerHTML = '<p class="no-data">No recent trades</p>';
            return;
        }
        
        const tradesHtml = trades.map(trade => {
            const side = trade.side === 0 ? 'BUY' : 'SELL';
            const time = new Date(trade.timestamp).toLocaleTimeString();
            const pnl = trade.profitAndLoss;
            
            let pnlHtml = '';
            if (pnl !== null && pnl !== undefined) {
                const pnlClass = pnl >= 0 ? 'profit' : 'loss';
                const pnlSign = pnl >= 0 ? '+' : '';
                pnlHtml = `<span class="trade-pnl ${pnlClass}">${pnlSign}$${pnl.toFixed(2)}</span>`;
            }
            
            return `
                <div class="trade-item">
                    <span class="trade-time">${time}</span>
                    <span class="trade-side ${side.toLowerCase()}">${side}</span>
                    <span>${trade.size} @ $${trade.price.toFixed(2)}</span>
                    ${pnlHtml}
                </div>
            `;
        }).join('');
        
        historyContainer.innerHTML = tradesHtml;
    }
    
    /**
     * Update price displays
     */
    updatePriceDisplays(quote) {
        const priceElements = document.querySelectorAll('.current-price');
        priceElements.forEach(element => {
            element.textContent = `$${quote.lastPrice.toFixed(2)}`;
        });
        
        const bidElements = document.querySelectorAll('.bid-price');
        bidElements.forEach(element => {
            element.textContent = `$${quote.bestBid?.toFixed(2) || '---'}`;
        });
        
        const askElements = document.querySelectorAll('.ask-price');
        askElements.forEach(element => {
            element.textContent = `$${quote.bestAsk?.toFixed(2) || '---'}`;
        });
    }
    
    /**
     * Update daily P&L display
     */
    updateDailyPnL(pnl) {
        const pnlElements = document.querySelectorAll('.daily-pnl');
        pnlElements.forEach(element => {
            const pnlClass = pnl >= 0 ? 'profit' : 'loss';
            const pnlSign = pnl >= 0 ? '+' : '';
            element.textContent = `${pnlSign}$${pnl.toFixed(2)}`;
            element.className = `daily-pnl ${pnlClass}`;
        });
    }
    
    /**
     * Flash price update animation
     */
    flashPriceUpdate(direction) {
        const priceElements = document.querySelectorAll('.current-price');
        priceElements.forEach(element => {
            element.classList.add(`flash-${direction}`);
            setTimeout(() => {
                element.classList.remove(`flash-${direction}`);
            }, 500);
        });
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info', persist = false) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to notifications container
        let container = document.getElementById('notifications');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notifications';
            container.className = 'notifications-container';
            document.body.appendChild(container);
        }
        
        container.appendChild(notification);
        
        // Auto-remove after delay
        if (!persist) {
            setTimeout(() => {
                notification.classList.add('fade-out');
                setTimeout(() => {
                    notification.remove();
                }, 500);
            }, 5000);
        }
    }
    
    /**
     * Play sound effect
     */
    playSound(type) {
        // Only play if sounds are enabled
        if (!localStorage.getItem('soundsEnabled')) return;
        
        const sounds = {
            fill: '/sounds/fill.mp3',
            profit: '/sounds/profit.mp3',
            loss: '/sounds/loss.mp3',
            error: '/sounds/error.mp3'
        };
        
        const soundUrl = sounds[type];
        if (soundUrl) {
            const audio = new Audio(soundUrl);
            audio.volume = 0.5;
            audio.play().catch(e => console.log('Could not play sound:', e));
        }
    }
    
    /**
     * Get order status text
     */
    getOrderStatus(status) {
        const statusMap = {
            0: 'None',
            1: 'Open',
            2: 'Filled',
            3: 'Cancelled',
            4: 'Expired',
            5: 'Rejected',
            6: 'Pending'
        };
        return statusMap[status] || 'Unknown';
    }
    
    /**
     * Get order type text
     */
    getOrderType(type) {
        const typeMap = {
            1: 'Limit',
            2: 'Market',
            3: 'Stop Limit',
            4: 'Stop',
            5: 'Trailing Stop',
            6: 'Join Bid',
            7: 'Join Ask'
        };
        return typeMap[type] || 'Unknown';
    }
}

// Initialize on page load
const realtimeHandler = new RealtimeUpdateHandler();
document.addEventListener('DOMContentLoaded', () => {
    realtimeHandler.init();
});

// Export for use in other modules
window.RealtimeHandler = realtimeHandler;