// WebSocket connection
let ws;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 5000;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeWebSocket();
    initializeCharts();
    setupEventListeners();
    loadInitialData();
});

// WebSocket handling
function initializeWebSocket() {
    ws = new WebSocket('ws://127.0.0.1:5678');
    
    ws.onopen = function() {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        document.getElementById('connectionStatus').className = 'badge bg-success';
        document.getElementById('connectionStatus').textContent = 'Connected';
    };
    
    ws.onclose = function() {
        console.log('WebSocket disconnected');
        document.getElementById('connectionStatus').className = 'badge bg-danger';
        document.getElementById('connectionStatus').textContent = 'Disconnected';
        
        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
            setTimeout(function() {
                reconnectAttempts++;
                initializeWebSocket();
            }, reconnectDelay);
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
}

// Initialize charts
function initializeCharts() {
    // Performance chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    window.performanceChart = new Chart(performanceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Account Balance',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Account Performance'
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    // Bot control buttons
    document.getElementById('startBot').addEventListener('click', function() {
        if (!confirm('Are you sure you want to start the trading bot?')) return;
        
        fetch('/api/bot/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert('success', 'Bot started successfully');
                updateBotStatus('Running');
            } else {
                showAlert('danger', data.error || 'Failed to start bot');
            }
        })
        .catch(error => {
            showAlert('danger', 'Error starting bot');
            console.error('Error:', error);
        });
    });
    
    document.getElementById('stopBot').addEventListener('click', function() {
        if (!confirm('Are you sure you want to stop the trading bot?')) return;
        
        fetch('/api/bot/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert('success', 'Bot stopped successfully');
                updateBotStatus('Stopped');
            } else {
                showAlert('danger', data.error || 'Failed to stop bot');
            }
        })
        .catch(error => {
            showAlert('danger', 'Error stopping bot');
            console.error('Error:', error);
        });
    });
    
    // Risk management form
    document.querySelector('form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        fetch('/api/settings/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(Object.fromEntries(formData))
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert('success', 'Settings updated successfully');
            } else {
                showAlert('danger', data.error || 'Failed to update settings');
            }
        })
        .catch(error => {
            showAlert('danger', 'Error updating settings');
            console.error('Error:', error);
        });
    });
}

// Load initial data
function loadInitialData() {
    fetch('/api/dashboard/data')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateDashboard(data.data);
            }
        })
        .catch(error => {
            console.error('Error loading initial data:', error);
            showAlert('danger', 'Error loading dashboard data');
        });
}

// Update dashboard with new data
function updateDashboard(data) {
    // Update account info
    document.getElementById('accountBalance').textContent = formatCurrency(data.balance);
    document.getElementById('equity').textContent = formatCurrency(data.equity);
    document.getElementById('openPositions').textContent = data.open_positions;
    document.getElementById('dailyProfit').textContent = formatCurrency(data.daily_profit);
    
    // Update price cards
    updatePriceCard('EURUSD', data.prices.EURUSD);
    updatePriceCard('GBPUSD', data.prices.GBPUSD);
    updatePriceCard('USDJPY', data.prices.USDJPY);
    updatePriceCard('AUDUSD', data.prices.AUDUSD);
    
    // Update performance chart
    updatePerformanceChart(data.equity_history);
    
    // Update positions table
    updatePositionsTable(data.positions);
}

// Helper functions
function updateBotStatus(status) {
    const statusElement = document.getElementById('botStatus');
    statusElement.textContent = status;
    statusElement.className = `badge bg-${status === 'Running' ? 'success' : 'danger'}`;
    
    document.getElementById('startBot').disabled = status === 'Running';
    document.getElementById('stopBot').disabled = status === 'Stopped';
}

function updatePriceCard(symbol, price) {
    const card = document.getElementById(`${symbol}Card`);
    if (!card) return;
    
    const priceElement = card.querySelector('.price');
    const oldPrice = parseFloat(priceElement.textContent);
    const newPrice = parseFloat(price);
    
    priceElement.textContent = price;
    priceElement.className = `price mb-0 ${newPrice > oldPrice ? 'text-success' : newPrice < oldPrice ? 'text-danger' : ''}`;
}

function updatePerformanceChart(data) {
    const chart = window.performanceChart;
    chart.data.labels = data.map(point => new Date(point.timestamp).toLocaleDateString());
    chart.data.datasets[0].data = data.map(point => point.equity);
    chart.update();
}

function updatePositionsTable(positions) {
    const tbody = document.getElementById('positionsTable');
    tbody.innerHTML = '';
    
    positions.forEach(position => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${position.symbol}</td>
            <td>${position.type}</td>
            <td>${position.volume}</td>
            <td>${formatPrice(position.entry_price)}</td>
            <td>${formatPrice(position.current_price)}</td>
            <td class="${position.profit >= 0 ? 'text-success' : 'text-danger'}">
                ${formatCurrency(position.profit)}
            </td>
            <td>
                <button class="btn btn-danger btn-sm" onclick="closePosition('${position.ticket}')">
                    Close
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function closePosition(ticket) {
    if (!confirm('Are you sure you want to close this position?')) return;
    
    fetch('/api/position/close', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ticket })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('success', 'Position closed successfully');
        } else {
            showAlert('danger', data.error || 'Failed to close position');
        }
    })
    .catch(error => {
        showAlert('danger', 'Error closing position');
        console.error('Error:', error);
    });
}

function showAlert(type, message) {
    const alertsContainer = document.getElementById('alertsContainer');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertsContainer.appendChild(alert);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPrice(value) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 5,
        maximumFractionDigits: 5
    }).format(value);
}
