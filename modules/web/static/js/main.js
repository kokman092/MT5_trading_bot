// Dark Mode Toggle
document.addEventListener('DOMContentLoaded', () => {
    // Check for saved dark mode preference
    const darkMode = localStorage.getItem('darkMode') === 'true';
    if (darkMode) {
        document.body.classList.add('dark-mode');
    }
    
    // Add dark mode toggle to navbar if it exists
    const navbar = document.querySelector('.navbar-nav');
    if (navbar) {
        const darkModeToggle = document.createElement('li');
        darkModeToggle.className = 'nav-item';
        darkModeToggle.innerHTML = `
            <button class="nav-link btn btn-link" id="darkModeToggle">
                <i class="fas ${darkMode ? 'fa-sun' : 'fa-moon'}"></i>
            </button>
        `;
        navbar.appendChild(darkModeToggle);
        
        // Handle dark mode toggle
        document.getElementById('darkModeToggle').addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDarkMode = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDarkMode);
            darkModeToggle.querySelector('i').className = `fas ${isDarkMode ? 'fa-sun' : 'fa-moon'}`;
        });
    }
});

// WebSocket Connection
let socket = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };
    
    socket.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connectionStatus');
    if (statusElement) {
        statusElement.innerHTML = connected ? 
            '<i class="fas fa-circle text-success"></i> Connected' :
            '<i class="fas fa-circle text-danger"></i> Disconnected';
    }
}

// Trading Controls
let botStatusInterval = null;

function startTrading() {
    fetch('/api/trading/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('success', 'Trading started successfully');
            updateTradingStatus(true);
            startStatusPolling();
        } else {
            showAlert('danger', `Failed to start trading: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('danger', 'Failed to start trading');
    });
}

function stopTrading() {
    fetch('/api/trading/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('success', 'Trading stopped successfully');
            updateTradingStatus(false);
            stopStatusPolling();
        } else {
            showAlert('danger', `Failed to stop trading: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('danger', 'Failed to stop trading');
    });
}

function updateTradingStatus(isTrading) {
    const startButton = document.getElementById('startTradingBtn');
    const stopButton = document.getElementById('stopTradingBtn');
    const statusIndicator = document.getElementById('tradingStatus');
    
    if (startButton && stopButton) {
        startButton.disabled = isTrading;
        stopButton.disabled = !isTrading;
    }
    
    if (statusIndicator) {
        statusIndicator.innerHTML = isTrading ? 
            '<i class="fas fa-circle text-success"></i> Trading Active' :
            '<i class="fas fa-circle text-danger"></i> Trading Inactive';
    }
}

function startStatusPolling() {
    // Stop any existing polling
    stopStatusPolling();
    
    // Start polling every 5 seconds
    botStatusInterval = setInterval(checkBotStatus, 5000);
}

function stopStatusPolling() {
    if (botStatusInterval) {
        clearInterval(botStatusInterval);
        botStatusInterval = null;
    }
}

function checkBotStatus() {
    fetch('/api/trading/status')
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                updateTradingStatus(data.running);
                
                // If bot is not running but we thought it was, stop polling
                if (!data.running && botStatusInterval) {
                    stopStatusPolling();
                }
            }
        })
        .catch(error => {
            console.error('Error checking bot status:', error);
        });
}

// Check initial bot status when page loads
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('tradingDashboard')) {
        checkBotStatus();
    }
});

// Dashboard Updates
function updateDashboard(data) {
    // Update account information
    if (data.account) {
        updateAccountInfo(data.account);
    }
    
    // Update open positions
    if (data.positions) {
        updatePositionsTable(data.positions);
    }
    
    // Update daily P/L
    if (data.daily_pl) {
        updateDailyPL(data.daily_pl);
    }
    
    // Update charts if they exist
    if (data.chart_data) {
        updateCharts(data.chart_data);
    }
}

function updateAccountInfo(account) {
    const elements = {
        balance: document.getElementById('accountBalance'),
        equity: document.getElementById('accountEquity'),
        margin: document.getElementById('usedMargin'),
        freeMargin: document.getElementById('freeMargin')
    };
    
    for (const [key, element] of Object.entries(elements)) {
        if (element && account[key] !== undefined) {
            element.textContent = formatCurrency(account[key]);
        }
    }
}

function updatePositionsTable(positions) {
    const table = document.getElementById('positionsTable');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = '';
    
    positions.forEach(position => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${position.symbol}</td>
            <td>${position.type}</td>
            <td>${position.volume}</td>
            <td>${formatCurrency(position.entry_price)}</td>
            <td>${formatCurrency(position.current_price)}</td>
            <td class="${position.profit >= 0 ? 'text-success' : 'text-danger'}">
                ${formatCurrency(position.profit)}
            </td>
            <td>
                <button class="btn btn-sm btn-danger" onclick="closePosition('${position.ticket}')">
                    Close
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function updateDailyPL(dailyPL) {
    const element = document.getElementById('dailyPL');
    if (element) {
        element.textContent = formatCurrency(dailyPL);
        element.className = dailyPL >= 0 ? 'text-success' : 'text-danger';
    }
}

// Utility Functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function showAlert(type, message) {
    const alertsContainer = document.querySelector('.alerts-container');
    if (!alertsContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 150);
    }, 5000);
}

// Form Validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return true;
    
    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.classList.add('is-invalid');
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// Initialize WebSocket connection when on dashboard
if (document.getElementById('tradingDashboard')) {
    connectWebSocket();
}

// Initialize tooltips and popovers
document.addEventListener('DOMContentLoaded', () => {
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
    
    const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
    popovers.forEach(popover => new bootstrap.Popover(popover));
}); 