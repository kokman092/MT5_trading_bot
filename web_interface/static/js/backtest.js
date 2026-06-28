document.addEventListener('DOMContentLoaded', function() {
    // Initialize date inputs
    const today = new Date();
    const oneMonthAgo = new Date();
    oneMonthAgo.setMonth(today.getMonth() - 1);
    
    document.querySelector('input[name="start_date"]').value = oneMonthAgo.toISOString().split('T')[0];
    document.querySelector('input[name="end_date"]').value = today.toISOString().split('T')[0];
    
    // Initialize chart
    const ctx = document.getElementById('equityChart').getContext('2d');
    let equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
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
                    text: 'Equity Curve'
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    // Handle form submission
    document.getElementById('backtestForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const params = Object.fromEntries(formData.entries());
        
        // Show loading state
        document.getElementById('runBacktest').disabled = true;
        document.getElementById('runBacktest').innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Running...';
        
        try {
            const response = await fetch('/api/backtest/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });
            
            const results = await response.json();
            
            if (results.success) {
                updateResults(results.data);
            } else {
                showError(results.error);
            }
        } catch (error) {
            showError('Failed to run backtest');
        } finally {
            // Reset button state
            document.getElementById('runBacktest').disabled = false;
            document.getElementById('runBacktest').innerHTML = '<i class="fas fa-play me-2"></i>Run Backtest';
        }
    });
    
    function updateResults(data) {
        // Update summary metrics
        document.getElementById('totalTrades').textContent = data.metrics.total_trades;
        document.getElementById('winRate').textContent = data.metrics.win_rate.toFixed(2) + '%';
        document.getElementById('totalProfit').textContent = '$' + data.metrics.total_profit.toFixed(2);
        document.getElementById('returnPct').textContent = data.metrics.return_pct.toFixed(2) + '%';
        document.getElementById('profitFactor').textContent = data.metrics.profit_factor.toFixed(2);
        document.getElementById('maxDrawdown').textContent = data.metrics.max_drawdown.toFixed(2) + '%';
        
        // Update equity chart
        const dates = data.equity_curve.map(point => new Date(point.datetime).toLocaleDateString());
        const equity = data.equity_curve.map(point => point.equity);
        
        equityChart.data.labels = dates;
        equityChart.data.datasets[0].data = equity;
        equityChart.update();
        
        // Update trades table
        const tradesTable = document.getElementById('tradesTable');
        tradesTable.innerHTML = '';
        
        data.trades.forEach(trade => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${new Date(trade.entry_time).toLocaleString()}</td>
                <td>${new Date(trade.exit_time).toLocaleString()}</td>
                <td>${trade.type}</td>
                <td>${trade.entry_price.toFixed(5)}</td>
                <td>${trade.exit_price.toFixed(5)}</td>
                <td>${trade.size.toFixed(2)}</td>
                <td class="${trade.profit > 0 ? 'text-success' : 'text-danger'}">
                    ${trade.profit > 0 ? '+' : ''}${trade.profit.toFixed(2)}
                </td>
            `;
            tradesTable.appendChild(row);
        });
    }
    
    function showError(message) {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to alerts container
        const container = document.querySelector('.container-fluid');
        container.insertBefore(alert, container.firstChild);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
});
