from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import MetaTrader5 as mt5
from ..analytics.market_analyzer import MarketAnalyzer
from ..risk.risk_manager import RiskManager
from ..deployment.error_handler import ErrorHandler

class DQNModel(nn.Module):
    """Deep Q-Network for strategy selection"""
    def __init__(self, input_size: int, output_size: int):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class AdaptiveStrategyManager:
    def __init__(self, config: Dict):
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.logger = logging.getLogger('adaptive_strategy_manager')
        self.market_analyzer = MarketAnalyzer(config)
        self.risk_manager = RiskManager(config)
        
        # Initialize strategies and metrics
        self._init_strategies()
        
        # Initialize RL components
        self._init_rl_components()
        
    def _init_strategies(self):
        """Initialize available strategies and their metrics"""
        self.strategies = {
            'trend_following': {
                'active': True,
                'metrics': {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                },
                'performance_history': []
            },
            'mean_reversion': {
                'active': True,
                'metrics': {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                },
                'performance_history': []
            },
            'breakout': {
                'active': True,
                'metrics': {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                },
                'performance_history': []
            },
            'momentum': {
                'active': True,
                'metrics': {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                },
                'performance_history': []
            }
        }
        
        # Strategy switching thresholds
        self.switch_thresholds = {
            'win_rate_min': 0.4,
            'profit_factor_min': 1.2,
            'sharpe_ratio_min': 0.5,
            'max_drawdown_max': 0.2
        }
        
    def _init_rl_components(self):
        """Initialize reinforcement learning components"""
        # State and action spaces
        self.state_size = 12  # Market features + strategy metrics
        self.action_size = len(self.strategies)
        
        # DQN model
        self.model = DQNModel(self.state_size, self.action_size)
        self.target_model = DQNModel(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.training_step = 0
        
    async def select_strategy(self, market_data: Dict) -> str:
        """Select best strategy based on current market conditions"""
        try:
            # Get market state
            market_state = await self._get_market_state(market_data)
            
            # Use RL model for strategy selection
            if random.random() > self.epsilon:
                state_tensor = torch.FloatTensor(market_state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                action = q_values.argmax().item()
            else:
                action = random.randrange(self.action_size)
                
            # Get strategy name
            strategy_name = list(self.strategies.keys())[action]
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return strategy_name
            
        except Exception as e:
            self.logger.error(f"Strategy selection error: {str(e)}")
            return list(self.strategies.keys())[0]  # Return default strategy
            
    async def update_strategy_metrics(
        self,
        strategy_name: str,
        trade_result: Dict
    ) -> None:
        """Update strategy performance metrics"""
        try:
            strategy = self.strategies[strategy_name]
            
            # Update metrics
            strategy['performance_history'].append(trade_result)
            
            # Calculate new metrics
            if len(strategy['performance_history']) > 0:
                trades = strategy['performance_history'][-100:]  # Last 100 trades
                
                # Win rate
                wins = sum(1 for t in trades if t['profit'] > 0)
                strategy['metrics']['win_rate'] = wins / len(trades)
                
                # Profit factor
                gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
                gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
                strategy['metrics']['profit_factor'] = (
                    gross_profit / gross_loss if gross_loss > 0 else float('inf')
                )
                
                # Sharpe ratio
                returns = [t['profit'] for t in trades]
                if len(returns) > 1:
                    strategy['metrics']['sharpe_ratio'] = (
                        np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                    )
                    
                # Max drawdown
                cumulative = np.cumsum(returns)
                max_dd = 0
                peak = cumulative[0]
                for value in cumulative:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                strategy['metrics']['max_drawdown'] = max_dd
                
            # Store reward for RL
            reward = self._calculate_reward(trade_result)
            
            # Store experience in memory
            if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
                self.memory.append(
                    (self.last_state, self.last_action, reward, market_state, False)
                )
                
            # Train model
            await self._train_model()
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {str(e)}")
            
    async def _get_market_state(self, market_data: Dict) -> List[float]:
        """Get current market state features"""
        try:
            state = []
            
            # Market regime features
            regime = await self.market_analyzer.detect_market_regime(
                market_data['symbol']
            )
            state.extend([
                regime['characteristics']['trend']['strength'],
                regime['characteristics']['volatility']['ratio'],
                1 if regime['regime'] == 'trending_up' else 0,
                1 if regime['regime'] == 'trending_down' else 0,
                1 if regime['regime'] == 'ranging' else 0
            ])
            
            # Volatility features
            volatility = await self.market_analyzer.forecast_volatility(
                market_data['symbol']
            )
            state.extend([
                volatility['current_volatility'],
                volatility['forecast_volatility']
            ])
            
            # Strategy performance features
            for strategy in self.strategies.values():
                state.extend([
                    strategy['metrics']['win_rate'],
                    strategy['metrics']['profit_factor'],
                    strategy['metrics']['sharpe_ratio'],
                    strategy['metrics']['max_drawdown']
                ])
                
            return state
            
        except Exception as e:
            self.logger.error(f"Market state calculation error: {str(e)}")
            return [0] * self.state_size
            
    def _calculate_reward(self, trade_result: Dict) -> float:
        """Calculate reward for reinforcement learning"""
        try:
            reward = 0
            
            # Profit/loss reward
            if trade_result['profit'] > 0:
                reward += 1
            else:
                reward -= 1
                
            # Stop-loss penalty
            if trade_result.get('stop_loss_hit', False):
                reward -= 0.5
                
            # Risk-adjusted reward
            if trade_result.get('risk_reward_ratio', 0) > 0:
                reward *= trade_result['risk_reward_ratio']
                
            return reward
            
        except Exception as e:
            self.logger.error(f"Reward calculation error: {str(e)}")
            return 0
            
    async def _train_model(self):
        """Train the DQN model"""
        try:
            if len(self.memory) < self.batch_size:
                return
                
            # Sample batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            # Current Q values
            current_q = self.model(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Calculate loss and update
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network
            self.training_step += 1
            if self.training_step % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            
    async def validate_strategy_switch(
        self,
        current_strategy: str,
        new_strategy: str
    ) -> bool:
        """Validate if strategy switch is appropriate"""
        try:
            current = self.strategies[current_strategy]
            new = self.strategies[new_strategy]
            
            # Check if new strategy meets minimum thresholds
            if (new['metrics']['win_rate'] < self.switch_thresholds['win_rate_min'] or
                new['metrics']['profit_factor'] < self.switch_thresholds['profit_factor_min'] or
                new['metrics']['sharpe_ratio'] < self.switch_thresholds['sharpe_ratio_min'] or
                new['metrics']['max_drawdown'] > self.switch_thresholds['max_drawdown_max']):
                return False
                
            # Compare performance
            if (new['metrics']['sharpe_ratio'] > current['metrics']['sharpe_ratio'] and
                new['metrics']['win_rate'] > current['metrics']['win_rate']):
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Strategy switch validation error: {str(e)}")
            return False
            
    async def save_model(self, path: str):
        """Save the DQN model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step
            }, path)
            
        except Exception as e:
            self.logger.error(f"Model save error: {str(e)}")
            
    async def load_model(self, path: str):
        """Load the DQN model"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            
        except Exception as e:
            self.logger.error(f"Model load error: {str(e)}")
