import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
import gymnasium as gym

class TradingEnvironment(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # Continuous action space for position size
        
        # State space: OHLCV + technical indicators + account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        return self._get_observation(), {}
        
    def step(self, action):
        # Execute action
        self._execute_trade(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, {}
        
    def _get_observation(self):
        """
        Construct the current state observation
        """
        current_data = self.data.iloc[self.current_step]
        
        # Market data features
        market_features = np.array([
            current_data['open'],
            current_data['high'],
            current_data['low'],
            current_data['close'],
            current_data['volume'],
            # Technical indicators
            current_data['sma_20'],
            current_data['sma_50'],
            current_data['rsi'],
            current_data['macd'],
            current_data['bbands_upper'],
            current_data['bbands_lower']
        ])
        
        # Account features
        account_features = np.array([
            self.balance,
            self.position,
            self._calculate_unrealized_pnl()
        ])
        
        return np.concatenate([market_features, account_features])
        
    def _execute_trade(self, action):
        """
        Execute trading action
        """
        current_price = self.data.iloc[self.current_step]['close']
        new_position = float(action)
        
        # Calculate position change
        position_change = new_position - self.position
        
        if position_change != 0:
            # Calculate trade cost
            trade_cost = abs(position_change) * current_price * 0.001  # 0.1% commission
            
            # Update balance and position
            self.balance -= trade_cost
            self.position = new_position
            
            # Record trade
            self.trades.append({
                'timestamp': self.data.index[self.current_step],
                'price': current_price,
                'position_change': position_change,
                'cost': trade_cost
            })
            
    def _calculate_reward(self):
        """
        Calculate the reward for the current step
        """
        # Calculate PnL
        unrealized_pnl = self._calculate_unrealized_pnl()
        
        # Calculate Sharpe ratio component
        returns = self._calculate_returns()
        sharpe = self._calculate_sharpe(returns)
        
        # Combine different reward components
        reward = (
            0.7 * unrealized_pnl / self.initial_balance +  # PnL component
            0.3 * sharpe  # Risk-adjusted return component
        )
        
        # Add penalties
        reward -= self._calculate_penalties()
        
        return reward
        
    def _calculate_unrealized_pnl(self):
        """
        Calculate unrealized PnL of current position
        """
        current_price = self.data.iloc[self.current_step]['close']
        return self.position * current_price - self.balance
        
    def _calculate_returns(self):
        """
        Calculate historical returns
        """
        if len(self.trades) < 2:
            return np.array([0])
            
        returns = []
        for i in range(1, len(self.trades)):
            pnl = (self.trades[i]['price'] - self.trades[i-1]['price']) * \
                  self.trades[i-1]['position_change']
            returns.append(pnl / self.initial_balance)
            
        return np.array(returns)
        
    def _calculate_sharpe(self, returns: np.array):
        """
        Calculate Sharpe ratio
        """
        if len(returns) < 2:
            return 0
            
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
    def _calculate_penalties(self):
        """
        Calculate penalties for risk management
        """
        penalties = 0
        
        # Penalty for excessive leverage
        if abs(self.position) > 2:
            penalties += 0.1 * abs(self.position - 2)
            
        # Penalty for excessive drawdown
        drawdown = (self.initial_balance - self.balance) / self.initial_balance
        if drawdown > 0.1:  # 10% drawdown threshold
            penalties += 0.2 * (drawdown - 0.1)
            
        return penalties

class ReinforcementLearningStrategy:
    def __init__(self, config: Dict):
        """Initialize the RL strategy with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize empty environment with dummy data
        dummy_data = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000],
            'sma_20': [100.0],
            'sma_50': [100.0],
            'rsi': [50.0],
            'macd': [0.0],
            'bbands_upper': [102.0],
            'bbands_lower': [98.0]
        })
        
        # Initialize environment with configuration parameters
        self.env = TradingEnvironment(
            data=dummy_data,
            initial_balance=self.config.get('trading_parameters', {}).get('initial_balance', 100000)
        )
        
        # Initialize model with configuration parameters
        self.model_config = self.config.get('model_parameters', {
            'model_type': 'PPO',
            'learning_rate': 0.0003,
            'batch_size': 64,
            'buffer_size': 100000
        })
        
        self._init_model()
        
        # Trading parameters from configuration
        trading_params = self.config.get('trading_parameters', {})
        self.min_confidence = trading_params.get('min_confidence', 0.6)
        self.position_sizing = trading_params.get('position_sizing', 'dynamic')
        self.max_position_size = trading_params.get('max_position_size', 1.0)
        
        # Performance tracking
        self.trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_trade': 0.0,
            'max_consecutive_losses': 0
        }
        
    def _init_model(self):
        """Initialize the RL model with proper error handling"""
        try:
            model_type = self.model_config.get('model_type', 'PPO')
            
            if model_type == 'PPO':
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.model_config.get('learning_rate', 0.0003),
                    batch_size=self.model_config.get('batch_size', 64),
                    buffer_size=self.model_config.get('buffer_size', 100000),
                    verbose=1
                )
            elif model_type == 'A2C':
                self.model = A2C(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.model_config.get('learning_rate', 0.0007),
                    verbose=1
                )
            elif model_type == 'SAC':
                self.model = SAC(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.model_config.get('learning_rate', 0.0003),
                    buffer_size=self.model_config.get('buffer_size', 100000),
                    verbose=1
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            self.logger.info(f"Successfully initialized {model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error initializing RL model: {str(e)}")
            raise

    def generate_signals(self, market_data: pd.DataFrame) -> Dict:
        """Generate trading signals using the RL model"""
        try:
            if market_data.empty:
                self.logger.error("Empty market data provided")
                return self._create_neutral_signal()
                
            # Update environment with new market data
            self.env.data = market_data
            self.env.reset()
            
            # Get model prediction
            observation = self.env._get_observation()
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Calculate signal confidence
            confidence = self._calculate_signal_confidence(action[0], market_data)
            
            # Generate signal only if confidence meets minimum threshold
            if abs(confidence) >= self.min_confidence:
                signal = self._create_signal(action[0], confidence, market_data)
            else:
                signal = self._create_neutral_signal()
                
            # Update performance metrics
            self._update_performance_metrics(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return self._create_neutral_signal()
            
    def _calculate_signal_confidence(self, action: float, market_data: pd.DataFrame) -> float:
        """Calculate confidence level of the signal"""
        try:
            # Get latest market data
            latest = market_data.iloc[-1]
            
            # Calculate trend strength using technical indicators
            trend_strength = 0.0
            
            # RSI contribution
            if 'rsi' in latest:
                rsi = latest['rsi']
                if rsi > 70:
                    trend_strength -= 0.2
                elif rsi < 30:
                    trend_strength += 0.2
                    
            # MACD contribution
            if all(x in latest for x in ['macd', 'macd_signal']):
                macd_diff = latest['macd'] - latest['macd_signal']
                trend_strength += np.clip(macd_diff / 2, -0.2, 0.2)
                
            # Bollinger Bands contribution
            if all(x in latest for x in ['close', 'bb_upper', 'bb_lower']):
                bb_pos = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                trend_strength += np.clip(bb_pos - 0.5, -0.2, 0.2)
                
            # Combine model confidence with technical indicators
            confidence = 0.7 * action + 0.3 * trend_strength
            
            return np.clip(confidence, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {str(e)}")
            return 0.0
            
    def _create_signal(self, action: float, confidence: float, market_data: pd.DataFrame) -> Dict:
        """Create a trading signal with position sizing"""
        try:
            latest_price = market_data.iloc[-1]['close']
            
            # Calculate position size based on confidence and configuration
            if self.position_sizing == 'dynamic':
                position_size = self.max_position_size * abs(confidence)
            else:
                position_size = self.max_position_size
                
            return {
                'timestamp': market_data.index[-1],
                'action': 'buy' if action > 0 else 'sell',
                'confidence': confidence,
                'position_size': position_size,
                'price': latest_price,
                'indicators': {
                    'rsi': market_data.iloc[-1].get('rsi', None),
                    'macd': market_data.iloc[-1].get('macd', None),
                    'sma_20': market_data.iloc[-1].get('sma_20', None),
                    'sma_50': market_data.iloc[-1].get('sma_50', None)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error creating signal: {str(e)}")
            return self._create_neutral_signal()
            
    def _create_neutral_signal(self) -> Dict:
        """Create a neutral (no trade) signal"""
        return {
            'timestamp': datetime.now(),
            'action': 'neutral',
            'confidence': 0.0,
            'position_size': 0.0,
            'price': None,
            'indicators': {}
        }
        
    def _update_performance_metrics(self, signal: Dict):
        """Update strategy performance metrics"""
        try:
            if signal['action'] != 'neutral':
                self.trades.append(signal)
                self.performance_metrics['total_trades'] += 1
                
                # Calculate basic metrics if we have enough trades
                if len(self.trades) >= 2:
                    returns = []
                    winning_trades = 0
                    losing_trades = 0
                    consecutive_losses = 0
                    max_consecutive_losses = 0
                    
                    for i in range(1, len(self.trades)):
                        prev_trade = self.trades[i-1]
                        curr_trade = self.trades[i]
                        
                        if prev_trade['action'] != curr_trade['action']:
                            pnl = (curr_trade['price'] - prev_trade['price']) * \
                                  prev_trade['position_size'] * \
                                  (1 if prev_trade['action'] == 'buy' else -1)
                                  
                            returns.append(pnl)
                            
                            if pnl > 0:
                                winning_trades += 1
                                consecutive_losses = 0
                            else:
                                losing_trades += 1
                                consecutive_losses += 1
                                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    if returns:
                        self.performance_metrics.update({
                            'winning_trades': winning_trades,
                            'losing_trades': losing_trades,
                            'total_profit': sum(returns),
                            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252),
                            'win_rate': winning_trades / (winning_trades + losing_trades),
                            'profit_factor': sum([r for r in returns if r > 0]) / abs(sum([r for r in returns if r < 0]) + 1e-6),
                            'average_trade': np.mean(returns),
                            'max_consecutive_losses': max_consecutive_losses
                        })
                        
        except Exception as e:
            self.logger.error(f"Error training RL model: {str(e)}")
            return None

    def _evaluate_model(self, eval_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance
        """
        try:
            eval_env = TradingEnvironment(eval_data)
            
            returns = []
            sharpe_ratios = []
            max_drawdowns = []
            
            for episode in range(self.config['eval_episodes']):
                obs, _ = eval_env.reset()
                done = False
                episode_return = 0
                
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, _, _ = eval_env.step(action)
                    episode_return += reward
                    
                returns.append(episode_return)
                sharpe_ratios.append(
                    self._calculate_sharpe_ratio(eval_env.trades)
                )
                max_drawdowns.append(
                    self._calculate_max_drawdown(eval_env.trades)
                )
                
            return {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'mean_sharpe': np.mean(sharpe_ratios),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'best_episode_return': max(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating RL model: {str(e)}")
            return None

    def predict(self, current_state: np.array) -> Tuple[float, Dict]:
        """
        Generate trading decision for current market state
        """
        try:
            # Get model prediction
            action, _ = self.model.predict(current_state)
            
            # Calculate prediction confidence
            confidence = self._calculate_confidence(current_state)
            
            return float(action), {
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {str(e)}")
            return 0.0, None

    def _calculate_confidence(self, state: np.array) -> float:
        """
        Calculate confidence in the prediction
        """
        try:
            # Get action probabilities
            action_probs = self.model.policy.get_distribution(
                torch.FloatTensor(state)
            ).distribution.probs
            
            # Calculate entropy-based confidence
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
            max_entropy = -torch.log(torch.tensor(1.0 / len(action_probs)))
            
            return 1 - (entropy / max_entropy).item()
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default medium confidence

    def save_model(self, path: str):
        """
        Save the trained model
        """
        try:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str):
        """
        Load a trained model
        """
        try:
            if self.config['model_type'] == 'PPO':
                self.model = PPO.load(path)
            elif self.config['model_type'] == 'A2C':
                self.model = A2C.load(path)
            elif self.config['model_type'] == 'SAC':
                self.model = SAC.load(path)
                
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")

    def update_model(self, new_data: pd.DataFrame):
        """
        Update the model with new data
        """
        try:
            # Create environment with new data
            update_env = DummyVecEnv([lambda: TradingEnvironment(new_data)])
            
            # Continue training
            self.model.set_env(update_env)
            self.model.learn(
                total_timesteps=self.config['train_episodes'] // 10
            )
            
            self.logger.info("Model updated with new data")
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")

    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """
        Calculate Sharpe ratio from trade history
        """
        if len(trades) < 2:
            return 0.0
            
        returns = np.array([trade['position_change'] for trade in trades])
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """
        Calculate maximum drawdown from trade history
        """
        if len(trades) < 2:
            return 0.0
            
        cumulative_returns = np.cumsum([
            trade['position_change'] for trade in trades
        ])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        
        return np.max(drawdown)
