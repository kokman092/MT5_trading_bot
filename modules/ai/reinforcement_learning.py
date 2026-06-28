import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM
from tensorflow.keras.optimizers import Adam
import logging
from collections import deque
import random
import json
from datetime import datetime
import os

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        current_data = self.data.iloc[self.current_step]
        
        state = [
            current_data['returns'],
            current_data['volatility'],
            current_data['rsi'] / 100,
            current_data['macd'] / 100,
            current_data['bb_position'],
            current_data['volume_ratio'] - 1,
            current_data['trend_strength'],
            self.position,
            self.balance / self.initial_balance - 1
        ]
        
        return np.array(state)
    
    def step(self, action):
        # Action: 0 (sell), 1 (hold), 2 (buy)
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute trade
        if action == 0 and self.position > 0:  # Sell
            self.balance += current_price * abs(self.position)
            self.trades.append({
                'type': 'sell',
                'price': current_price,
                'size': self.position,
                'balance': self.balance
            })
            self.position = 0
            
        elif action == 2 and self.position <= 0:  # Buy
            position_size = self.balance * 0.1 / current_price  # Use 10% of balance
            self.balance -= current_price * position_size
            self.position = position_size
            self.trades.append({
                'type': 'buy',
                'price': current_price,
                'size': position_size,
                'balance': self.balance
            })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        if done:
            if self.position > 0:
                self.balance += current_price * self.position
                self.position = 0
        
        reward = (self.balance / self.initial_balance - 1) * 100
        
        return self._get_state(), reward, done

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # DQN parameters
        self.gamma = config.get('GAMMA', 0.95)
        self.epsilon = config.get('EPSILON', 1.0)
        self.epsilon_min = config.get('EPSILON_MIN', 0.01)
        self.epsilon_decay = config.get('EPSILON_DECAY', 0.995)
        self.learning_rate = config.get('LEARNING_RATE', 0.001)
        self.batch_size = config.get('BATCH_SIZE', 32)
        self.memory = deque(maxlen=config.get('MEMORY_SIZE', 2000))
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Load model if exists
        self.model_path = config.get('RL_MODEL_PATH', 'models/rl_model.h5')
        if os.path.exists(self.model_path):
            self.load_model()
    
    def _build_model(self):
        """Neural Network for Deep Q-learning"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target model weights with current model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load_model(self):
        """Load saved model"""
        try:
            self.model.load_weights(self.model_path)
            self.target_model.load_weights(self.model_path)
            self.logger.info("Loaded RL model from disk")
        except Exception as e:
            self.logger.error(f"Error loading RL model: {str(e)}")
    
    def save_model(self):
        """Save model weights"""
        try:
            self.model.save_weights(self.model_path)
            self.logger.info("Saved RL model to disk")
        except Exception as e:
            self.logger.error(f"Error saving RL model: {str(e)}")

class RLTrader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment and agent
        self.state_size = 9  # Defined by TradingEnvironment._get_state
        self.action_size = 3  # sell, hold, buy
        self.agent = DQNAgent(self.state_size, self.action_size, config)
        
        # Training parameters
        self.episodes = config.get('RL_EPISODES', 100)
        self.train_interval = config.get('RL_TRAIN_INTERVAL', 24)  # hours
        self.last_training = None
    
    def train(self, data):
        """Train the RL agent"""
        try:
            self.logger.info("Starting RL training...")
            env = TradingEnvironment(data)
            
            best_reward = float('-inf')
            training_history = []
            
            for episode in range(self.episodes):
                state = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action = self.agent.act(state)
                    next_state, reward, done = env.step(action)
                    
                    self.agent.remember(state, action, reward, next_state, done)
                    self.agent.replay()
                    
                    state = next_state
                    total_reward += reward
                
                if episode % 5 == 0:
                    self.agent.update_target_model()
                
                training_history.append({
                    'episode': episode,
                    'reward': total_reward,
                    'epsilon': self.agent.epsilon
                })
                
                if total_reward > best_reward:
                    best_reward = total_reward
                    self.agent.save_model()
                
                self.logger.info(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
            
            self.last_training = datetime.now()
            
            # Save training history
            with open('training_history.json', 'w') as f:
                json.dump(training_history, f, indent=4)
            
            return training_history
            
        except Exception as e:
            self.logger.error(f"Error in RL training: {str(e)}")
            return None
    
    def predict(self, state):
        """Get trading action from RL agent"""
        try:
            action = self.agent.act(state, training=False)
            return {
                'action': action,
                'action_name': ['sell', 'hold', 'buy'][action],
                'confidence': 1.0  # RL agents don't provide confidence scores
            }
        except Exception as e:
            self.logger.error(f"Error in RL prediction: {str(e)}")
            return None
    
    def needs_training(self):
        """Check if model needs retraining"""
        if self.last_training is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
        return hours_since_training >= self.train_interval
