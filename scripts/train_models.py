import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from yaml file"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return None

def load_training_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load training data from CSV file"""
    try:
        filename = f'data/ml_training_data_{symbol}_{timeframe}.csv'
        df = pd.DataFrame()
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return pd.DataFrame()

def prepare_sequences(data: pd.DataFrame, sequence_length: int = 20) -> tuple:
    """Prepare sequences for training"""
    try:
        # Define features to use
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'sma_20', 'ema_20', 'adx', 'rsi', 'macd', 'macd_signal',
            'atr', 'bb_upper', 'bb_lower'
        ]
        
        # Prepare feature data
        feature_data = data[feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length):
            sequence = scaled_data[i:i + sequence_length]
            target = 1 if scaled_data[i + sequence_length][3] > scaled_data[i + sequence_length - 1][3] else 0
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets), scaler
        
    except Exception as e:
        logger.error(f"Error preparing sequences: {str(e)}")
        return None, None, None

def build_model(input_shape: tuple) -> tf.keras.Model:
    """Build the ML model"""
    try:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # LSTM layers with residual connections
        lstm_out1 = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        lstm_out1 = tf.keras.layers.Dropout(0.2)(lstm_out1)
        
        # Self-attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=16
        )(lstm_out1, lstm_out1, lstm_out1)
        attention = tf.keras.layers.LayerNormalization()(attention)
        
        # Add residual connection
        x = tf.keras.layers.Add()([lstm_out1, attention])
        
        # Second LSTM layer
        lstm_out2 = tf.keras.layers.LSTM(32)(x)
        lstm_out2 = tf.keras.layers.Dropout(0.2)(lstm_out2)
        
        # Dense layers
        x = tf.keras.layers.Dense(32, activation='relu')(lstm_out2)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        return None

def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray, symbol: str, timeframe: str):
    """Train the ML model"""
    try:
        # Create checkpoints directory
        os.makedirs('models/checkpoints', exist_ok=True)
        
        # Setup callbacks
        checkpoint_path = f'models/checkpoints/model_{symbol}_{timeframe}.h5'
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'training_history': {
                'accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history['val_accuracy'][-1],
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            }
        }
        
        metadata_path = checkpoint_path.replace('.h5', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Model trained and saved: {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

def main():
    """Main function to train ML models"""
    try:
        # Load configuration
        config = load_config()
        if config is None:
            return
            
        # Get symbols and timeframes
        symbols = config['trading']['symbols']
        timeframes = config['market_analysis']['timeframes']
        
        # Train models for each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Training model for {symbol} {timeframe}...")
                
                # Load training data
                data = load_training_data(symbol, timeframe)
                if data.empty:
                    logger.error(f"No training data for {symbol} {timeframe}")
                    continue
                    
                # Prepare sequences
                X, y, scaler = prepare_sequences(data)
                if X is None or y is None:
                    continue
                    
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )
                
                # Build and train model
                model = build_model(input_shape=(X.shape[1], X.shape[2]))
                if model is None:
                    continue
                    
                success = train_model(
                    model, X_train, y_train, X_val, y_val,
                    symbol, timeframe
                )
                
                if success:
                    logger.info(f"Successfully trained model for {symbol} {timeframe}")
                else:
                    logger.error(f"Failed to train model for {symbol} {timeframe}")
                    
        logger.info("Model training complete")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == '__main__':
    main() 