import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.data_processor import DataProcessor
from env.forex_env import ForexEnv

def train_model(symbol: str = "USDJPY", timeframe: str = "5", 
                start_date: datetime = datetime(2023, 1, 1),
                end_date: datetime = datetime(2023, 12, 31),
                total_timesteps: int = 100000):
    """Train the PPO model"""
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Prepare data
    data_processor = DataProcessor()
    df = data_processor.fetch_data(symbol, timeframe, start_date, end_date)
    
    if df is None or df.empty:
        print("Failed to fetch training data")
        return
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: ForexEnv(df)])
    
    # Initialize the model
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./logs/")
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model_path = f"models/ppo_forex_{symbol}_{timeframe}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
