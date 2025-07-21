import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.data_collector import DataCollector
from ml.feature_engineering import FeatureEngineer
from config.settings import LOGIN, PASSWORD, SERVER, TRADING_CONFIG
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    # Generate sample OHLCV data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='5T')
    
    np.random.seed(42)
    base_price = 50000
    
    sample_data = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Random walk
        change = np.random.normal(0, 0.01)
        current_price *= (1 + change)
        
        # Generate OHLC
        high = current_price * (1 + abs(np.random.normal(0, 0.005)))
        low = current_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        close = current_price
        volume = np.random.randint(100, 1000)
        
        sample_data.append({
            'time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': volume
        })
    
    return pd.DataFrame(sample_data)

def improved_prepare_dataset(df: pd.DataFrame):
    """Improved dataset preparation"""
    print("🔧 Preparing dataset with improved methods...")
    
    # Basic feature engineering
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Create target (simplified)
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    
    # Use PERCENTILE-based targets (key improvement!)
    returns = df['future_return'].dropna()
    buy_threshold = returns.quantile(0.75)   # Top 25% = BUY
    sell_threshold = returns.quantile(0.25)  # Bottom 25% = SELL
    
    df['target'] = 1  # Default HOLD
    df.loc[df['future_return'] > buy_threshold, 'target'] = 2    # BUY
    df.loc[df['future_return'] < sell_threshold, 'target'] = 0   # SELL
    
    # Select features
    feature_cols = ['returns', 'sma_20', 'sma_50', 'rsi', 'volatility']
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 100:
        return None, None
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"✅ Dataset ready: {len(X)} samples")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def calculate_rsi(prices, period=14):
    """Simple RSI calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_simple_model():
    """Train with simple, reliable method"""
    setup_logging()
    
    print("🤖 Starting IMPROVED ML Training")
    print("=" * 50)
    
    choice = input("""
Choose data source:
1. Use sample data (recommended for testing)
2. Use real MT5 data
Enter choice (1 or 2): """)
    
    if choice == "1":
        # Use sample data
        print("📊 Using sample data...")
        df = create_sample_data()
        
    else:
        # Use real data
        print("📊 Collecting real data from MT5...")
        try:
            collector = DataCollector(LOGIN, PASSWORD, SERVER)
            df = collector.collect_training_data(
                symbol=TRADING_CONFIG["symbol"],
                days_back=180,  # Reduce to 6 months
                timeframe=TRADING_CONFIG["timeframe"]
            )
            
            if df is None:
                print("❌ Failed to collect real data, using sample data...")
                df = create_sample_data()
                
        except Exception as e:
            print(f"❌ Error collecting data: {e}")
            print("📊 Using sample data instead...")
            df = create_sample_data()
    
    # Prepare dataset
    X, y = improved_prepare_dataset(df)
    
    if X is None:
        print("❌ Dataset preparation failed")
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {len(X_train)}")
    print(f"📊 Test set: {len(X_test)}")
    
    # Train simple Random Forest (more reliable than LightGBM)
    print("🌳 Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n📊 Results:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, test_pred))
    
    # Save model
    os.makedirs('models/trained_models', exist_ok=True)
    joblib.dump(model, 'models/trained_models/improved_model.pkl')
    
    # Save feature names
    feature_info = {
        'feature_names': list(X.columns),
        'model_type': 'RandomForest',
        'accuracy': test_acc
    }
    joblib.dump(feature_info, 'models/trained_models/model_info.pkl')
    
    print("\n✅ Model saved successfully!")
    print("📁 Location: models/trained_models/improved_model.pkl")
    
    return True

if __name__ == "__main__":
    success = train_simple_model()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now use the improved model in your bot.")
    else:
        print("\n❌ Training failed. Please check the logs.")