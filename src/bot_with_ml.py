import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predictor import ImprovedMLPredictor
from mt5_integration.connection import Connection
from config.settings import LOGIN, PASSWORD, SERVER, TRADING_CONFIG
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

class MLTradingBot:
    def __init__(self):
        self.connection = Connection(LOGIN, PASSWORD, SERVER)
        self.predictor = ImprovedMLPredictor()
        self.symbol = TRADING_CONFIG["symbol"]
        self.timeframe = TRADING_CONFIG["timeframe"]
        
    def get_market_data(self, bars=200):
        """Get recent market data"""
        try:
            # Map timeframe
            timeframes = {
                1: mt5.TIMEFRAME_M1,
                5: mt5.TIMEFRAME_M5,
                15: mt5.TIMEFRAME_M15,
                30: mt5.TIMEFRAME_M30,
                60: mt5.TIMEFRAME_H1
            }
            
            tf = timeframes.get(self.timeframe, mt5.TIMEFRAME_M5)
            
            # Get data
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
            
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting market data: {e}")
            return None
    
    def run_ml_analysis(self):
        """Run ML analysis on current market"""
        print(f"🤖 Running ML Analysis for {self.symbol}")
        print("=" * 50)
        
        if not self.connection.connect():
            print("❌ Failed to connect to MT5")
            return
            
        if self.predictor.model is None:
            print("❌ ML Model not loaded")
            return
        
        try:
            # Get market data
            print("📊 Getting market data...")
            market_data = self.get_market_data(200)
            
            if market_data is None:
                print("❌ Failed to get market data")
                return
            
            print(f"✅ Got {len(market_data)} bars of data")
            
            # Get ML prediction
            print("🔮 Making ML prediction...")
            prediction = self.predictor.predict_signal(market_data)
            
            if "error" in prediction:
                print(f"❌ Prediction error: {prediction['error']}")
                return
            
            # Display results
            self.display_analysis(market_data.iloc[-1], prediction)
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
        
        finally:
            self.connection.disconnect()
    
    def display_analysis(self, current_bar, prediction):
        """Display analysis results"""
        print("\n" + "="*50)
        print("📈 MARKET ANALYSIS RESULTS")
        print("="*50)
        
        # Current market info
        print(f"💰 Symbol: {self.symbol}")
        print(f"📅 Time: {current_bar['time']}")
        print(f"💲 Price: {current_bar['close']:.5f}")
        print(f"📊 Volume: {current_bar.get('tick_volume', 'N/A')}")
        
        print("\n" + "-"*50)
        print("🤖 ML PREDICTION")
        print("-"*50)
        
        # ML prediction
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        # Color coding
        if signal == "BUY":
            emoji = "🟢"
            color = "GREEN"
        elif signal == "SELL":
            emoji = "🔴"
            color = "RED"
        else:
            emoji = "🟡"
            color = "YELLOW"
        
        print(f"{emoji} Signal: {signal}")
        print(f"🎯 Confidence: {confidence:.2%}")
        print(f"📊 Prediction Class: {prediction['prediction']}")
        
        # Probabilities
        probs = prediction['probabilities']
        print(f"\n📊 Probabilities:")
        print(f"   🔴 SELL: {probs['sell']:.2%}")
        print(f"   🟡 HOLD: {probs['hold']:.2%}")
        print(f"   🟢 BUY:  {probs['buy']:.2%}")
        
        # Features used
        print(f"\n🔧 Features:")
        features = prediction.get('features', {})
        for name, value in features.items():
            print(f"   {name}: {value:.6f}")
        
        # Trading recommendation
        print(f"\n💡 RECOMMENDATION:")
        if confidence > 0.6:
            if signal == "BUY":
                print(f"   ✅ STRONG BUY signal - Consider opening long position")
            elif signal == "SELL":
                print(f"   ✅ STRONG SELL signal - Consider opening short position")
            else:
                print(f"   ⏸️ HOLD - Wait for better opportunity")
        elif confidence > 0.4:
            print(f"   ⚠️ MODERATE {signal} signal - Use with caution")
        else:
            print(f"   ❌ WEAK signal - Not recommended to trade")
        
        print("="*50)
    
    def run_continuous_monitoring(self, interval_seconds=60):
        """Run continuous ML monitoring"""
        print(f"🔄 Starting continuous monitoring (every {interval_seconds}s)")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                self.run_ml_analysis()
                print(f"\n⏳ Waiting {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

def main():
    bot = MLTradingBot()
    
    choice = input("""
Choose mode:
1. Single analysis
2. Continuous monitoring (every 60s)
3. Quick test (every 10s, 5 times)

Enter choice (1/2/3): """)
    
    if choice == "1":
        bot.run_ml_analysis()
    elif choice == "2":
        bot.run_continuous_monitoring(60)
    elif choice == "3":
        print("🧪 Quick test mode...")
        for i in range(5):
            print(f"\n--- Test {i+1}/5 ---")
            bot.run_ml_analysis()
            if i < 4:
                time.sleep(10)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()