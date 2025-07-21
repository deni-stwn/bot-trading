import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new predictor
from ml.predictor_v2 import ImprovedMLPredictorV2
from final.live_trading_bot import LiveTradingBot
from final.performance_tracker import PerformanceTracker
from mt5_integration.connection import Connection
from config.settings import LOGIN, PASSWORD, SERVER, TRADING_CONFIG
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

class EnhancedLiveTradingBot(LiveTradingBot):
    def __init__(self, debug_mode=True, min_confidence=0.50):  # LOWERED TO 50%
        # Initialize parent class
        super().__init__()
        
        # Override with improved predictor
        self.predictor = ImprovedMLPredictorV2()
        
        self.debug_mode = debug_mode
        self.cycle_count = 0
        
        # LOWERED CONFIDENCE THRESHOLD
        self.min_confidence = min_confidence  # Configurable now
        
        # More aggressive settings
        self.max_positions = 5  # Allow more positions
        self.min_time_between_trades = 180  # Reduce to 3 minutes
    
    def trading_cycle(self):
        """Enhanced trading cycle with v2 model"""
        try:
            self.cycle_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if self.debug_mode:
                print(f"\n🔄 CYCLE {self.cycle_count} - {current_time} [MODEL V2 - {self.min_confidence:.0%} threshold]")
                print("-" * 60)
            
            # 1. Check daily loss limit
            if self.debug_mode:
                print("1️⃣ Checking daily loss limit...")
            
            if not self.check_daily_loss_limit():
                if self.debug_mode:
                    print("❌ Daily loss limit reached!")
                return
            
            if self.debug_mode:
                print("✅ Daily loss limit OK")
            
            # 2. Get market data
            if self.debug_mode:
                print("2️⃣ Getting market data...")
            
            market_data = self.get_market_data()
            if market_data is None:
                if self.debug_mode:
                    print("❌ Failed to get market data")
                return
            
            current_price = market_data.iloc[-1]['close']
            if self.debug_mode:
                print(f"✅ Market data OK - Current price: {current_price:.5f}")
            
            # 3. Get ML prediction
            if self.debug_mode:
                print("3️⃣ Getting ML prediction...")
            
            prediction = self.predictor.predict_signal(market_data)
            if "error" in prediction:
                if self.debug_mode:
                    print(f"❌ Prediction error: {prediction['error']}")
                return
            
            signal = prediction['signal']
            confidence = prediction['confidence']
            
            if self.debug_mode:
                print(f"✅ ML Prediction: {signal} (Confidence: {confidence:.2%})")
                
                # Show probabilities
                probs = prediction['probabilities']
                print(f"   📊 Probabilities: SELL({probs['sell']:.2%}) | HOLD({probs['hold']:.2%}) | BUY({probs['buy']:.2%})")
            
            # 4. Check trading conditions
            if self.debug_mode:
                print("4️⃣ Checking trading conditions...")
            
            # Check if we should trade
            should_trade, reason = self.should_trade_detailed(prediction)
            
            if not should_trade:
                if self.debug_mode:
                    print(f"❌ {reason}")
                return
            
            if self.debug_mode:
                print("✅ All conditions met - EXECUTING TRADE!")
            
            # 5. Execute trade
            self.execute_trade(prediction, market_data.iloc[-1])
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error in enhanced trading cycle: {e}")
            self.logger.error(f"Error in enhanced trading cycle: {e}")
    
    def should_trade_detailed(self, prediction):
        """Enhanced trading condition check"""
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        # Check confidence level
        if confidence < self.min_confidence:
            return False, f"Confidence too low: {confidence:.2%} < {self.min_confidence:.2%}"
        
        # Check signal strength
        if signal == "HOLD":
            return False, "Signal is HOLD - no action needed"
        
        # Check time between trades
        if self.last_trade_time:
            time_diff = (datetime.now() - self.last_trade_time).seconds
            if time_diff < self.min_time_between_trades:
                return False, f"Cooldown active: {time_diff}s < {self.min_time_between_trades}s"
        
        # Check max positions
        open_positions = self.get_open_positions()
        if self.debug_mode:
            print(f"📊 Current positions: {len(open_positions)}")
        
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached: {len(open_positions)} >= {self.max_positions}"
        
        return True, "All conditions met"

def main():
    print("🚀 ENHANCED TRADING SYSTEM V2 - FIXED")
    print("=" * 60)
    print("🔥 Using Improved Model (58.71% accuracy)")
    print("📊 Enhanced Features: Bollinger Bands + Price Change")
    print("🎯 LOWERED Threshold for More Trading Opportunities")
    
    # Choose confidence threshold
    threshold_choice = input("""
Choose confidence threshold:
1. Conservative (60% - fewer trades, higher quality)
2. Balanced (55% - moderate trading)
3. Aggressive (50% - more trades, lower quality)
4. Very Aggressive (45% - maximum trades)

Enter choice (1/2/3/4): """)
    
    threshold_map = {
        "1": 0.60,
        "2": 0.55,
        "3": 0.50,
        "4": 0.45
    }
    
    min_confidence = threshold_map.get(threshold_choice, 0.50)
    
    print(f"🎯 Selected confidence threshold: {min_confidence:.0%}")
    
    # Choose mode
    mode_choice = input("""
Choose trading mode:
1. Normal mode (minimal output)
2. Debug mode (detailed output)
3. Cancel

Enter choice (1/2/3): """)
    
    if mode_choice == "3":
        print("❌ Trading cancelled.")
        return
    
    debug_mode = (mode_choice == "2")
    
    # Initialize enhanced bot with chosen threshold
    bot = EnhancedLiveTradingBot(debug_mode=debug_mode, min_confidence=min_confidence)
    
    if not bot.connection.connect():
        print("❌ Failed to connect to MT5")
        return
        
    if bot.predictor.model is None:
        print("❌ Enhanced ML Model not loaded")
        return
    
    print("✅ Enhanced Model V2 loaded successfully!")
    
    # Safety confirmation
    confirmation = input(f"""
⚠️  ENHANCED LIVE TRADING WARNING ⚠️
This will use the IMPROVED MODEL with REAL MONEY!
Confidence threshold: {min_confidence:.0%}
Type 'START ENHANCED TRADING' to continue: """)
    
    if confirmation != "START ENHANCED TRADING":
        print("❌ Trading cancelled.")
        return
    
    # Get initial balance
    account_info = mt5.account_info()
    if account_info:
        bot.starting_balance = account_info.balance
        bot.current_balance = account_info.balance
        print(f"💰 Starting Balance: ${bot.starting_balance:,.2f}")
    
    bot.is_trading = True
    
    try:
        print(f"🔥 Enhanced trading started with V2 model! ({min_confidence:.0%} threshold)")
        
        while bot.is_trading:
            bot.trading_cycle()
            
            if debug_mode:
                print(f"\n⏳ Waiting 30 seconds...")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n🛑 Enhanced trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        bot.stop_trading()

if __name__ == "__main__":
    main()