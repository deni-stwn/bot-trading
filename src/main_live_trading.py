import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final.live_trading_bot import LiveTradingBot
from final.performance_tracker import PerformanceTracker
from ml.predictor import ImprovedMLPredictor
from mt5_integration.connection import Connection
from config.settings import LOGIN, PASSWORD, SERVER, TRADING_CONFIG
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

class DebugLiveTradingBot(LiveTradingBot):
    def __init__(self, debug_mode=True):
        super().__init__()
        self.debug_mode = debug_mode
        self.cycle_count = 0
        
    def trading_cycle(self):
        """Enhanced trading cycle with optional debug output"""
        try:
            self.cycle_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if self.debug_mode:
                print(f"\n🔄 CYCLE {self.cycle_count} - {current_time}")
                print("-" * 50)
            
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
                print(f"❌ Error in trading cycle: {e}")
            self.logger.error(f"Error in trading cycle: {e}")
    
    def should_trade_detailed(self, prediction):
        """Detailed trading condition check with reasons"""
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
        
        # Check if we already have same direction trade
        if open_positions:
            existing_directions = [pos.type for pos in open_positions]
            if ((signal == "BUY" and mt5.POSITION_TYPE_BUY in existing_directions) or 
                (signal == "SELL" and mt5.POSITION_TYPE_SELL in existing_directions)):
                return False, f"Already have {signal} position open"
        
        return True, "All conditions met"
    
    def execute_trade(self, prediction, current_bar):
        """Enhanced execute trade with debug output"""
        if self.debug_mode:
            print(f"\n💰 EXECUTING TRADE")
            print("=" * 30)
        
        try:
            signal = prediction['signal']
            confidence = prediction['confidence']
            current_price = current_bar['close']
            
            if self.debug_mode:
                print(f"📊 Signal: {signal}")
                print(f"🎯 Confidence: {confidence:.2%}")
                print(f"💲 Current Price: {current_price:.5f}")
            
            # Calculate position size
            lot_size = self.calculate_position_size(current_price)
            if self.debug_mode:
                print(f"📏 Lot Size: {lot_size}")
            
            # Calculate SL and TP
            sl_price, tp_price = self.calculate_sl_tp(signal, current_price)
            if self.debug_mode:
                print(f"🛡️ Stop Loss: {sl_price:.5f}")
                print(f"🎯 Take Profit: {tp_price:.5f}")
            
            # Get current tick
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                if self.debug_mode:
                    print("❌ Failed to get current tick")
                return
            
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                if self.debug_mode:
                    print(f"🟢 BUY order at ASK: {price:.5f}")
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                if self.debug_mode:
                    print(f"🔴 SELL order at BID: {price:.5f}")
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": f"ML_Bot_{signal}_{confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if self.debug_mode:
                print(f"📤 Sending order request...")
                print(f"   Symbol: {request['symbol']}")
                print(f"   Volume: {request['volume']}")
                print(f"   Price: {request['price']:.5f}")
            
            # Send order
            result = mt5.order_send(request)
            
            if self.debug_mode:
                print(f"📨 Order result:")
                print(f"   Return code: {result.retcode}")
                print(f"   Comment: {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                success_msg = f"✅ TRADE EXECUTED: {signal} {lot_size} lots at {result.price:.5f}"
                print(success_msg)  # Always show successful trades
                
                if self.debug_mode:
                    print(f"🎫 Ticket: {result.order}")
                
                # Update history
                trade_info = {
                    "timestamp": datetime.now(),
                    "ticket": result.order,
                    "signal": signal,
                    "confidence": confidence,
                    "entry_price": result.price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "lot_size": lot_size,
                }
                
                self.trades_history.append(trade_info)
                self.last_trade_time = datetime.now()
                
                self.logger.info(success_msg)
                
            else:
                error_msg = f"❌ TRADE FAILED: {result.comment}"
                print(error_msg)  # Always show failed trades
                self.logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"❌ Error executing trade: {e}"
            print(error_msg)
            self.logger.error(error_msg)

def main():
    print("🚀 STARTING LIVE TRADING SYSTEM")
    print("=" * 60)
    
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
    
    if debug_mode:
        print("🔍 DEBUG MODE ENABLED - Detailed output will be shown")
    else:
        print("🤫 NORMAL MODE - Minimal output (only trades will be shown)")
    
    # Initialize performance tracker
    tracker = PerformanceTracker(magic_number=234567)
    
    # Show current performance
    print(tracker.generate_performance_report())
    
    # Safety confirmation
    confirmation = input("""
⚠️  LIVE TRADING WARNING ⚠️
This will trade with REAL MONEY!
Type 'GO' to continue: """)
    
    if confirmation != "GO":
        print("❌ Trading cancelled.")
        return
    
    # Start live trading with chosen mode
    bot = DebugLiveTradingBot(debug_mode=debug_mode)
    
    if not bot.connection.connect():
        print("❌ Failed to connect to MT5")
        return
        
    if bot.predictor.model is None:
        print("❌ ML Model not loaded")
        return
    
    # Get initial balance
    account_info = mt5.account_info()
    if account_info:
        bot.starting_balance = account_info.balance
        bot.current_balance = account_info.balance
        print(f"💰 Starting Balance: ${bot.starting_balance:,.2f}")
    
    bot.is_trading = True
    
    try:
        if debug_mode:
            print("🔍 Debug trading started. Press Ctrl+C to stop...")
        else:
            print("🎯 Live trading started. Press Ctrl+C to stop...")
        
        while bot.is_trading:
            bot.trading_cycle()
            
            if debug_mode:
                print(f"\n⏳ Waiting 30 seconds...")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Show final performance
        bot.stop_trading()
        print(tracker.generate_performance_report())

if __name__ == "__main__":
    main()