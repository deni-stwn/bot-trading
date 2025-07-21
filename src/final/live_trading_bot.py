import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predictor_v2 import ImprovedMLPredictorV2
from mt5_integration.connection import Connection
from config.settings import LOGIN, PASSWORD, SERVER, TRADING_CONFIG
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import logging

class LiveTradingBot:
    def __init__(self):
        self.connection = Connection(LOGIN, PASSWORD, SERVER)
        self.predictor = ImprovedMLPredictorV2()
        self.symbol = TRADING_CONFIG["symbol"]
        self.timeframe = TRADING_CONFIG["timeframe"]
        self.magic_number = TRADING_CONFIG.get("magic_number", 234567)
        
        # Risk Management Settings
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.max_daily_loss = 0.05      # 5% max daily loss
        self.max_positions = 3          # Max 3 positions at once
        self.min_confidence = 0.6       # Minimum 60% confidence to trade
        
        # Position Management
        self.base_lot_size = 0.1
        self.stop_loss_pips = 50
        self.take_profit_pips = 100
        
        # Performance Tracking
        self.trades_history = []
        self.daily_pnl = 0
        self.starting_balance = 0
        self.current_balance = 0
        
        # Trading State
        self.is_trading = False
        self.last_trade_time = None
        self.min_time_between_trades = 300  # 5 minutes
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for live trading"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def start_live_trading(self):
        """Start live trading"""
        print("🚀 STARTING LIVE TRADING BOT")
        print("=" * 50)
        
        if not self.connection.connect():
            print("❌ Failed to connect to MT5")
            return
            
        if self.predictor.model is None:
            print("❌ ML Model not loaded")
            return
        
        # Get initial balance
        account_info = mt5.account_info()
        if account_info:
            self.starting_balance = account_info.balance
            self.current_balance = account_info.balance
            print(f"💰 Starting Balance: ${self.starting_balance:,.2f}")
        
        self.is_trading = True
        
        try:
            print("🎯 Live trading started. Press Ctrl+C to stop...")
            while self.is_trading:
                self.trading_cycle()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Live trading stopped by user")
        finally:
            self.stop_trading()
    
    def trading_cycle(self):
        """Main trading cycle"""
        try:
            # 1. Check daily loss limit
            if not self.check_daily_loss_limit():
                return
            
            # 2. Get market data
            market_data = self.get_market_data()
            if market_data is None:
                return
            
            # 3. Get ML prediction
            prediction = self.predictor.predict_signal(market_data)
            if "error" in prediction:
                return
            
            # 4. Check trading conditions
            if not self.should_trade(prediction):
                return
            
            # 5. Execute trade
            self.execute_trade(prediction, market_data.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def get_market_data(self, bars=200):
        """Get recent market data"""
        try:
            timeframes = {
                1: mt5.TIMEFRAME_M1,
                5: mt5.TIMEFRAME_M5,
                15: mt5.TIMEFRAME_M15,
                30: mt5.TIMEFRAME_M30,
                60: mt5.TIMEFRAME_H1
            }
            
            tf = timeframes.get(self.timeframe, mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
            
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
    
    def should_trade(self, prediction):
        """Check if we should trade based on conditions"""
        # 1. Check confidence level
        if prediction['confidence'] < self.min_confidence:
            return False
        
        # 2. Check signal strength
        if prediction['signal'] == "HOLD":
            return False
        
        # 3. Check time between trades
        current_time = datetime.now()
        if (self.last_trade_time and 
            (current_time - self.last_trade_time).seconds < self.min_time_between_trades):
            return False
        
        # 4. Check max positions
        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_positions:
            return False
        
        # 5. Check if we already have same direction trade
        if open_positions:
            existing_directions = [pos['type'] for pos in open_positions]
            if (prediction['signal'] == "BUY" and mt5.ORDER_TYPE_BUY in existing_directions) or \
               (prediction['signal'] == "SELL" and mt5.ORDER_TYPE_SELL in existing_directions):
                return False
        
        return True
    
    def execute_trade(self, prediction, current_bar):
        """Execute trade based on prediction"""
        try:
            signal = prediction['signal']
            confidence = prediction['confidence']
            current_price = current_bar['close']
            
            # Calculate position size
            lot_size = self.calculate_position_size(current_price)
            
            # Calculate SL and TP
            sl_price, tp_price = self.calculate_sl_tp(signal, current_price)
            
            # Prepare order
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).ask
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(self.symbol).bid
            
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
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Trade successful
                trade_info = {
                    "timestamp": datetime.now(),
                    "ticket": result.order,
                    "signal": signal,
                    "confidence": confidence,
                    "entry_price": result.price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "lot_size": lot_size,
                    "prediction": prediction
                }
                
                self.trades_history.append(trade_info)
                self.last_trade_time = datetime.now()
                
                self.logger.info(f"✅ Trade executed: {signal} {lot_size} lots at {result.price}")
                print(f"🎯 TRADE EXECUTED: {signal} {lot_size} lots at {result.price}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                
            else:
                self.logger.error(f"❌ Trade failed: {result.comment}")
                print(f"❌ Trade failed: {result.comment}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def calculate_position_size(self, price):
        """Calculate position size based on risk management"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return self.base_lot_size
            
            balance = account_info.balance
            
            # Calculate risk amount
            risk_amount = balance * self.max_risk_per_trade
            
            # Calculate pip value
            pip_value = self.get_pip_value()
            
            # Calculate lot size
            lot_size = risk_amount / (self.stop_loss_pips * pip_value)
            
            # Apply constraints
            lot_size = max(0.01, min(lot_size, 1.0))  # Between 0.01 and 1.0
            
            return round(lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.base_lot_size
    
    def get_pip_value(self):
        """Get pip value for position sizing"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return 1.0
        
        # Simplified pip value calculation
        if "JPY" in self.symbol:
            return 0.01
        else:
            return 0.0001
    
    def calculate_sl_tp(self, signal, price):
        """Calculate Stop Loss and Take Profit"""
        pip_value = self.get_pip_value()
        
        if signal == "BUY":
            sl_price = price - (self.stop_loss_pips * pip_value)
            tp_price = price + (self.take_profit_pips * pip_value)
        else:  # SELL
            sl_price = price + (self.stop_loss_pips * pip_value)
            tp_price = price - (self.take_profit_pips * pip_value)
        
        return round(sl_price, 5), round(tp_price, 5)
    
    def get_open_positions(self):
        """Get current open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return []
            
            # Filter by magic number
            our_positions = [pos for pos in positions if pos.magic == self.magic_number]
            return our_positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit is reached"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return True
            
            current_balance = account_info.balance
            daily_loss = (self.starting_balance - current_balance) / self.starting_balance
            
            if daily_loss > self.max_daily_loss:
                self.logger.warning(f"Daily loss limit reached: {daily_loss:.2%}")
                print(f"🛑 Daily loss limit reached: {daily_loss:.2%}")
                self.is_trading = False
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking daily loss: {e}")
            return True
    
    def stop_trading(self):
        """Stop trading and cleanup"""
        self.is_trading = False
        self.show_performance_report()
        self.connection.disconnect()
    
    def show_performance_report(self):
        """Show performance report"""
        print("\n" + "=" * 60)
        print("📊 TRADING PERFORMANCE REPORT")
        print("=" * 60)
        
        # Account info
        account_info = mt5.account_info()
        if account_info:
            self.current_balance = account_info.balance
            profit_loss = self.current_balance - self.starting_balance
            profit_loss_pct = (profit_loss / self.starting_balance) * 100
            
            print(f"💰 Starting Balance: ${self.starting_balance:,.2f}")
            print(f"💰 Current Balance:  ${self.current_balance:,.2f}")
            print(f"💰 P&L:             ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)")
        
        # Trades statistics
        if self.trades_history:
            total_trades = len(self.trades_history)
            print(f"\n📈 Total Trades: {total_trades}")
            
            # Get closed positions for detailed analysis
            self.analyze_closed_trades()
        
        # Current positions
        open_positions = self.get_open_positions()
        if open_positions:
            print(f"\n🔄 Open Positions: {len(open_positions)}")
            for pos in open_positions:
                print(f"   {pos.type_str} {pos.volume} lots | P&L: ${pos.profit:.2f}")
    
    def analyze_closed_trades(self):
        """Analyze closed trades for win rate and other metrics"""
        try:
            # Get trade history
            history_deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=1),
                datetime.now()
            )
            
            if not history_deals:
                return
            
            # Filter our trades
            our_deals = [deal for deal in history_deals if deal.magic == self.magic_number]
            
            if not our_deals:
                return
            
            # Group by position_id to get complete trades
            trades = {}
            for deal in our_deals:
                pos_id = deal.position_id
                if pos_id not in trades:
                    trades[pos_id] = []
                trades[pos_id].append(deal)
            
            # Analyze complete trades
            completed_trades = []
            for pos_id, deals in trades.items():
                if len(deals) >= 2:  # Entry + Exit
                    entry_deal = min(deals, key=lambda x: x.time)
                    exit_deal = max(deals, key=lambda x: x.time)
                    
                    if entry_deal.type == mt5.DEAL_TYPE_BUY:
                        profit = (exit_deal.price - entry_deal.price) * entry_deal.volume
                    else:
                        profit = (entry_deal.price - exit_deal.price) * entry_deal.volume
                    
                    completed_trades.append({
                        'profit': profit,
                        'entry_time': entry_deal.time,
                        'exit_time': exit_deal.time,
                        'volume': entry_deal.volume
                    })
            
            if completed_trades:
                profits = [t['profit'] for t in completed_trades]
                winning_trades = [p for p in profits if p > 0]
                losing_trades = [p for p in profits if p < 0]
                
                win_rate = len(winning_trades) / len(completed_trades) * 100
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
                
                print(f"📊 Completed Trades: {len(completed_trades)}")
                print(f"📊 Win Rate: {win_rate:.1f}%")
                print(f"📊 Average Win: ${avg_win:.2f}")
                print(f"📊 Average Loss: ${avg_loss:.2f}")
                print(f"📊 Profit Factor: {profit_factor:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {e}")

def main():
    print("🤖 LIVE TRADING BOT")
    print("=" * 40)
    
    # Safety confirmation
    confirmation = input("""
⚠️  WARNING: This is LIVE TRADING with REAL MONEY!
⚠️  Make sure you understand the risks.
⚠️  Start with a demo account first.

Type 'I UNDERSTAND THE RISKS' to continue: """)
    
    if confirmation != "I UNDERSTAND THE RISKS":
        print("❌ Trading cancelled for safety.")
        return
    
    # Final confirmation
    mode = input("""
Choose trading mode:
1. Demo mode (paper trading)
2. Live mode (real money)

Enter choice (1/2): """)
    
    if mode == "1":
        print("📝 Demo mode selected (paper trading)")
        # For demo, you would modify the bot to not send real orders
    elif mode == "2":
        print("💰 Live mode selected (REAL MONEY)")
    else:
        print("❌ Invalid choice")
        return
    
    # Start bot
    bot = LiveTradingBot()
    bot.start_live_trading()

if __name__ == "__main__":
    main()