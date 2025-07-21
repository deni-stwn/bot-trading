import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import schedule
import pandas as pd

from mt5_integration.connection import Connection
from mt5_integration.orders import Orders
from data.market_data import MarketData
from bot.strategy import Strategy
from bot.risk_management import RiskManagement

class TradingBot:
    def __init__(self, login: int, password: str, server: str):
        self.connection = Connection(login, password, server)
        self.orders = Orders(self.connection)
        self.market_data = MarketData(self.connection)
        self.strategy = Strategy()
        self.risk_management = RiskManagement()
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.symbol = "BTCUSDm"
        self.timeframe = 5  # 5 minutes
        self.magic_number = 234000
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        
    def connect(self) -> bool:
        """Connect to MetaTrader 5"""
        if self.connection.connect():
            account_info = self.orders.get_account_info()
            if "error" not in account_info:
                self.start_balance = account_info["balance"]
                self.logger.info(f"Bot connected. Starting balance: {self.start_balance}")
                return True
        return False
    
    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5"""
        self.running = False
        self.connection.disconnect()
        self.logger.info("Bot disconnected")
    
    def start_trading(self) -> None:
        """Start the automated trading loop"""
        if not self.connection.is_connected():
            self.logger.error("Not connected to MT5. Cannot start trading.")
            return
        
        self.running = True
        self.logger.info("Starting automated trading...")
        
        # Schedule market analysis every minute
        schedule.every(1).minutes.do(self._trading_cycle)
        
        # Schedule daily performance report
        schedule.every().day.at("00:00").do(self._daily_report)
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
                
                # Check connection every 5 minutes
                if datetime.now().second == 0 and datetime.now().minute % 5 == 0:
                    self._check_connection()
                    
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
        except Exception as e:
            self.logger.error(f"Unexpected error in trading loop: {e}")
        finally:
            self.disconnect()
    
    def stop_trading(self) -> None:
        """Stop the automated trading"""
        self.running = False
        self.logger.info("Trading stop requested")
    
    def _trading_cycle(self) -> None:
        """Main trading cycle"""
        try:
            self.logger.debug("Executing trading cycle...")
            
            # Check if market is open
            if not self._is_market_open():
                self.logger.debug("Market is closed")
                return
            
            # Get market data
            market_data = self.market_data.get_rates(
                symbol=self.symbol,
                timeframe=self.timeframe,
                count=200
            )
            
            if market_data is None or len(market_data) < 100:
                self.logger.warning("Insufficient market data")
                return
            
            # Analyze market
            analysis_results = self.strategy.analyze_market(market_data)
            if "error" in analysis_results:
                self.logger.warning(f"Analysis error: {analysis_results['error']}")
                return
            
            # Make trading decision
            decision = self.strategy.make_decision(analysis_results)
            
            # Apply risk management
            risk_assessment = self.risk_management.assess_risk(decision, analysis_results)
            if not risk_assessment["allow_trade"]:
                self.logger.info(f"Trade blocked by risk management: {risk_assessment['reason']}")
                return
            
            # Execute trade if signal is not HOLD
            if decision["signal"] != "HOLD":
                self._execute_trading_decision(decision)
            
            # Monitor existing positions
            self._monitor_positions()
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def _execute_trading_decision(self, decision: Dict) -> None:
        """Execute trading decision"""
        try:
            self.logger.info(f"Executing {decision['signal']} signal")
            self.logger.info(f"Confluences: {', '.join(decision['confluences'])}")
            
            # Place order
            order_result = self.orders.place_order(
                symbol=self.symbol,
                order_type=decision["signal"],
                volume=decision["position_size"],
                sl=decision["stop_loss"],
                tp=decision["take_profit"],
                comment=f"AutoBot-{decision['signal']}",
                magic=self.magic_number
            )
            
            if order_result["success"]:
                self.total_trades += 1
                self.logger.info(f"Order placed successfully: {order_result['order_id']}")
                
                # Execute trade through strategy
                trade_result = self.strategy.execute_trade(decision)
                if trade_result["status"] == "trade_executed":
                    self.logger.info(f"Trade executed: {trade_result['trade_id']}")
            else:
                self.logger.error(f"Order failed: {order_result['error']}")
                
        except Exception as e:
            self.logger.error(f"Error executing trading decision: {e}")
    
    def _monitor_positions(self) -> None:
        """Monitor and manage existing positions"""
        try:
            positions_info = self.orders.list_open_orders(symbol=self.symbol)
            
            if "error" in positions_info:
                self.logger.warning(f"Error getting positions: {positions_info['error']}")
                return
            
            # Check open positions
            for position in positions_info["open_positions"]:
                if position["magic"] == self.magic_number:
                    self._manage_position(position)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def _manage_position(self, position: Dict) -> None:
        """Manage individual position"""
        try:
            # Update profit tracking
            if position["profit"] > 0 and position["ticket"] not in [t.get("position_id") for t in self.strategy.trades_history]:
                self.winning_trades += 1
                self.total_profit += position["profit"]
            
            # Implement trailing stop or other position management logic
            # This can be enhanced based on specific requirements
            
        except Exception as e:
            self.logger.error(f"Error managing position {position['ticket']}: {e}")
    
    def _check_connection(self) -> None:
        """Check and maintain connection"""
        if not self.connection.is_connected():
            self.logger.warning("Connection lost. Attempting to reconnect...")
            if self.connection.reconnect():
                self.logger.info("Reconnected successfully")
            else:
                self.logger.error("Failed to reconnect")
                self.stop_trading()
    
    def _is_market_open(self) -> bool:
        """Check if market is open for trading"""
        try:
            # Get current market data to check if market is active
            tick = self.market_data.get_current_tick(self.symbol)
            return tick is not None
        except Exception:
            return False
    
    def _daily_report(self) -> None:
        """Generate daily performance report"""
        try:
            account_info = self.orders.get_account_info()
            if "error" in account_info:
                return
            
            current_balance = account_info["balance"]
            daily_profit = current_balance - self.start_balance
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            self.logger.info("="*50)
            self.logger.info("DAILY PERFORMANCE REPORT")
            self.logger.info("="*50)
            self.logger.info(f"Starting Balance: {self.start_balance:.2f}")
            self.logger.info(f"Current Balance: {current_balance:.2f}")
            self.logger.info(f"Daily Profit: {daily_profit:.2f}")
            self.logger.info(f"Total Trades: {self.total_trades}")
            self.logger.info(f"Winning Trades: {self.winning_trades}")
            self.logger.info(f"Win Rate: {win_rate:.1f}%")
            self.logger.info(f"Total Profit: {self.total_profit:.2f}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        try:
            account_info = self.orders.get_account_info()
            if "error" in account_info:
                return {"error": account_info["error"]}
            
            current_balance = account_info["balance"]
            total_return = ((current_balance - self.start_balance) / self.start_balance * 100) if self.start_balance > 0 else 0
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            return {
                "start_balance": self.start_balance,
                "current_balance": current_balance,
                "total_return": total_return,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": win_rate,
                "total_profit": self.total_profit,
                "equity": account_info["equity"],
                "free_margin": account_info["free_margin"]
            }
            
        except Exception as e:
            return {"error": str(e)}