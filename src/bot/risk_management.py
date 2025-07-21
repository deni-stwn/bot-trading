import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config.settings import RISK_CONFIG

class RiskManagement:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        
        # Risk parameters from config
        self.max_daily_loss = RISK_CONFIG["max_daily_loss"]
        self.max_drawdown = RISK_CONFIG["max_drawdown"]
        self.max_risk_per_trade = RISK_CONFIG["max_risk_per_trade"]
        self.min_profit_factor = RISK_CONFIG["min_profit_factor"]

    def assess_risk(self, decision: Dict, analysis_results: Dict) -> Dict:
        """
        Assess risk for a trading decision
        
        Returns:
            Dict with allow_trade boolean and reason
        """
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                return {
                    "allow_trade": False,
                    "reason": f"Daily loss limit reached: {self.daily_pnl:.2f}"
                }
            
            # Check maximum drawdown
            if self.current_drawdown >= self.max_drawdown:
                return {
                    "allow_trade": False,
                    "reason": f"Maximum drawdown reached: {self.current_drawdown:.1f}%"
                }
            
            # Check trade frequency (prevent overtrading)
            recent_trades = self._get_recent_trades(minutes=60)
            if len(recent_trades) >= 5:
                return {
                    "allow_trade": False,
                    "reason": "Too many trades in the last hour"
                }
            
            # Check signal strength
            if decision.get("strength", 0) < 0.6:
                return {
                    "allow_trade": False,
                    "reason": f"Signal strength too low: {decision.get('strength', 0):.2f}"
                }
            
            # Check risk-reward ratio
            rr_ratio = decision.get("risk_reward_ratio", 0)
            if rr_ratio < self.min_profit_factor:
                return {
                    "allow_trade": False,
                    "reason": f"Risk-reward ratio too low: {rr_ratio:.2f}"
                }
            
            # Check market volatility
            volatility = analysis_results.get("volatility", {})
            if volatility.get("volatility_level") == "high" and decision.get("strength", 0) < 0.8:
                return {
                    "allow_trade": False,
                    "reason": "High volatility requires stronger signal"
                }
            
            # All checks passed
            return {
                "allow_trade": True,
                "reason": "Risk assessment passed"
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {
                "allow_trade": False,
                "reason": f"Risk assessment error: {str(e)}"
            }

    def calculate_position_size(self, account_balance: float, risk_distance: float, 
                              current_price: float) -> float:
        """
        Calculate optimal position size based on risk management
        """
        try:
            # Calculate risk amount (percentage of balance)
            risk_amount = account_balance * (self.max_risk_per_trade / 100)
            
            # Calculate position size based on risk distance
            if risk_distance <= 0:
                return 0.01  # Minimum position size
            
            # Position size = Risk Amount / (Risk Distance * Current Price)
            position_size = risk_amount / (risk_distance * current_price)
            
            # Apply constraints
            position_size = max(0.01, min(position_size, 1.0))  # Between 0.01 and 1.0
            
            return round(position_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        
        # Reset daily tracking at midnight
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            self.daily_trades.clear()
            self.daily_pnl = 0.0

    def update_drawdown(self, current_balance: float) -> None:
        """Update drawdown calculation"""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100

    def record_trade(self, trade_info: Dict) -> None:
        """Record trade for risk tracking"""
        trade_record = {
            "timestamp": datetime.now(),
            "signal": trade_info.get("signal"),
            "volume": trade_info.get("volume"),
            "entry_price": trade_info.get("entry_price"),
            "stop_loss": trade_info.get("stop_loss"),
            "take_profit": trade_info.get("take_profit")
        }
        self.daily_trades.append(trade_record)

    def _get_recent_trades(self, minutes: int) -> List:
        """Get trades from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [trade for trade in self.daily_trades if trade["timestamp"] > cutoff_time]

    def get_risk_status(self) -> Dict:
        """Get current risk management status"""
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades_count": len(self.daily_trades),
            "current_drawdown": self.current_drawdown,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown": self.max_drawdown,
            "trades_last_hour": len(self._get_recent_trades(60))
        }