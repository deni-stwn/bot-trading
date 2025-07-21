import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class RiskManager:
    def __init__(self, max_daily_loss: float = 0.05, max_positions: int = 3):
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.max_risk_per_trade = 0.02
        self.max_correlation_exposure = 0.1
        
    def check_risk_limits(self, symbol: str, signal: str, lot_size: float) -> Dict:
        """Comprehensive risk check before placing trade"""
        checks = {
            "allow_trade": True,
            "reasons": [],
            "risk_score": 0
        }
        
        # 1. Daily loss check
        if not self._check_daily_loss():
            checks["allow_trade"] = False
            checks["reasons"].append("Daily loss limit exceeded")
        
        # 2. Position count check
        if not self._check_position_count():
            checks["allow_trade"] = False
            checks["reasons"].append("Maximum positions reached")
        
        # 3. Correlation check
        correlation_risk = self._check_correlation_risk(symbol, signal)
        if correlation_risk > 0.8:
            checks["allow_trade"] = False
            checks["reasons"].append("High correlation risk")
        
        # 4. Margin check
        if not self._check_margin_requirements(symbol, lot_size):
            checks["allow_trade"] = False
            checks["reasons"].append("Insufficient margin")
        
        # 5. Volatility check
        volatility_risk = self._check_volatility_risk(symbol)
        if volatility_risk > 0.9:
            checks["allow_trade"] = False
            checks["reasons"].append("Extreme volatility")
        
        # Calculate overall risk score
        checks["risk_score"] = self._calculate_risk_score(
            correlation_risk, volatility_risk
        )
        
        return checks
    
    def _check_daily_loss(self) -> bool:
        """Check daily loss limit"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return False
            
            # Get today's starting balance (simplified)
            current_balance = account_info.balance
            equity = account_info.equity
            
            # If equity is significantly lower than balance, we might be at risk
            if equity < current_balance * (1 - self.max_daily_loss):
                return False
            
            return True
        except:
            return False
    
    def _check_position_count(self) -> bool:
        """Check maximum position count"""
        try:
            positions = mt5.positions_get()
            return len(positions or []) < self.max_positions
        except:
            return False
    
    def _check_correlation_risk(self, symbol: str, signal: str) -> float:
        """Check correlation risk with existing positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0
            
            # Simplified correlation check
            same_direction_positions = 0
            for pos in positions:
                if ((pos.type == mt5.POSITION_TYPE_BUY and signal == "BUY") or
                    (pos.type == mt5.POSITION_TYPE_SELL and signal == "SELL")):
                    same_direction_positions += 1
            
            return same_direction_positions / len(positions)
        except:
            return 0
    
    def _check_margin_requirements(self, symbol: str, lot_size: float) -> bool:
        """Check if we have enough margin"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return False
            
            # Calculate required margin
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False
            
            margin_required = lot_size * symbol_info.margin_initial
            free_margin = account_info.margin_free
            
            return free_margin > margin_required * 2  # 200% safety margin
        except:
            return False
    
    def _check_volatility_risk(self, symbol: str) -> float:
        """Check current volatility risk"""
        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
            if rates is None:
                return 0.5
            
            df = pd.DataFrame(rates)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Normalize volatility (simplified)
            return min(volatility * 100, 1.0)
        except:
            return 0.5
    
    def _calculate_risk_score(self, correlation_risk: float, volatility_risk: float) -> float:
        """Calculate overall risk score"""
        return (correlation_risk + volatility_risk) / 2

    def calculate_optimal_position_size(self, symbol: str, entry_price: float, 
                                      sl_price: float) -> float:
        """Calculate optimal position size based on risk"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return 0.01
            
            # Risk amount
            risk_amount = account_info.balance * self.max_risk_per_trade
            
            # Calculate pip difference
            pip_diff = abs(entry_price - sl_price)
            
            # Get pip value
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Calculate lot size
            lot_size = risk_amount / (pip_diff * 100000)  # Simplified
            
            # Apply constraints
            return max(0.01, min(lot_size, 1.0))
            
        except:
            return 0.01