import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Dict, List
import matplotlib.pyplot as plt
import os

class PerformanceTracker:
    def __init__(self, magic_number: int):
        self.magic_number = magic_number
        self.performance_file = "data/performance_history.json"
        self.metrics_history = []
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Load existing data
        self.load_performance_history()
    
    def record_trade(self, trade_info: Dict):
        """Record a new trade"""
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "ticket": trade_info.get("ticket"),
            "symbol": trade_info.get("symbol"),
            "signal": trade_info.get("signal"),
            "confidence": trade_info.get("confidence"),
            "entry_price": trade_info.get("entry_price"),
            "sl_price": trade_info.get("sl_price"),
            "tp_price": trade_info.get("tp_price"),
            "lot_size": trade_info.get("lot_size")
        }
        
        self.metrics_history.append(trade_record)
        self.save_performance_history()
    
    def get_daily_performance(self) -> Dict:
        """Get today's performance metrics"""
        try:
            # Get today's deals
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            deals = mt5.history_deals_get(today_start, datetime.now())
            if not deals:
                return {"trades": 0, "pnl": 0, "win_rate": 0}
            
            # Filter our deals
            our_deals = [deal for deal in deals if deal.magic == self.magic_number]
            
            # Group by position_id
            positions = {}
            for deal in our_deals:
                pos_id = deal.position_id
                if pos_id not in positions:
                    positions[pos_id] = []
                positions[pos_id].append(deal)
            
            # Calculate metrics
            total_pnl = 0
            winning_trades = 0
            total_trades = 0
            
            for pos_id, pos_deals in positions.items():
                if len(pos_deals) >= 2:  # Complete trade
                    entry_deal = min(pos_deals, key=lambda x: x.time)
                    exit_deal = max(pos_deals, key=lambda x: x.time)
                    
                    # Calculate P&L
                    if entry_deal.type == mt5.DEAL_TYPE_BUY:
                        pnl = (exit_deal.price - entry_deal.price) * entry_deal.volume
                    else:
                        pnl = (entry_deal.price - exit_deal.price) * entry_deal.volume
                    
                    total_pnl += pnl
                    total_trades += 1
                    
                    if pnl > 0:
                        winning_trades += 1
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                "trades": total_trades,
                "pnl": total_pnl,
                "win_rate": win_rate,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades
            }
            
        except Exception as e:
            print(f"Error getting daily performance: {e}")
            return {"trades": 0, "pnl": 0, "win_rate": 0}
    
    def get_weekly_performance(self) -> Dict:
        """Get this week's performance"""
        try:
            week_start = datetime.now() - timedelta(days=7)
            
            deals = mt5.history_deals_get(week_start, datetime.now())
            if not deals:
                return self._empty_performance()
            
            return self._calculate_performance_metrics(deals)
            
        except Exception as e:
            print(f"Error getting weekly performance: {e}")
            return self._empty_performance()
    
    def get_monthly_performance(self) -> Dict:
        """Get this month's performance"""
        try:
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            deals = mt5.history_deals_get(month_start, datetime.now())
            if not deals:
                return self._empty_performance()
            
            return self._calculate_performance_metrics(deals)
            
        except Exception as e:
            print(f"Error getting monthly performance: {e}")
            return self._empty_performance()
    
    def _calculate_performance_metrics(self, deals) -> Dict:
        """Calculate comprehensive performance metrics"""
        our_deals = [deal for deal in deals if deal.magic == self.magic_number]
        
        if not our_deals:
            return self._empty_performance()
        
        # Group by position
        positions = {}
        for deal in our_deals:
            pos_id = deal.position_id
            if pos_id not in positions:
                positions[pos_id] = []
            positions[pos_id].append(deal)
        
        # Calculate metrics
        total_pnl = 0
        winning_trades = 0
        losing_trades = 0
        total_trades = 0
        profits = []
        losses = []
        trade_durations = []
        
        for pos_id, pos_deals in positions.items():
            if len(pos_deals) >= 2:
                entry_deal = min(pos_deals, key=lambda x: x.time)
                exit_deal = max(pos_deals, key=lambda x: x.time)
                
                # Calculate P&L
                if entry_deal.type == mt5.DEAL_TYPE_BUY:
                    pnl = (exit_deal.price - entry_deal.price) * entry_deal.volume * 100000
                else:
                    pnl = (entry_deal.price - exit_deal.price) * entry_deal.volume * 100000
                
                total_pnl += pnl
                total_trades += 1
                
                # Duration
                duration = exit_deal.time - entry_deal.time
                trade_durations.append(duration)
                
                if pnl > 0:
                    winning_trades += 1
                    profits.append(pnl)
                else:
                    losing_trades += 1
                    losses.append(abs(pnl))
        
        # Calculate advanced metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = (sum(profits) / sum(losses)) if losses else float('inf')
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Risk metrics
        returns = [p for p in profits] + [-l for l in losses]
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if returns else 0
        max_drawdown = self._calculate_max_drawdown(returns) if returns else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_duration_minutes": avg_duration / 60 if avg_duration else 0,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    
    def _empty_performance(self) -> Dict:
        """Return empty performance metrics"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_duration_minutes": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        return np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        daily = self.get_daily_performance()
        weekly = self.get_weekly_performance()
        monthly = self.get_monthly_performance()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    TRADING PERFORMANCE REPORT                ║
╠══════════════════════════════════════════════════════════════╣

📊 TODAY'S PERFORMANCE:
   Trades: {daily['trades']}
   P&L: ${daily['pnl']:.2f}
   Win Rate: {daily['win_rate']:.1f}%

📈 WEEKLY PERFORMANCE:
   Trades: {weekly['total_trades']}
   Winning: {weekly['winning_trades']} | Losing: {weekly['losing_trades']}
   Win Rate: {weekly['win_rate']:.1f}%
   Total P&L: ${weekly['total_pnl']:.2f}
   Avg Win: ${weekly['avg_win']:.2f} | Avg Loss: ${weekly['avg_loss']:.2f}
   Profit Factor: {weekly['profit_factor']:.2f}

📊 MONTHLY PERFORMANCE:
   Trades: {monthly['total_trades']}
   Win Rate: {monthly['win_rate']:.1f}%
   Total P&L: ${monthly['total_pnl']:.2f}
   Sharpe Ratio: {monthly['sharpe_ratio']:.3f}
   Max Drawdown: ${monthly['max_drawdown']:.2f}
   Avg Trade Duration: {monthly['avg_duration_minutes']:.1f} minutes

╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
    
    def save_performance_history(self):
        """Save performance history to file"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"Error saving performance history: {e}")
    
    def load_performance_history(self):
        """Load performance history from file"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    self.metrics_history = json.load(f)
        except Exception as e:
            print(f"Error loading performance history: {e}")
            self.metrics_history = []