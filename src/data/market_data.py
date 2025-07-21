import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import time
import MetaTrader5 as mt5

class MarketData:
    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        
        # Timeframe mapping
        self.TIMEFRAMES = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }

    def get_rates(self, symbol: str, timeframe: int, count: int = 100, 
                  start_pos: int = 0) -> Optional[pd.DataFrame]:
        """
        Get historical price data from MT5
        """
        try:
            if not self.connection.is_connected():
                self.logger.error("Not connected to MT5")
                return None
            
            # Get timeframe constant
            tf = self.TIMEFRAMES.get(timeframe)
            if tf is None:
                self.logger.error(f"Unsupported timeframe: {timeframe}")
                return None
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
            
            if rates is None:
                self.logger.error(f"Failed to get rates for {symbol}")
                return None
            
            if len(rates) == 0:
                self.logger.warning(f"No data received for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Fix column names - MT5 uses different volume column names
            if 'tick_volume' not in df.columns and 'real_volume' in df.columns:
                df['tick_volume'] = df['real_volume']
            elif 'tick_volume' not in df.columns:
                # Create dummy tick_volume if not available
                df['tick_volume'] = 1  # Default volume
                self.logger.warning(f"No volume data available for {symbol}, using default")
            
            # Ensure we have required columns
            required_columns = ['time', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return None
            
            self.logger.debug(f"Retrieved {len(df)} bars for {symbol} {timeframe}M")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting rates: {e}")
            return None

    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get current tick data for symbol
        """
        try:
            if not self.connection.is_connected():
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'symbol': symbol,
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tick data: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol information
        """
        try:
            if not self.connection.is_connected():
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            return {
                'symbol': symbol,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'currency_margin': symbol_info.currency_margin,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_stops_level': symbol_info.trade_stops_level,
                'spread': symbol_info.spread,
                'visible': symbol_info.visible,
                'trade_mode': symbol_info.trade_mode
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None