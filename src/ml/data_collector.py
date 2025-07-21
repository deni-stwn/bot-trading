import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
import pickle
import logging
from typing import Optional, Dict, List
from mt5_integration.connection import Connection

class DataCollector:
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
        # Data storage paths
        self.data_dir = "data/historical"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Timeframes untuk collecting data
        self.timeframes = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1
        }

    def connect(self) -> bool:
        """Connect to MT5"""
        try:
            if not mt5.initialize():
                self.logger.error("Failed to initialize MT5")
                return False
            
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                self.logger.error(f"Login failed: {error}")
                return False
            
            self.logger.info("Connected to MT5 for data collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    def collect_training_data(self, symbol: str, days_back: int = 365, 
                            timeframe: int = 5) -> Optional[pd.DataFrame]:
        """
        Collect historical data for ML training
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDm")
            days_back: Number of days to collect (default: 1 year)
            timeframe: Timeframe in minutes (default: 5M)
        """
        try:
            if not self.connect():
                return None
            
            self.logger.info(f"Collecting {days_back} days of {symbol} data on {timeframe}M timeframe")
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get timeframe constant
            tf = self.timeframes.get(timeframe)
            if tf is None:
                self.logger.error(f"Unsupported timeframe: {timeframe}")
                return None
            
            # Calculate number of bars needed
            # Approximate bars per day based on timeframe
            bars_per_day = {
                1: 1440,    # 1M
                5: 288,     # 5M  
                15: 96,     # 15M
                30: 48,     # 30M
                60: 24,     # 1H
                240: 6,     # 4H
                1440: 1     # 1D
            }
            
            total_bars = days_back * bars_per_day.get(timeframe, 288)
            
            # Collect data in chunks (MT5 has limitations)
            all_data = []
            chunk_size = 10000  # Max bars per request
            
            for start_pos in range(0, total_bars, chunk_size):
                count = min(chunk_size, total_bars - start_pos)
                
                self.logger.info(f"Collecting chunk: {start_pos} to {start_pos + count}")
                
                rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, count)
                
                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No data for chunk starting at {start_pos}")
                    continue
                
                chunk_df = pd.DataFrame(rates)
                all_data.append(chunk_df)
            
            if not all_data:
                self.logger.error("No data collected")
                return None
            
            # Combine all chunks
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by time
            combined_df['time'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['time']).reset_index(drop=True)
            
            self.logger.info(f"✅ Collected {len(combined_df)} bars of {symbol} data")
            
            # Save to file
            filename = f"{symbol}_{timeframe}M_{days_back}days_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            combined_df.to_csv(filepath, index=False)
            self.logger.info(f"Data saved to {filepath}")
            
            mt5.shutdown()
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            mt5.shutdown()
            return None

    def collect_multiple_symbols(self, symbols: List[str], 
                                days_back: int = 180) -> Dict[str, pd.DataFrame]:
        """Collect data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Collecting data for {symbol}")
            data = self.collect_training_data(symbol, days_back)
            
            if data is not None:
                results[symbol] = data
                self.logger.info(f"✅ {symbol}: {len(data)} bars collected")
            else:
                self.logger.error(f"❌ Failed to collect data for {symbol}")
        
        return results

    def collect_realtime_features(self, symbol: str, timeframe: int = 5, 
                                 bars_count: int = 200) -> Optional[pd.DataFrame]:
        """
        Collect recent data for real-time prediction
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            bars_count: Number of recent bars to collect
        """
        try:
            if not self.connect():
                return None
            
            tf = self.timeframes.get(timeframe)
            if tf is None:
                return None
            
            # Get recent data
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars_count)
            
            if rates is None:
                self.logger.error(f"Failed to get realtime data for {symbol}")
                mt5.shutdown()
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            mt5.shutdown()
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting realtime data: {e}")
            mt5.shutdown()
            return None

    def update_historical_data(self, symbol: str, timeframe: int = 5) -> bool:
        """Update existing historical data with new bars"""
        try:
            # Find latest data file
            data_files = [f for f in os.listdir(self.data_dir) 
                         if f.startswith(f"{symbol}_{timeframe}M")]
            
            if not data_files:
                self.logger.warning("No existing data file found, collecting fresh data")
                self.collect_training_data(symbol, days_back=365, timeframe=timeframe)
                return True
            
            # Load latest file
            latest_file = sorted(data_files)[-1]
            filepath = os.path.join(self.data_dir, latest_file)
            existing_data = pd.read_csv(filepath)
            existing_data['time'] = pd.to_datetime(existing_data['time'])
            
            # Get last timestamp
            last_time = existing_data['time'].max()
            self.logger.info(f"Last data timestamp: {last_time}")
            
            # Calculate days since last update
            days_since = (datetime.now() - last_time).days
            
            if days_since < 1:
                self.logger.info("Data is up to date")
                return True
            
            # Collect new data
            new_data = self.collect_training_data(symbol, days_back=days_since + 1, timeframe=timeframe)
            
            if new_data is None:
                return False
            
            # Filter only new bars
            new_bars = new_data[new_data['time'] > last_time]
            
            if len(new_bars) > 0:
                # Append new data
                updated_data = pd.concat([existing_data, new_bars], ignore_index=True)
                updated_data = updated_data.drop_duplicates(subset=['time']).sort_values('time')
                
                # Save updated data
                new_filename = f"{symbol}_{timeframe}M_updated_{datetime.now().strftime('%Y%m%d')}.csv"
                new_filepath = os.path.join(self.data_dir, new_filename)
                updated_data.to_csv(new_filepath, index=False)
                
                self.logger.info(f"✅ Added {len(new_bars)} new bars to {symbol} data")
                return True
            else:
                self.logger.info("No new bars to add")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating historical data: {e}")
            return False

    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """Validate data quality for ML training"""
        validation_report = {
            "total_rows": len(data),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "date_range": {
                "start": data['time'].min() if 'time' in data.columns else None,
                "end": data['time'].max() if 'time' in data.columns else None
            },
            "price_anomalies": [],
            "volume_anomalies": []
        }
        
        # Check for price anomalies (extreme movements)
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            extreme_returns = returns[abs(returns) > 0.2]  # >20% movements
            validation_report["price_anomalies"] = len(extreme_returns)
        
        # Check for volume anomalies
        volume_cols = ['tick_volume', 'real_volume', 'volume']
        for vol_col in volume_cols:
            if vol_col in data.columns:
                zero_volume = (data[vol_col] == 0).sum()
                validation_report["volume_anomalies"].append({
                    "column": vol_col,
                    "zero_volume_bars": zero_volume
                })
        
        return validation_report

    def prepare_training_dataset(self, symbol: str, timeframe: int = 5,
                               min_samples: int = 1000) -> Optional[pd.DataFrame]:
        """Prepare clean dataset for ML training"""
        try:
            # Load or collect data
            data_files = [f for f in os.listdir(self.data_dir) 
                         if f.startswith(f"{symbol}_{timeframe}M")]
            
            if data_files:
                # Load existing data
                latest_file = sorted(data_files)[-1]
                filepath = os.path.join(self.data_dir, latest_file)
                data = pd.read_csv(filepath)
                self.logger.info(f"Loaded existing data: {len(data)} rows")
            else:
                # Collect new data
                self.logger.info("No existing data found, collecting fresh data")
                data = self.collect_training_data(symbol, days_back=365, timeframe=timeframe)
                
                if data is None:
                    return None
            
            # Validate data quality
            validation = self.validate_data_quality(data)
            self.logger.info(f"Data validation: {validation}")
            
            # Clean data
            data = data.dropna()  # Remove missing values
            data = data.drop_duplicates(subset=['time'])  # Remove duplicates
            
            # Filter minimum samples
            if len(data) < min_samples:
                self.logger.error(f"Insufficient data: {len(data)} < {min_samples}")
                return None
            
            # Sort by time
            data['time'] = pd.to_datetime(data['time'])
            data = data.sort_values('time').reset_index(drop=True)
            
            self.logger.info(f"✅ Prepared training dataset: {len(data)} clean samples")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing training dataset: {e}")
            return None

    def export_data_for_external_analysis(self, symbol: str, format: str = "csv"):
        """Export data for external analysis tools"""
        try:
            data_files = [f for f in os.listdir(self.data_dir) 
                         if f.startswith(f"{symbol}_")]
            
            if not data_files:
                self.logger.error("No data files found for export")
                return False
            
            latest_file = sorted(data_files)[-1]
            filepath = os.path.join(self.data_dir, latest_file)
            data = pd.read_csv(filepath)
            
            # Export in different formats
            export_dir = os.path.join(self.data_dir, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == "csv":
                export_path = os.path.join(export_dir, f"{symbol}_export_{timestamp}.csv")
                data.to_csv(export_path, index=False)
                
            elif format.lower() == "excel":
                export_path = os.path.join(export_dir, f"{symbol}_export_{timestamp}.xlsx")
                data.to_excel(export_path, index=False)
                
            elif format.lower() == "pickle":
                export_path = os.path.join(export_dir, f"{symbol}_export_{timestamp}.pkl")
                data.to_pickle(export_path)
            
            self.logger.info(f"✅ Data exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False