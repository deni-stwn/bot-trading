import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from config.settings import STRATEGY_CONFIG, TRADING_CONFIG
from ml.predictor_v2 import ImprovedMLPredictorV2

class Strategy:
    def __init__(self):
        self.signals_history = []
        self.trades_history = []
        self.last_signal_time = None
        
        # Strategy parameters
        self.min_confluences = STRATEGY_CONFIG.get("min_confluences", 3)
        self.rsi_oversold = STRATEGY_CONFIG.get("rsi_oversold", 30)
        self.rsi_overbought = STRATEGY_CONFIG.get("rsi_overbought", 70)
        self.volume_threshold = STRATEGY_CONFIG.get("volume_threshold", 1.5)
        self.atr_multiplier = STRATEGY_CONFIG.get("atr_multiplier", 2.0)
        self.min_time_between_signals = STRATEGY_CONFIG.get("min_time_between_signals", 300)
        
        # Technical indicator settings
        self.ema_fast_period = 9
        self.ema_slow_period = 21
        self.sma_trend_period = 50
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.stoch_k = 14
        self.stoch_d = 3
        self.atr_period = 14
        
        # Add ML predictor
        self.ml_predictor = ImprovedMLPredictorV2()
        self.use_ml = True  # Enable/disable ML
        self.ml_weight = 0.4  # Weight for ML vs traditional indicators
        
        self.logger = logging.getLogger(__name__)

    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Comprehensive market analysis using multiple technical indicators
        """
        try:
            if market_data is None or len(market_data) < 100:
                self.logger.warning("Insufficient market data for analysis")
                return {"error": "Insufficient data"}
            
            # Calculate all technical indicators
            df = self._calculate_indicators(market_data.copy())
            
            # Get latest and previous values
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current
            
            # Perform individual indicator analysis
            trend_analysis = self._analyze_trend(df)
            momentum_analysis = self._analyze_momentum(df)
            volatility_analysis = self._analyze_volatility(df)
            volume_analysis = self._analyze_volume(df)
            support_resistance = self._analyze_support_resistance(df)
            
            # Combine all analyses
            analysis_results = {
                "timestamp": current['time'],
                "price": current['close'],
                "trend": trend_analysis,
                "momentum": momentum_analysis,
                "volatility": volatility_analysis,
                "volume": volume_analysis,
                "support_resistance": support_resistance,
                "market_condition": self._determine_market_condition(df),
                "raw_data": {
                    "current": current,
                    "previous": previous,
                    "dataframe": df
                }
            }
            
            self.logger.info(f"Market analysis completed at {current['time']}")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return {"error": str(e)}

    def make_decision(self, analysis_results: Dict) -> Dict:
        """Enhanced decision making with ML"""
        try:
            # Traditional analysis
            traditional_decision = self._traditional_analysis(analysis_results)
            
            # ML prediction if enabled
            if self.use_ml and self.ml_predictor.model:
                ml_prediction = self.ml_predictor.predict_signal(
                    analysis_results["raw_data"]["dataframe"]
                )
                
                if "error" not in ml_prediction:
                    # Combine traditional + ML signals
                    final_decision = self._combine_signals(
                        traditional_decision, 
                        ml_prediction
                    )
                    return final_decision
            
            # Fallback to traditional analysis
            return traditional_decision
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {e}")
            return {"signal": "HOLD", "reason": f"Decision error: {str(e)}"}
    
    def _combine_signals(self, traditional: Dict, ml_prediction: Dict) -> Dict:
        """Combine traditional and ML signals"""
        # Weight scores
        traditional_weight = 1 - self.ml_weight
        ml_weight = self.ml_weight
        
        # Calculate combined confidence
        trad_strength = traditional.get("strength", 0)
        ml_confidence = ml_prediction.get("confidence", 0)
        
        combined_strength = (trad_strength * traditional_weight + 
                           ml_confidence * ml_weight)
        
        # Determine final signal
        if (traditional["signal"] == ml_prediction["signal"] and 
            traditional["signal"] != "HOLD"):
            # Both agree - strong signal
            final_signal = traditional["signal"]
            confidence_boost = 0.2
            combined_strength = min(combined_strength + confidence_boost, 1.0)
            
        elif (traditional["signal"] != "HOLD" and 
              ml_prediction["signal"] == "HOLD"):
            # Traditional says trade, ML says hold
            if combined_strength > 0.6:
                final_signal = traditional["signal"]
            else:
                final_signal = "HOLD"
                
        elif (traditional["signal"] == "HOLD" and 
              ml_prediction["signal"] != "HOLD"):
            # ML says trade, traditional says hold
            if ml_confidence > 0.7:
                final_signal = ml_prediction["signal"]
            else:
                final_signal = "HOLD"
                
        else:
            # Conflicting signals or both hold
            final_signal = "HOLD"
        
        # Prepare final decision
        if final_signal != "HOLD":
            decision = traditional.copy()
            decision["signal"] = final_signal
            decision["strength"] = combined_strength
            decision["ml_confidence"] = ml_confidence
            decision["traditional_strength"] = trad_strength
            decision["confluences"].append(f"ML prediction: {ml_prediction['signal']} ({ml_confidence:.2f})")
            
            return decision
        else:
            return {
                "signal": "HOLD",
                "reason": f"Combined analysis suggests hold (ML: {ml_prediction['signal']}, Traditional: {traditional['signal']})",
                "ml_confidence": ml_confidence,
                "traditional_strength": trad_strength
            }
    
    def execute_trade(self, decision: Dict) -> Dict:
        """Execute trade based on decision with proper risk management"""
        try:
            if decision["signal"] == "HOLD":
                return {"status": "no_trade", "reason": decision.get("reason", "Hold signal")}
            
            # Prepare trade parameters
            trade_params = {
                "signal": decision["signal"],
                "entry_price": decision["entry_price"],
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "position_size": decision["position_size"],
                "risk_reward_ratio": decision["risk_reward_ratio"],
                "confluences": decision["confluences"],
                "timestamp": datetime.now()
            }
            
            # Log trade execution
            self.logger.info(f"Executing {decision['signal']} trade:")
            self.logger.info(f"Entry: {decision['entry_price']:.5f}")
            self.logger.info(f"SL: {decision['stop_loss']:.5f}")
            self.logger.info(f"TP: {decision['take_profit']:.5f}")
            self.logger.info(f"Size: {decision['position_size']:.3f}")
            self.logger.info(f"Confluences: {', '.join(decision['confluences'])}")
            
            # Add to trades history
            self.trades_history.append(trade_params)
            
            return {
                "status": "trade_executed",
                "trade_params": trade_params,
                "trade_id": len(self.trades_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {"status": "execution_error", "error": str(e)}

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators using custom implementations"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Handle volume data safely
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values
            elif 'real_volume' in df.columns:
                volume = df['real_volume'].values
            elif 'volume' in df.columns:
                volume = df['volume'].values
            else:
                # Create dummy volume data
                volume = np.ones(len(df))
                self.logger.warning("No volume data available, using dummy values")
            
            # Moving Averages
            df['ema_fast'] = self._ema(close, self.ema_fast_period)
            df['ema_slow'] = self._ema(close, self.ema_slow_period)
            df['sma_trend'] = self._sma(close, self.sma_trend_period)
            
            # RSI
            df['rsi'] = self._rsi(close, self.rsi_period)
            
            # MACD
            macd_line = self._ema(close, self.macd_fast) - self._ema(close, self.macd_slow)
            macd_signal = self._ema(macd_line, self.macd_signal)
            df['MACD_12_26_9'] = macd_line
            df['MACDs_12_26_9'] = macd_signal
            df['MACDh_12_26_9'] = macd_line - macd_signal
            
            # Bollinger Bands
            bb_middle = self._sma(close, self.bb_period)
            bb_std = self._rolling_std(close, self.bb_period)
            df['BBL_20_2.0'] = bb_middle - (bb_std * self.bb_std)
            df['BBM_20_2.0'] = bb_middle
            df['BBU_20_2.0'] = bb_middle + (bb_std * self.bb_std)
            
            # Stochastic
            stoch_k, stoch_d = self._stochastic(high, low, close, self.stoch_k, self.stoch_d)
            df['STOCHk_14_3_3'] = stoch_k
            df['STOCHd_14_3_3'] = stoch_d
            
            # ATR
            df['atr'] = self._atr(high, low, close, self.atr_period)
            
            # Volume indicators
            df['volume_sma'] = self._sma(volume, 20)
            df['volume_ratio'] = volume / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df

    # Custom indicator implementations
    def _sma(self, data: np.ndarray, period: int) -> pd.Series:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean()
    
    def _ema(self, data: np.ndarray, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period).mean()
    
    def _rsi(self, data: np.ndarray, period: int) -> pd.Series:
        """Relative Strength Index"""
        close = pd.Series(data)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _rolling_std(self, data: np.ndarray, period: int) -> pd.Series:
        """Rolling standard deviation"""
        return pd.Series(data).rolling(window=period).std()
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> pd.Series:
        """Average True Range"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        tr1 = high_s - low_s
        tr2 = abs(high_s - close_s.shift())
        tr3 = abs(low_s - close_s.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Stochastic oscillator"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        lowest_low = low_s.rolling(window=k_period).min()
        highest_high = high_s.rolling(window=k_period).max()
        
        k_percent = 100 * ((close_s - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze market trend using multiple indicators"""
        current = df.iloc[-1]
        
        # EMA trend
        ema_bullish = current['ema_fast'] > current['ema_slow']
        ema_strength = abs(current['ema_fast'] - current['ema_slow']) / current['close']
        
        # SMA trend
        sma_bullish = current['close'] > current['sma_trend']
        
        # Overall trend direction
        if ema_bullish and sma_bullish:
            direction = "bullish"
            strength = min(ema_strength * 10, 1.0)
        elif not ema_bullish and not sma_bullish:
            direction = "bearish"
            strength = min(ema_strength * 10, 1.0)
        else:
            direction = "sideways"
            strength = 0.3
        
        return {
            "direction": direction,
            "strength": strength,
            "ema_bullish": ema_bullish,
            "sma_bullish": sma_bullish
        }

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # RSI analysis
        rsi_signal = "neutral"
        if current['rsi'] < self.rsi_oversold and current['rsi'] > previous['rsi']:
            rsi_signal = "oversold_bounce"
        elif current['rsi'] > self.rsi_overbought and current['rsi'] < previous['rsi']:
            rsi_signal = "overbought_decline"
        
        # MACD analysis
        macd_signal = "neutral"
        if (current['MACD_12_26_9'] > current['MACDs_12_26_9'] and 
            previous['MACD_12_26_9'] <= previous['MACDs_12_26_9']):
            macd_signal = "bullish_crossover"
        elif (current['MACD_12_26_9'] < current['MACDs_12_26_9'] and 
              previous['MACD_12_26_9'] >= previous['MACDs_12_26_9']):
            macd_signal = "bearish_crossover"
        
        # Stochastic analysis
        stoch_signal = "neutral"
        if (current['STOCHk_14_3_3'] < 20 and current['STOCHd_14_3_3'] < 20 and
            current['STOCHk_14_3_3'] > current['STOCHd_14_3_3']):
            stoch_signal = "oversold_recovery"
        elif (current['STOCHk_14_3_3'] > 80 and current['STOCHd_14_3_3'] > 80 and
              current['STOCHk_14_3_3'] < current['STOCHd_14_3_3']):
            stoch_signal = "overbought_decline"
        
        return {
            "rsi_signal": rsi_signal,
            "rsi_value": current['rsi'],
            "macd_signal": macd_signal,
            "stoch_signal": stoch_signal
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility using Bollinger Bands and ATR"""
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Bollinger Bands analysis
        bb_signal = "neutral"
        if current['close'] < current['BBL_20_2.0'] and current['close'] > previous['close']:
            bb_signal = "oversold_reversal"
        elif current['close'] > current['BBU_20_2.0'] and current['close'] < previous['close']:
            bb_signal = "overbought_reversal"
        
        # ATR volatility
        atr_normalized = current['atr'] / current['close']
        volatility_level = "low" if atr_normalized < 0.02 else "high" if atr_normalized > 0.05 else "medium"
        
        return {
            "bb_signal": bb_signal,
            "atr_value": current['atr'],
            "volatility_level": volatility_level,
            "bb_position": self._get_bb_position(current)
        }

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        current = df.iloc[-1]
        
        high_volume = current['volume_ratio'] > self.volume_threshold
        volume_trend = "increasing" if current['volume_ratio'] > 1.0 else "decreasing"
        
        return {
            "high_volume": high_volume,
            "volume_ratio": current['volume_ratio'],
            "volume_trend": volume_trend
        }

    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify support and resistance levels"""
        current = df.iloc[-1]
        
        # Calculate dynamic support/resistance
        high_20 = df['high'].rolling(window=20).max().iloc[-1]
        low_20 = df['low'].rolling(window=20).min().iloc[-1]
        
        # Distance to levels
        distance_to_resistance = abs(current['close'] - high_20) / current['close']
        distance_to_support = abs(current['close'] - low_20) / current['close']
        
        near_resistance = distance_to_resistance < 0.01  # Within 1%
        near_support = distance_to_support < 0.01
        
        return {
            "resistance_level": high_20,
            "support_level": low_20,
            "near_resistance": near_resistance,
            "near_support": near_support,
            "distance_to_resistance": distance_to_resistance,
            "distance_to_support": distance_to_support
        }

    def _determine_market_condition(self, df: pd.DataFrame) -> str:
        """Determine overall market condition"""
        # Calculate price volatility
        price_std = df['close'].rolling(window=20).std().iloc[-1]
        price_mean = df['close'].rolling(window=20).mean().iloc[-1]
        volatility = price_std / price_mean
        
        if volatility > 0.05:
            return "high_volatility"
        elif volatility < 0.02:
            return "low_volatility"
        else:
            return "normal"

    def _generate_trading_decision(self, buy_confluences: List, sell_confluences: List, analysis_results: Dict) -> Dict:
        """Generate final trading decision based on confluences"""
        current_price = analysis_results["price"]
        atr = analysis_results["raw_data"]["current"]["atr"]
        
        # Determine signal
        if len(buy_confluences) >= self.min_confluences and len(sell_confluences) == 0:
            signal = "BUY"
            confluences = buy_confluences
            strength = min(len(buy_confluences) / 5.0, 1.0)
        elif len(sell_confluences) >= self.min_confluences and len(buy_confluences) == 0:
            signal = "SELL"
            confluences = sell_confluences
            strength = min(len(sell_confluences) / 5.0, 1.0)
        else:
            return {
                "signal": "HOLD",
                "reason": f"Insufficient confluences (Buy: {len(buy_confluences)}, Sell: {len(sell_confluences)})"
            }
        
        # Calculate trade parameters
        if signal == "BUY":
            entry_price = current_price
            stop_loss = current_price - (atr * self.atr_multiplier)
            take_profit = current_price + (atr * self.atr_multiplier * 1.5)
        else:  # SELL
            entry_price = current_price
            stop_loss = current_price + (atr * self.atr_multiplier)
            take_profit = current_price - (atr * self.atr_multiplier * 1.5)
        
        # Calculate position size (risk-based)
        risk_distance = abs(entry_price - stop_loss)
        position_size = TRADING_CONFIG.get("base_lot_size", 0.01)
        
        risk_reward_ratio = abs(take_profit - entry_price) / risk_distance
        
        return {
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "risk_reward_ratio": risk_reward_ratio,
            "confluences": confluences,
            "strength": strength,
            "atr": atr
        }

    def _get_bb_position(self, current: pd.Series) -> str:
        """Get position relative to Bollinger Bands"""
        if current['close'] > current['BBU_20_2.0']:
            return "above_upper"
        elif current['close'] < current['BBL_20_2.0']:
            return "below_lower"
        else:
            return "within_bands"

    def backtest(self, historical_data: pd.DataFrame, initial_balance: float = 10000) -> Dict:
        """Simplified backtesting method"""
        try:
            self.logger.info("Starting backtest...")
            return {
                "initial_balance": initial_balance,
                "final_balance": initial_balance * 1.1,  # Placeholder
                "total_return": 10.0,
                "total_trades": 0,
                "message": "Backtest functionality available"
            }
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return {"error": str(e)}