import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import time

class Orders:
    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        
        # Order type mapping
        self.ORDER_TYPES = {
            'BUY': mt5.ORDER_TYPE_BUY,
            'SELL': mt5.ORDER_TYPE_SELL,
            'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
            'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
            'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
            'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
        }
        
        # Filling mode
        self.FILLING_MODE = mt5.ORDER_FILLING_IOC

    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None, comment: str = "AutoBot", 
                   magic: int = 234000) -> Dict:
        """
        Place a trading order with comprehensive error handling
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDm')
            order_type: Order type ('BUY', 'SELL', 'BUY_LIMIT', etc.)
            volume: Position size
            price: Entry price (for limit/stop orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number for order identification
            
        Returns:
            Dict with order result and details
        """
        try:
            # Validate connection
            if not self.connection.is_connected():
                return {"success": False, "error": "Not connected to MT5"}
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Symbol {symbol} not found"}
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                self.logger.warning(f"Symbol {symbol} not visible, attempting to select")
                if not mt5.symbol_select(symbol, True):
                    return {"success": False, "error": f"Failed to select symbol {symbol}"}
            
            # Get current prices
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": f"Failed to get tick data for {symbol}"}
            
            # Determine order price
            if order_type in ['BUY', 'SELL']:
                # Market order
                order_price = tick.ask if order_type == 'BUY' else tick.bid
            else:
                # Pending order
                if price is None:
                    return {"success": False, "error": "Price required for pending orders"}
                order_price = price
            
            # Normalize volume
            volume = self._normalize_volume(symbol, volume)
            if volume <= 0:
                return {"success": False, "error": "Invalid volume"}
            
            # Normalize prices
            order_price = self._normalize_price(symbol, order_price)
            if sl is not None:
                sl = self._normalize_price(symbol, sl)
            if tp is not None:
                tp = self._normalize_price(symbol, tp)
            
            # Validate stop loss and take profit
            validation_result = self._validate_sl_tp(symbol, order_type, order_price, sl, tp)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL if order_type in ['BUY', 'SELL'] else mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": self.ORDER_TYPES[order_type],
                "price": order_price,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.FILLING_MODE,
            }
            
            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                return {"success": False, "error": "Order send failed - no result"}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {self._get_error_description(result.retcode)}"
                self.logger.error(error_msg)
                return {
                    "success": False, 
                    "error": error_msg,
                    "retcode": result.retcode,
                    "result": result._asdict()
                }
            
            # Order successful
            order_info = {
                "success": True,
                "order_id": result.order,
                "deal_id": result.deal,
                "position_id": result.position_id if hasattr(result, 'position_id') else None,
                "volume": result.volume,
                "price": result.price,
                "comment": comment,
                "magic": magic,
                "timestamp": datetime.now(),
                "symbol": symbol,
                "type": order_type,
                "sl": sl,
                "tp": tp,
                "result": result._asdict()
            }
            
            self.logger.info(f"Order placed successfully: {order_type} {volume} {symbol} at {result.price}")
            return order_info
            
        except Exception as e:
            error_msg = f"Error placing order: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def modify_order(self, order_id: int, new_volume: Optional[float] = None, 
                    new_price: Optional[float] = None, new_sl: Optional[float] = None, 
                    new_tp: Optional[float] = None) -> Dict:
        """
        Modify an existing pending order or position
        """
        try:
            # Check if it's a pending order
            orders = mt5.orders_get(ticket=order_id)
            if orders:
                # It's a pending order
                order = orders[0]
                return self._modify_pending_order(order, new_volume, new_price, new_sl, new_tp)
            
            # Check if it's an open position
            positions = mt5.positions_get(ticket=order_id)
            if positions:
                # It's a position
                position = positions[0]
                return self._modify_position(position, new_sl, new_tp)
            
            return {"success": False, "error": f"Order/Position {order_id} not found"}
            
        except Exception as e:
            error_msg = f"Error modifying order: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def close_order(self, order_id: int, volume: Optional[float] = None) -> Dict:
        """
        Close an order (cancel pending order or close position)
        """
        try:
            # Check if it's a pending order
            orders = mt5.orders_get(ticket=order_id)
            if orders:
                return self._cancel_pending_order(order_id)
            
            # Check if it's an open position
            positions = mt5.positions_get(ticket=order_id)
            if positions:
                position = positions[0]
                return self._close_position(position, volume)
            
            return {"success": False, "error": f"Order/Position {order_id} not found"}
            
        except Exception as e:
            error_msg = f"Error closing order: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def get_order_status(self, order_id: int) -> Dict:
        """
        Get comprehensive status of an order or position
        """
        try:
            status_info = {"order_id": order_id, "found": False}
            
            # Check pending orders
            orders = mt5.orders_get(ticket=order_id)
            if orders:
                order = orders[0]
                status_info.update({
                    "found": True,
                    "type": "pending_order",
                    "symbol": order.symbol,
                    "volume": order.volume_initial,
                    "price_open": order.price_open,
                    "sl": order.sl,
                    "tp": order.tp,
                    "time_setup": datetime.fromtimestamp(order.time_setup),
                    "comment": order.comment,
                    "magic": order.magic,
                    "state": order.state,
                    "order_type": order.type
                })
                return status_info
            
            # Check open positions
            positions = mt5.positions_get(ticket=order_id)
            if positions:
                position = positions[0]
                status_info.update({
                    "found": True,
                    "type": "position",
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "price_open": position.price_open,
                    "price_current": position.price_current,
                    "sl": position.sl,
                    "tp": position.tp,
                    "profit": position.profit,
                    "swap": position.swap,
                    "time_open": datetime.fromtimestamp(position.time),
                    "comment": position.comment,
                    "magic": position.magic,
                    "position_type": position.type
                })
                return status_info
            
            # Check order history
            history_orders = mt5.history_orders_get(ticket=order_id)
            if history_orders:
                order = history_orders[0]
                status_info.update({
                    "found": True,
                    "type": "historical_order",
                    "symbol": order.symbol,
                    "volume": order.volume_initial,
                    "price": order.price_open,
                    "time_setup": datetime.fromtimestamp(order.time_setup),
                    "time_done": datetime.fromtimestamp(order.time_done),
                    "state": order.state,
                    "comment": order.comment
                })
                return status_info
            
            return status_info
            
        except Exception as e:
            error_msg = f"Error getting order status: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def list_open_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        List all open orders and positions
        """
        try:
            result = {
                "pending_orders": [],
                "open_positions": [],
                "total_pending": 0,
                "total_positions": 0
            }
            
            # Get pending orders
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            if orders:
                for order in orders:
                    order_info = {
                        "ticket": order.ticket,
                        "symbol": order.symbol,
                        "type": order.type,
                        "volume": order.volume_initial,
                        "price_open": order.price_open,
                        "sl": order.sl,
                        "tp": order.tp,
                        "time_setup": datetime.fromtimestamp(order.time_setup),
                        "comment": order.comment,
                        "magic": order.magic
                    }
                    result["pending_orders"].append(order_info)
                
                result["total_pending"] = len(orders)
            
            # Get open positions
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions:
                for position in positions:
                    position_info = {
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                        "type": position.type,
                        "volume": position.volume,
                        "price_open": position.price_open,
                        "price_current": position.price_current,
                        "sl": position.sl,
                        "tp": position.tp,
                        "profit": position.profit,
                        "swap": position.swap,
                        "time_open": datetime.fromtimestamp(position.time),
                        "comment": position.comment,
                        "magic": position.magic
                    }
                    result["open_positions"].append(position_info)
                
                result["total_positions"] = len(positions)
            
            return result
            
        except Exception as e:
            error_msg = f"Error listing orders: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def close_all_positions(self, symbol: Optional[str] = None, magic: Optional[int] = None) -> Dict:
        """
        Close all open positions for a symbol or magic number
        """
        try:
            closed_positions = []
            errors = []
            
            # Get positions to close
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if not positions:
                return {"success": True, "message": "No positions to close", "closed": []}
            
            for position in positions:
                # Filter by magic number if specified
                if magic is not None and position.magic != magic:
                    continue
                
                close_result = self._close_position(position)
                if close_result["success"]:
                    closed_positions.append(close_result)
                else:
                    errors.append(f"Failed to close {position.ticket}: {close_result['error']}")
            
            return {
                "success": len(errors) == 0,
                "closed_positions": closed_positions,
                "total_closed": len(closed_positions),
                "errors": errors
            }
            
        except Exception as e:
            error_msg = f"Error closing all positions: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def get_account_info(self) -> Dict:
        """
        Get current account information
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {"error": "Failed to get account info"}
            
            return {
                "login": account_info.login,
                "server": account_info.server,
                "name": account_info.name,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "profit": account_info.profit,
                "currency": account_info.currency,
                "leverage": account_info.leverage,
                "company": account_info.company
            }
            
        except Exception as e:
            error_msg = f"Error getting account info: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    # Private helper methods
    def _normalize_volume(self, symbol: str, volume: float) -> float:
        """Normalize volume according to symbol specifications"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0
            
            min_volume = symbol_info.volume_min
            max_volume = symbol_info.volume_max
            volume_step = symbol_info.volume_step
            
            # Normalize to step
            volume = round(volume / volume_step) * volume_step
            
            # Ensure within bounds
            volume = max(min_volume, min(max_volume, volume))
            
            return volume
            
        except Exception:
            return 0

    def _normalize_price(self, symbol: str, price: float) -> float:
        """Normalize price according to symbol specifications"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return price
            
            digits = symbol_info.digits
            return round(price, digits)
            
        except Exception:
            return price

    def _validate_sl_tp(self, symbol: str, order_type: str, price: float, 
                       sl: Optional[float], tp: Optional[float]) -> Dict:
        """Validate stop loss and take profit levels"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"valid": False, "error": "Symbol info not available"}
            
            stops_level = symbol_info.trade_stops_level
            point = symbol_info.point
            min_distance = stops_level * point
            
            if sl is not None:
                if order_type == 'BUY':
                    if sl >= price:
                        return {"valid": False, "error": "Stop loss must be below entry price for BUY"}
                    if (price - sl) < min_distance:
                        return {"valid": False, "error": f"Stop loss too close to entry price (min: {min_distance})"}
                else:  # SELL
                    if sl <= price:
                        return {"valid": False, "error": "Stop loss must be above entry price for SELL"}
                    if (sl - price) < min_distance:
                        return {"valid": False, "error": f"Stop loss too close to entry price (min: {min_distance})"}
            
            if tp is not None:
                if order_type == 'BUY':
                    if tp <= price:
                        return {"valid": False, "error": "Take profit must be above entry price for BUY"}
                    if (tp - price) < min_distance:
                        return {"valid": False, "error": f"Take profit too close to entry price (min: {min_distance})"}
                else:  # SELL
                    if tp >= price:
                        return {"valid": False, "error": "Take profit must be below entry price for SELL"}
                    if (price - tp) < min_distance:
                        return {"valid": False, "error": f"Take profit too close to entry price (min: {min_distance})"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

    def _modify_pending_order(self, order, new_volume=None, new_price=None, new_sl=None, new_tp=None) -> Dict:
        """Modify a pending order"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": order.ticket,
                "volume": new_volume if new_volume is not None else order.volume_initial,
                "price": new_price if new_price is not None else order.price_open,
                "sl": new_sl if new_sl is not None else order.sl,
                "tp": new_tp if new_tp is not None else order.tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Modify failed: {result.retcode}"}
            
            return {
                "success": True,
                "message": f"Order {order.ticket} modified successfully",
                "result": result._asdict()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error modifying pending order: {str(e)}"}

    def _modify_position(self, position, new_sl=None, new_tp=None) -> Dict:
        """Modify position SL/TP"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl if new_sl is not None else position.sl,
                "tp": new_tp if new_tp is not None else position.tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Modify failed: {result.retcode}"}
            
            return {
                "success": True,
                "message": f"Position {position.ticket} modified successfully",
                "result": result._asdict()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error modifying position: {str(e)}"}

    def _cancel_pending_order(self, order_id: int) -> Dict:
        """Cancel a pending order"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order_id,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Cancel failed: {result.retcode}"}
            
            return {
                "success": True,
                "message": f"Order {order_id} cancelled successfully",
                "result": result._asdict()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error cancelling order: {str(e)}"}

    def _close_position(self, position, volume: Optional[float] = None) -> Dict:
        """Close a position"""
        try:
            # Determine close volume
            close_volume = volume if volume is not None else position.volume
            close_volume = self._normalize_volume(position.symbol, close_volume)
            
            # Get current prices
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return {"success": False, "error": "Failed to get current prices"}
            
            # Determine close price and order type
            if position.type == mt5.POSITION_TYPE_BUY:
                close_price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": position.magic,
                "comment": f"Close {position.ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.FILLING_MODE,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "error": f"Close failed: {result.retcode}"}
            
            return {
                "success": True,
                "message": f"Position {position.ticket} closed successfully",
                "close_price": result.price,
                "volume_closed": close_volume,
                "result": result._asdict()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error closing position: {str(e)}"}

    def _get_error_description(self, retcode: int) -> str:
        """Get human-readable error description"""
        error_codes = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request cancelled",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Request partially filled",
            mt5.TRADE_RETCODE_ERROR: "Common error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "No money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "Off quotes",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid expiration",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
            mt5.TRADE_RETCODE_NO_CHANGES: "No changes",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "Autotrading disabled by server",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "Autotrading disabled by client",
            mt5.TRADE_RETCODE_LOCKED: "Request locked",
            mt5.TRADE_RETCODE_FROZEN: "Order or position frozen",
            mt5.TRADE_RETCODE_INVALID_FILL: "Invalid fill type",
            mt5.TRADE_RETCODE_CONNECTION: "No connection",
            mt5.TRADE_RETCODE_ONLY_REAL: "Only real accounts allowed",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Orders limit reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
            mt5.TRADE_RETCODE_INVALID_ORDER: "Invalid order",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "Position already closed"
        }
        
        return error_codes.get(retcode, f"Unknown error code: {retcode}")