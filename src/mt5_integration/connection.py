import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, Optional
import time

class Connection:
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self, retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Connect to MetaTrader 5 with retry mechanism
        """
        for attempt in range(retries):
            try:
                self.logger.info(f"Attempting to connect to MT5 (Attempt {attempt + 1}/{retries})")
                
                # Initialize MT5
                if not mt5.initialize():
                    self.logger.error("Failed to initialize MT5")
                    if attempt < retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return False
                
                # Login to account
                if not mt5.login(self.login, password=self.password, server=self.server):
                    error = mt5.last_error()
                    self.logger.error(f"Failed to login: {error}")
                    mt5.shutdown()
                    if attempt < retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return False
                
                # Verify connection
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Failed to get account info after login")
                    mt5.shutdown()
                    if attempt < retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return False
                
                self.connected = True
                self.logger.info(f"Successfully connected to MT5")
                self.logger.info(f"Account: {account_info.login}")
                self.logger.info(f"Server: {account_info.server}")
                self.logger.info(f"Balance: {account_info.balance} {account_info.currency}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False
        
        return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from MetaTrader 5
        """
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self.logger.info("Disconnected from MT5")
                return True
            return True
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to MT5
        """
        try:
            if not self.connected:
                return False
                
            # Test connection by getting account info
            account_info = mt5.account_info()
            return account_info is not None
            
        except Exception:
            self.connected = False
            return False
    
    def reconnect(self) -> bool:
        """
        Reconnect to MetaTrader 5
        """
        self.logger.info("Attempting to reconnect...")
        self.disconnect()
        return self.connect()
    
    def get_connection_info(self) -> Dict:
        """
        Get connection information
        """
        try:
            if not self.is_connected():
                return {"connected": False, "error": "Not connected"}
            
            account_info = mt5.account_info()
            terminal_info = mt5.terminal_info()
            
            return {
                "connected": True,
                "account": {
                    "login": account_info.login,
                    "server": account_info.server,
                    "name": account_info.name,
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "currency": account_info.currency,
                    "leverage": account_info.leverage,
                    "company": account_info.company
                },
                "terminal": {
                    "version": terminal_info.version,
                    "build": terminal_info.build,
                    "name": terminal_info.name,
                    "path": terminal_info.path,
                    "connected": terminal_info.connected,
                    "trade_allowed": terminal_info.trade_allowed
                }
            }
            
        except Exception as e:
            return {"connected": False, "error": str(e)}