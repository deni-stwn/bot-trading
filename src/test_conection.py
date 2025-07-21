import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from mt5_integration.connection import Connection
from config.settings import LOGIN, PASSWORD, SERVER
import logging

logging.basicConfig(level=logging.INFO)

def diagnose_mt5():
    """Diagnose MT5 installation and connection issues"""
    print("🔍 Diagnosing MT5 setup...")
    
    # Check if MT5 package is installed
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 package is installed")
        print(f"   Version: {mt5.__version__ if hasattr(mt5, '__version__') else '5.0.4993'}")
    except ImportError:
        print("❌ MetaTrader5 package not found!")
        print("   Install it with: pip install MetaTrader5")
        return False
    
    # Check MT5 terminal availability
    print("\n🔍 Checking MT5 terminal...")
    
    # Try basic initialization without login
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"❌ MT5 initialization failed: {error}")
        
        # Common solutions
        print("\n💡 Possible solutions:")
        print("1. Make sure MetaTrader 5 terminal is installed")
        print("2. Run MetaTrader 5 at least once manually")
        print("3. Enable 'Allow automated trading' in Tools -> Options -> Expert Advisors")
        print("4. Check if antivirus is blocking MT5")
        print("5. Try running this script as Administrator")
        
        return False
    else:
        print("✅ MT5 initialized successfully")
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"   Terminal: {getattr(terminal_info, 'name', 'MetaTrader 5')}")
            
            # Safe attribute access
            if hasattr(terminal_info, 'version'):
                print(f"   Version: {terminal_info.version}")
            
            if hasattr(terminal_info, 'build'):
                print(f"   Build: {terminal_info.build}")
            
            if hasattr(terminal_info, 'path'):
                print(f"   Path: {terminal_info.path}")
            
            if hasattr(terminal_info, 'connected'):
                print(f"   Connected: {terminal_info.connected}")
            
            if hasattr(terminal_info, 'trade_allowed'):
                print(f"   Trade allowed: {terminal_info.trade_allowed}")
                
                if not terminal_info.trade_allowed:
                    print("   ⚠️ WARNING: Trading is not allowed!")
                    print("   Enable 'Allow automated trading' in Expert Advisors settings")
            
            # Show all available attributes for debugging
            print(f"   Available attributes: {[attr for attr in dir(terminal_info) if not attr.startswith('_')]}")
        
        mt5.shutdown()
        return True

def test_login():
    """Test login with credentials"""
    print("\n🔍 Testing login credentials...")
    
    if not mt5.initialize():
        print("❌ Cannot initialize MT5")
        return False
    
    # Try to login
    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        error = mt5.last_error()
        print(f"❌ Login failed: {error}")
        
        print(f"\n📋 Login details:")
        print(f"   Login: {LOGIN}")
        print(f"   Server: {SERVER}")
        print(f"   Password: {'*' * len(PASSWORD)}")
        
        # More specific error handling
        if error[0] == 10004:  # Invalid account
            print("\n💡 Invalid Account Solutions:")
            print("1. Check login number is correct")
            print("2. Make sure this is a real/demo account number")
            print("3. Verify with your broker")
        elif error[0] == 10015:  # Invalid server
            print("\n💡 Invalid Server Solutions:")
            print("1. Check server name spelling (case-sensitive)")
            print("2. Try server variations like:")
            print("   - Exness-MT5Trial7")
            print("   - Exness-MT5Trial")
            print("   - ExnessTrial-MT5")
        else:
            print("\n💡 General Solutions:")
            print("1. Check if credentials are correct")
            print("2. Check if server name is exact (case-sensitive)")
            print("3. Make sure account is not locked/suspended")
            print("4. Try logging in manually in MT5 terminal first")
            print("5. Check internet connection")
        
        mt5.shutdown()
        return False
    else:
        print("✅ Login successful!")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"   Account: {account_info.login}")
            print(f"   Name: {getattr(account_info, 'name', 'N/A')}")
            print(f"   Server: {account_info.server}")
            print(f"   Balance: {account_info.balance} {account_info.currency}")
            print(f"   Leverage: 1:{account_info.leverage}")
            print(f"   Company: {getattr(account_info, 'company', 'N/A')}")
        
        mt5.shutdown()
        return True

def test_symbol():
    """Test BTC symbol availability"""
    print("\n🔍 Testing BTC symbol...")
    
    if not mt5.initialize():
        print("❌ Cannot initialize MT5")
        return False
    
    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print("❌ Cannot login")
        mt5.shutdown()
        return False
    
    # Test different BTC symbol variations for Exness
    btc_symbols = ["BTCUSDm", "BTCUSD", "Bitcoin", "XBTUSD", "BTC/USD", "BTCUSDT"]
    found_symbol = None
    
    print("   Searching for BTC symbols...")
    for symbol in btc_symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"✅ Found symbol: {symbol}")
            print(f"   Description: {getattr(symbol_info, 'description', 'N/A')}")
            print(f"   Digits: {symbol_info.digits}")
            print(f"   Point: {symbol_info.point}")
            print(f"   Visible: {symbol_info.visible}")
            
            # Make symbol visible if it's not
            if not symbol_info.visible:
                if mt5.symbol_select(symbol, True):
                    print(f"   ✅ Symbol {symbol} made visible")
                else:
                    print(f"   ❌ Failed to make {symbol} visible")
            
            # Try to get tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"   Current Ask: {tick.ask}")
                print(f"   Current Bid: {tick.bid}")
                print(f"   Spread: {tick.ask - tick.bid}")
                found_symbol = symbol
            break
        else:
            print(f"   ❌ Symbol not found: {symbol}")
    
    if not found_symbol:
        print("\n🔍 Looking for all available symbols...")
        symbols = mt5.symbols_get()
        if symbols:
            btc_like = [s.name for s in symbols if 'btc' in s.name.lower() or 'bitcoin' in s.name.lower()]
            if btc_like:
                print(f"   Found BTC-related symbols: {btc_like}")
            else:
                print("   No BTC symbols found in available symbols")
                # Show first 10 available symbols as example
                print(f"   Available symbols (first 10): {[s.name for s in symbols[:10]]}")
        
        print("\n💡 No BTC symbols found. Possible solutions:")
        print("1. Check with your broker what BTC symbols are available")
        print("2. Look in Market Watch in MT5 terminal")
        print("3. Try enabling more symbols in Market Watch")
        print("4. Contact Exness support for available crypto symbols")
    
    mt5.shutdown()
    return found_symbol is not None

def test_connection_with_class():
    """Test using our Connection class"""
    print("\n🔍 Testing with Connection class...")
    
    try:
        connection = Connection(LOGIN, PASSWORD, SERVER)
        if connection.connect():
            print("✅ Connection class works!")
            
            info = connection.get_connection_info()
            if "error" not in info:
                print(f"   Account: {info['account']['login']}")
                print(f"   Balance: {info['account']['balance']} {info['account']['currency']}")
                print(f"   Server: {info['account']['server']}")
            
            connection.disconnect()
            return True
        else:
            print("❌ Connection class failed")
            return False
    except Exception as e:
        print(f"❌ Connection class error: {e}")
        return False

def main():
    print("🚀 MT5 Connection Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: Diagnose MT5
    if not diagnose_mt5():
        return
    
    # Step 2: Test login
    if not test_login():
        return
    
    # Step 3: Test symbol
    symbol_found = test_symbol()
    
    # Step 4: Test our connection class
    test_connection_with_class()
    
    print("\n" + "=" * 50)
    if symbol_found:
        print("✅ All checks completed successfully!")
        print("The bot should now work properly.")
        print("\nNext steps:")
        print("1. Run: python src/main.py")
        print("2. The bot will start trading automatically")
    else:
        print("⚠️ Checks completed with warnings!")
        print("BTC symbol not found - you may need to:")
        print("1. Contact Exness to enable crypto trading")
        print("2. Check available symbols in MT5 Market Watch")
        print("3. Update the symbol name in config/settings.py")

if __name__ == "__main__":
    main()