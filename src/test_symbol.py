import MetaTrader5 as mt5
from config.settings import LOGIN, PASSWORD, SERVER

def test_btc_symbols():
    print("🔍 Testing BTC Symbol Data")
    print("=" * 30)
    
    if not mt5.initialize():
        print("❌ MT5 init failed")
        return
    
    if not mt5.login(LOGIN, password=PASSWORD, server=SERVER):
        print("❌ Login failed")
        return
    
    # Test different BTC symbols
    symbols_to_test = [
        "BTCUSDm", "BTCUSD", "Bitcoin", "XBTUSD", 
        "BTC", "BTCUSDT", "XBTUSD", "XAUUSDm"
    ]
    
    for symbol in symbols_to_test:
        print(f"\n🔍 Testing symbol: {symbol}")
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"   ❌ Symbol {symbol} not found")
            continue
        
        print(f"   ✅ Symbol found: {symbol}")
        print(f"   Description: {getattr(symbol_info, 'description', 'N/A')}")
        
        # Make symbol visible
        if not symbol_info.visible:
            if mt5.symbol_select(symbol, True):
                print(f"   ✅ Symbol made visible")
            else:
                print(f"   ❌ Failed to make visible")
        
        # Test getting data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None and len(rates) > 0:
            print(f"   ✅ Data available: {len(rates)} bars")
            
            # Check what columns are available
            import pandas as pd
            df = pd.DataFrame(rates)
            print(f"   Columns: {list(df.columns)}")
            
            # Show sample data
            print(f"   Latest price: {rates[-1]['close']}")
            if 'tick_volume' in df.columns:
                print(f"   Latest volume: {rates[-1]['tick_volume']}")
            elif 'real_volume' in df.columns:
                print(f"   Latest real volume: {rates[-1]['real_volume']}")
            else:
                print(f"   ⚠️ No volume data available")
        else:
            print(f"   ❌ No data available")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_btc_symbols()