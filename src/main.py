from bot.trading_bot import TradingBot
from bot.strategy import Strategy
from config.settings import LOGIN, PASSWORD, SERVER
import logging
import json

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*50)
    logger.info("STARTING ADVANCED BTC/USD TRADING BOT")
    logger.info("="*50)
    
    try:
        # Initialize strategy
        strategy = Strategy()
        logger.info("Strategy initialized successfully")
        
        # Initialize trading bot
        bot = TradingBot(login=LOGIN, password=PASSWORD, server=SERVER)
        bot.strategy = strategy  # Inject strategy
        
        # Connect to MT5
        if bot.connect():
            logger.info("Connected to MetaTrader5 successfully")
            
            # Start trading
            logger.info("Starting automated trading...")
            bot.start_trading()
        else:
            logger.error("Failed to connect to MetaTrader5")
            
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    main()