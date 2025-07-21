import unittest
from src.bot.trading_bot import TradingBot

class TestTradingBot(unittest.TestCase):

    def setUp(self):
        self.bot = TradingBot(login=00000000, password="passs", server="Exness-MT5Trial7")

    def test_initialization(self):
        self.assertIsNotNone(self.bot)
        self.assertEqual(self.bot.login, 00000000)
        self.assertEqual(self.bot.password, "passs")
        self.assertEqual(self.bot.server, "Exness-MT5Trial7")

    def test_execute_trade(self):
        # Assuming execute_trade method returns True on success
        result = self.bot.execute_trade("BTCUSD", 1.0, "buy")
        self.assertTrue(result)

    def test_monitor_market_conditions(self):
        # Assuming monitor_market_conditions returns a dictionary with market data
        market_data = self.bot.monitor_market_conditions("BTCUSD")
        self.assertIn("price", market_data)
        self.assertIn("volume", market_data)

    def tearDown(self):
        self.bot.disconnect()

if __name__ == '__main__':
    unittest.main()