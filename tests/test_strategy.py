import unittest
from src.bot.strategy import Strategy

class TestStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = Strategy()

    def test_analyze_market_data(self):
        # Example market data
        market_data = {
            'prices': [100, 102, 101, 105, 104],
            'volume': [10, 15, 10, 20, 25]
        }
        decision = self.strategy.analyze_market_data(market_data)
        self.assertIn(decision, ['buy', 'sell', 'hold'])

    def test_trading_decision(self):
        # Example conditions for trading decision
        conditions = {
            'signal': 'buy',
            'risk_level': 'low'
        }
        decision = self.strategy.trading_decision(conditions)
        self.assertEqual(decision, 'execute')

    def test_invalid_market_data(self):
        # Test with invalid market data
        market_data = None
        with self.assertRaises(ValueError):
            self.strategy.analyze_market_data(market_data)

if __name__ == '__main__':
    unittest.main()