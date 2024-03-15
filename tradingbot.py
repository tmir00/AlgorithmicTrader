from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime

ALPACA_CONFIG = {
    "API_KEY": "PK4DPLSH5NPINXTG81E9",
    "API_SECRET": "BD29IYArElQjybWzbD0atZEw43junmHhlsncNZzu",
    "ENDPOINT": "https://paper-api.alpaca.markets/v2"
}


class AlgorithmicTrader(Strategy):
    def initialize(self, sleeptime='24h', symbol="", last_trade=None):
        pass
    def on_trading_iteration(self):
        pass

