import math

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
import warnings
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from ml_utils import determine_sentiment

BASE_URL = 'https://paper-api.alpaca.markets'
API_KEY = "PK9OUSKMDJF8OOGQ3Y97"
API_SECRET = "CCq25Y1fhFwi7BuWEhvKrlKM4Ja8bR8dEpLXxGmw"

ALPACA_CONFIG = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

prob_threshold = 0.9

class AlgorithmicTrader(Strategy):
    def initialize(self, sleeptime='24H', symbol="SPY", cash_at_risk=0.5):
        self.symbol = symbol
        self.sleeptime = sleeptime
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        """
        Determine the number of shares/contract to trade.
        :return:
        """
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = math.floor(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def get_date(self):
        today = self.get_datetime()
        start_date = today - timedelta(days=2)
        return today.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d')

    def get_sentiment(self):
        today, start_date = self.get_date()
        news = self.api.get_news(symbol=self.symbol, start=start_date, end=today)
        news = [event.__dict__["_raw"]["summary"] for event in news]
        sentiment = determine_sentiment(news)
        return sentiment

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()

        if cash > last_price * quantity:
            label, score = self.get_sentiment()
            if label.lower() == "positive" and score > prob_threshold:
                # If selling from before
                if self.last_trade == "sell":
                    self.sell_all()

                # Buy order
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price*1.2,
                    stop_loss_price=last_price*0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

            if label.lower() == "negative" and score > prob_threshold:
                # If selling from before
                if self.last_trade == "buy":
                    self.sell_all()

                # Buy order
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"


warnings.simplefilter(action='ignore', category=FutureWarning)
broker = Alpaca(ALPACA_CONFIG)
strategy = AlgorithmicTrader(name='algtrader', broker=broker)
backtesting_start = datetime(2023, 1, 1)
backtesting_end = datetime(2023, 12, 31)

strategy.backtest(
    YahooDataBacktesting,
    backtesting_start,
    backtesting_end,
    parameters={
        "symbol": "SPY",
        "cache_at_risk": 0.5
    },
)
