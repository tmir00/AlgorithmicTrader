import math

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
import warnings
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from ml_utils import determine_sentiment
from typing import Tuple

# Set up configuration to connect to Alpaca's trading API.
BASE_URL = 'https://paper-api.alpaca.markets'
API_KEY = "PK9OUSKMDJF8OOGQ3Y97"
API_SECRET = "CCq25Y1fhFwi7BuWEhvKrlKM4Ja8bR8dEpLXxGmw"

ALPACA_CONFIG = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

prob_threshold = 0.9


# Here we define a trading strategy.
class AlgorithmicTrader(Strategy):
    def initialize(self, sleeptime='2D', symbol="SPY", cash_at_risk=0.5) -> None:
        """
        Initializes the trading strategy.

        :param sleeptime: The interval to wait between trade evaluations ('12H' for 12 hours).
        :param symbol: The ticker symbol of the asset to be traded ('SPY' for S&P 500).
        :param cash_at_risk: The fraction of the cash that is at risk for each trade (0.5 for 50%).
        """
        self.symbol = symbol
        self.sleeptime = sleeptime
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        # This is for retrieving relevant news information.
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self) -> Tuple[float, float, int]:
        """
        Calculates the number of shares to trade based on available cash and the amount at risk.

        :return: Tuple containing the cash available, the last price of the asset, and the calculated quantity of shares to trade.
        """
        # Get the current cash available in the account.
        cash = self.get_cash()
        # Get the last price of the asset.
        last_price = self.get_last_price(self.symbol)

        # Calculate how many shares to trade
        quantity = math.floor(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def get_date(self):
        """
        Calculates the date range for fetching news.

        :return: Tuple containing today's date and the start date as strings formatted as 'YYYY-MM-DD'.
        """
        today = self.get_datetime()
        start_date = today - timedelta(days=2)

        # Return datetime objects as formatted strings.
        return today.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d')

    def get_sentiment(self):
        """
        Fetches news for the specified symbol and evaluates the sentiment of the news articles.

        :return: A tuple of the most common sentiment label and the average sentiment score.
        """
        # Get the current date and start date for fetching news articles
        today, start_date = self.get_date()
        # Fetch news articles for the specified symbol within the specified date range
        news = self.api.get_news(symbol=self.symbol, start=start_date, end=today)
        # Extract summary texts from the news articles
        news = [event.__dict__["_raw"]["summary"] for event in news]

        # Determine the sentiment and it's probability then return
        sentiment = determine_sentiment(news)
        return sentiment

    def on_trading_iteration(self):
        """
        Runs a single iteration of the trading strategy, evaluating whether to buy or sell
        based on the sentiment of news articles.

        Makes decisions to place buy or sell orders based on positive or negative sentiment.
        """
        cash, last_price, quantity = self.position_sizing()

        # Check if there's enough cash to make a trade.
        if cash > last_price * quantity:
            label, score = self.get_sentiment()

            # Here the sentiment is positive, we expect a rise in price so buy.
            if label.lower() == "positive" and score > prob_threshold:
                # Model predicts prices to rise, if you were in a short position before (sold shares expecting prices
                # to drop to buy them back) buy these shares back now because prices will rise next.
                if self.last_trade == "sell":
                    self.sell_all()

                # Initiating a buy order based on positive market sentiment. The `take_profit_price` is set to 20%
                # above the last price to automatically sell and secure profits if the price reaches this level.
                # The `stop_loss_price` is set to 5% below the last price to minimize losses if the market moves
                # against the expected direction.
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

            # Sell signal because model expects a drop in price.
            if label.lower() == "negative" and score > prob_threshold:
                # If buying from before, sell your shares because model expects a drop in price.
                if self.last_trade == "buy":
                    self.sell_all()

                # Placing a sell order due to negative sentiment, indicating expected price decline. The
                # `take_profit_price` is set at 20% below the last price, aiming to buy back at this lower
                # price for profit in a short position. The `stop_loss_price` is 5% above the last price,
                # limiting potential losses if the price unexpectedly rises.
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
# Initialize broker to facilitate buying and selling.
broker = Alpaca(ALPACA_CONFIG)
# Initialize our Algorithmic Trader.
strategy = AlgorithmicTrader(name='algtrader', broker=broker)
backtesting_start = datetime(2023, 1, 1)
backtesting_end = datetime(2023, 12, 31)

# Perform Backtesting (testing with historical data)
strategy.backtest(
    YahooDataBacktesting,
    backtesting_start,
    backtesting_end,
    parameters={
        "symbol": "SPY",
        "cache_at_risk": 0.5
    },
)
