# Algorithmic Trading Bot: Leveraging News Sentiment

## Project Overview

This project implements an algorithmic trading bot designed to navigate the stock market with a strategy rooted in news sentiment analysis. By analyzing the mood of financial news using machine learning, the bot aims to anticipate market movements and execute trades that capitalize on these predictions.

### Core Components

- **Sentiment Analysis with BERT**: At the forefront of this approach is a robust sentiment analysis powered by a BERT model specialized in financial news, [FinancialBERT-Sentiment-Analysis](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis). This model, leveraging the `transformers.pipeline`, determines the sentiments embedded in news articles, guiding our trading decisions.

- **Lumibot Framework**: Built upon the [Lumibot framework](https://lumibot.lumiwealth.com/), this bot benefits from a robust structure for algorithmic strategy implementation. Lumibot not only helps with the development process but also offers tools for backtesting.

- **Alpaca for Trading and Backtesting**: Integrating with [Alpaca](https://alpaca.markets/), this bot gains access to trading services and backtesting capabilities. Alpaca's platform supports both paper trading for strategy testing and live trading for real-market execution (we do not use live trading here).

### Strategy Insight

The essence of this trading bot's strategy lies in its response to the sentiment derived from financial news. Positive sentiment signals for long positions, anticipating price rises, while negative sentiment signals the potential for short positions, expecting declines. We also implement risk management practices, including strategic take-profit and stop-loss orders, ensuring a balanced approach to trading.

### Why This Project?

This project is more than a trading bot for me; it's an exploration into the field of algorithmic trading, bridging my interest in Machine Learning, Finance and Computer Science. It showcases the potential of machine learning in interpreting human language and sentiments and its application in making calculated financial decisions. This bot is a testament to the synergy between technology and finance.

<br>

## Getting Started

Follow these steps to get the trading bot up and running on your local machine:

1. Clone the repository:
```bash
git clone https://github.com/tmir00/AlgorithmicTrader.git
```

2. Navigate to the project directory:
```bash
cd AlgorithmicTrader
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Change to the source code directory:
```bash
pip install -r requirements.txt
```

5. Run the trading bot:
```bash
python tradingbot.py
```

<br>

## Results
If you have run the previous steps correctly, and the bot runs, you should see results such as this:

![TradingResults](https://github.com/tmir00/AlgorithmicTrader/blob/main/images/tradingresult2.png)

![TradingResultsDetailed](https://github.com/tmir00/AlgorithmicTrader/blob/main/images/tradingresult1.png)

<br>

## Project Components

### `tradingbot.py`

Thhis is the core of the trading system, orchestrating the trading strategy based on news sentiment analysis.

- **`initialize`**: Sets up the trading strategy, including the symbol to trade, the interval between trade evaluations, and the fraction of cash at risk.
- **`position_sizing`**: Calculates the size of a position based on the available cash and predefined risk parameters.
- **`get_date`**: Determines the date range for fetching relevant news articles for sentiment analysis.
- **`get_sentiment`**: Fetches and analyzes the sentiment of news articles to guide trading decisions.
- **`on_trading_iteration`**: Executes the trading logic in each iteration, making buy or sell decisions based on sentiment analysis results.

### `model_eval.py`

Evaluates the sentiment analysis model's performance on a predefined dataset, this was used to determine the best BERT model to use.

- **`evaluate_model`**: Runs the sentiment analysis on text data and compares the predictions with actual sentiment labels to calculate accuracy.

The models were testing against this dataset from Kaggle: [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

### `ml_utils.py`

Provides utility functions for sentiment analysis within the trading strategy.

- **`determine_sentiment`**: Analyzes the sentiment of a given list of texts using a pre-trained BERT model. It filters out neutral sentiments, identifies the most common sentiment, and calculates the average score of that sentiment.

These scripts combine the latest advancements in NLP and financial trading strategies, aiming to create a sophisticated algorithmic trading bot that makes informed decisions based on the sentiment of financial news.

<br>

## Process Flow Diagram
The diagram below summarizes and illustrates the operational processes within the project, from initial model evaluation to the final trades made by the trading bot.

![Process Flow](https://github.com/tmir00/AlgorithmicTrader/blob/main/images/ProjectFlow.png)

<br>

## References
**Dataset:** [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

**Financial BERT Model:** [ahmedrachid/FinancialBERT-Sentiment-Analysis](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis)

**Youtube Guide: [Nicholas Renotte](https://www.youtube.com/watch?v=c9OjEThuJjY&ab_channel=NicholasRenotte)**
