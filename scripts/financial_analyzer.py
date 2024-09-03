import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import os
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

class FinancialAnalyzer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    # def retrieve_stock_data(self):
    #     return yf.download(self.ticker, start=self.start_date, end=self.end_date)

    def calculate_moving_average(self, data, window_size):
        return ta.SMA(data, timeperiod=window_size)

    def calculate_technical_indicators(self, data):
        # Calculate various technical indicators
        data['SMA'] = self.calculate_moving_average(data['Close'], 20)
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        data['EMA'] = ta.EMA(data['Close'], timeperiod=20)
        macd, macd_signal, _ = ta.MACD(data['Close'])
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        # Add more indicators as needed
        return data

    def plot_stock_data(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'SMA'], title='Stock Price with Moving Average')
        fig.show()

    def plot_rsi(self, data):
        fig = px.line(data, x=data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.show()

    def plot_ema(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'EMA'], title='Stock Price with Exponential Moving Average')
        fig.show()

    def plot_macd(self, data):
        fig = px.line(data, x=data.index, y=['MACD', 'MACD_Signal'], title='Moving Average Convergence Divergence (MACD)')
        fig.show()
    
    def calculate_portfolio_weights(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        weights = dict(zip(tickers, weights.values()))
        return weights

    def calculate_portfolio_performance(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        mu = expected_returns.mean_historical_return(data)
        cov = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_sharpe()
        portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
        return portfolio_return, portfolio_volatility, sharpe_ratio
    def load_stock_data(self, ticker):

        file_path = f'../yfinance_data/{ticker}_historical_data.csv'  # Adjust path if needed

        # Check if file exists before reading
        if os.path.isfile(file_path):
            # Load the data and set 'Date' as the index
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            return data
        else:
            print(f"Warning: File not found for ticker {ticker} at path {file_path}")
            return None

    def get_min_max_dates(self, tickers):

        stock_dates = {}

        for ticker in tickers:
            data = self.load_stock_data(ticker)
            if data is not None:
                # Ensure the 'Date' column is in datetime format
                data['Date'] = pd.to_datetime(data['Date'])

                # Get the minimum and maximum dates
                min_date = data['Date'].min()
                max_date = data['Date'].max()

                # Store the min and max dates in the dictionary
                stock_dates[ticker] = (min_date, max_date)

        return stock_dates



