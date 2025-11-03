# Nathan Johnson - HW4 Machine Learning

The dataset, or I should say library that I will be using is the yfinance library for python. https://pypi.org/project/yfinance/ This is essentially a python library that wraps the Yahoo finance API allowing users to easily pull stock market data into there code. Some sample code for loading the data is below.

---

``` python
import yfinance as yf

# Download daily prices for Apple and Microsoft
data = yf.download(["AAPL", "MSFT"], start="2020-01-01", end="2025-01-01")
data.head()
```

---

The dataset contains historical stock market data collected from Yahoo Finance. It covers daily trading information for publicly traded companies such as Apple.

**Data Type:**

- Primarily numerical data (prices, volumes, returns).


**Typical Features (Columns):**
- Date (Trading date index)

- Open (Price at market open)

- High (Highest price of the day)

- Low  (Lowest price of the day)

- Close (Price at market close)

- Volume (Number of shares traded)

General size is around 250 trading days per year per company so 5 years would be around 1,250 rows per stock (5 * 250)

An example use case would be predicting stock price changes.