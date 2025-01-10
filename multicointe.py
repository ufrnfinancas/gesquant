import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt

# Step 1: Generate example data (replace this with your actual stock data)
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=100)
stock1 = np.cumsum(np.random.randn(100)) + 100  # Example data for stock 1
stock2 = np.cumsum(np.random.randn(100)) + 100  # Example data for stock 2
stock3 = np.cumsum(np.random.randn(100)) + 100  # Example data for stock 3
df = pd.DataFrame({'Stock1': stock1, 'Stock2': stock2, 'Stock3': stock3}, index=dates)

# Step 2: Plot the stock prices
df.plot(title='Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Step 3: Perform Johansen test for cointegration
def test_cointegration(df):
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1                       # Trace statistic
    crit_vals = result.cvt                        # Critical values (95% confidence)
    eig_stat = result.lr2                         # Eigen statistic
    eig_crit_vals = result.cvm                    # Critical values for eigen statistic (95% confidence)

    print(f"Trace statistic: {trace_stat}")
    print(f"Critical values (95% confidence):\n{crit_vals}")
    print(f"Eigen statistic: {eig_stat}")
    print(f"Critical values for eigen statistic (95% confidence):\n{eig_crit_vals}")

test_cointegration(df)
