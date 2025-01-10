import yfinance as yf

def get_pinning_candidates(ticker):
    # Get options data
    options = yf.Ticker(ticker).options

    # Filter options expiring within the next 1-2 weeks
    near_expiry_options = [option for option in options if '2024-02' in option]

    pinning_candidates = []

    for option in near_expiry_options:
        option_chain = yf.Ticker(ticker).option_chain(option)
        calls = option_chain.calls

        # Calculate implied volatility and strike prices
        for index, call in calls.iterrows():
            implied_volatility = call['impliedVolatility']
            strike_price = call['strike']

            # Check if the stock price is close to the strike price
            if abs(call['lastPrice'] - strike_price) < 1 and implied_volatility > 0.3:
                pinning_candidates.append((option, strike_price, implied_volatility))

    return pinning_candidates

# Example usage
ticker_symbol = 'AAPL'
pinning_candidates = get_pinning_candidates(ticker_symbol)
print("Pinning Candidates for", ticker_symbol)
for candidate in pinning_candidates:
    print(candidate)
