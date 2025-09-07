def Placing_Market_Order(session, category, symbol, side, qty, marketUnit):

# Function to place a Limit order with stoploss

    response = session.place_order(
        category=category,
        symbol=symbol,
        side=side,
        orderType="Market",         # "Limit" for market orders
        qty=qty,                    # For Limit orders, the quantity is always in ETH/BTC etc. and not in USDT!
        marketUnit= marketUnit,
    )

    return response  # <-- Add this line to return the API response