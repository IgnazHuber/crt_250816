def Checking_Balance(session):
    response = session.get_wallet_balance(accountType="UNIFIED")
    # response = session.get_wallet_balance(accountType="SPOT")

    orders = response["result"]["list"]

    # Initialize variables to store the values
    btc_balance = 0
    eth_balance = 0
    ada_balance = 0
    sol_balance = 0
    usdt_balance = 0

    # Iterate through the coins and assign values to variables
    for coin_info in orders[0]['coin']:
        coin_name = coin_info['coin']
        walletBalance = coin_info['walletBalance']

        if coin_name == 'BTC':
            btc_balance = float(walletBalance)
        if coin_name == 'ETH':
            eth_balance = float(walletBalance)
        if coin_name == 'ADA':
            eth_balance = float(walletBalance)
        if coin_name == 'SOL':
            eth_balance = float(walletBalance)
        if coin_name == 'USDT':
            usdt_balance = float(walletBalance)

    return btc_balance, eth_balance, ada_balance, sol_balance, usdt_balance