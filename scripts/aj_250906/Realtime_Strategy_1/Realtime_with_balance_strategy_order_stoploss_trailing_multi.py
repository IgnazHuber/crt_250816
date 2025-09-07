# Realtime_with_balance_strategy_order_stoploss_trailing.py
import time
import pandas as pd
import warnings
from datetime import datetime, timezone, timedelta
from config import BYBIT_DEMO_API_KEY, BYBIT_DEMO_API_SECRET
from pybit.unified_trading import HTTP
import os
import json
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from Goldenratio_vectorized import calculate_golden_ratios
from Support_Resistance_vectorized import calculate_support_levels
from Trendline_Up_Support_vectorized import calc_TL_Up_Support
from Trendline_Up_Resistance_vectorized import calc_TL_Up_Resistance
from Trendline_Down_Resistance_vectorized import calc_TL_Down_Resistance
from Get_Account_Balance import Checking_Balance
from Placing_Market_Order import Placing_Market_Order
from Get_candlestick_data import get_candlestick_data
from Get_Trade_History import get_trade_history
from Strategy_1 import strategy_1
import multiprocessing
import math
from functools import partial
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for risk management
RISK_PERC = 0.01  # 1% risk per trade
BROKERAGE_BUY = 0.001  # 0.1% taker fee for spot market orders on Bybit
BROKERAGE_SELL = 0.001  # 0.1% taker fee for spot market orders on Bybit

# Initialize Bybit API session
session_demo = HTTP(api_key=BYBIT_DEMO_API_KEY, api_secret=BYBIT_DEMO_API_SECRET, demo=True)

# Define order-related columns
ORDER_COLUMNS = ['Order_ID', 'Order_Type', 'Order_Qty', 'Order_Price', 'Order_Status', 'Order_Fees',
                 'Order_Created_Time', 'Actual_Buy', 'Actual_Sell', 'Trade_Qty', 'Trade_Price', 'Trade_Fee',
                 'Trade_Time']

# Define precision rules for quantity (based on typical Bybit spot trading requirements)
SYMBOL_PRECISION = {
    'BTCUSDT': 6,  # e.g., 0.000001 BTC
    'ETHUSDT': 3,  # e.g., 0.001 ETH
    'ADAUSDT': 1,  # e.g., 0.1 ADA
}


def get_quantity_precision(symbol):
    """Return the number of decimal places allowed for the symbol's quantity."""
    return SYMBOL_PRECISION.get(symbol, 4)  # Default to 4 if symbol not found


def round_down(value, decimals):
    """Round down the value to the specified number of decimal places, preserving the exact precision."""
    if value <= 0:
        return 0
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def get_interval_delta(interval):
    """Get timedelta for the given interval."""
    if interval == "D":
        return timedelta(days=1)
    return timedelta(minutes=int(interval))


def format_date(date_value):
    """Format date to string in 'YYYY-MM-DD HH:MM:SS' format."""
    if isinstance(date_value, str):
        try:
            date_obj = pd.to_datetime(date_value)
        except ValueError:
            return date_value
    else:
        date_obj = date_value
    return date_obj.strftime('%Y-%m-%d %H:%M:%S')


def compute_indicators(df, symbol, interval):
    """Compute technical indicators on the DataFrame with interval-specific tolerances for CBullDivg_analysis."""
    try:
        df = Initialize_RSI_EMA_MACD(df)
        Candlestick_Type(df)
        Level_1_Max_Min(df)

        # Set tolerance for CBullDivg_analysis based on interval
        if interval == "D":
            CBullDivg_analysis(df, 0.1, 3.25)
            HBullDivg_analysis(df, 0.1, 3.25)
        elif interval == "240":
            CBullDivg_analysis(df, 0.05, 3.25)
            HBullDivg_analysis(df, 0.05, 3.25)
        else:
            CBullDivg_analysis(df, 0.01, 3.25)
            HBullDivg_analysis(df, 0.01, 3.25)
        return df

    except Exception as e:
        print(f"Error computing indicators: {e}")
        return None


def save_parquet(df, folder):
    """Save DataFrame to parquet file in the specified folder, named with the last timestamp."""
    if df.empty or len(df) < 1:
        print("DataFrame is empty, skipping save.")
        return
    os.makedirs(folder, exist_ok=True)
    last_timestamp = df.iloc[-1]['timestamp']
    file_timestamp = last_timestamp.replace(':', '-').replace(' ', '_')
    parquet_path = os.path.join(folder, f"historical_data_{file_timestamp}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet file: {parquet_path} with {len(df)} rows")


def load_pending_order(folder):
    """Load pending order from JSON file if it exists."""
    pending_order_file = os.path.join(folder, 'pending_order.json')
    if os.path.exists(pending_order_file):
        try:
            with open(pending_order_file, 'r') as f:
                pending_order = json.load(f)
            logger.info(f"Loaded pending_order from {pending_order_file}")
            return pending_order, pending_order_file
        except Exception as e:
            logger.error(f"Error loading pending_order from {pending_order_file}: {e}")
            os.remove(pending_order_file)
    return None, pending_order_file


def save_pending_order(pending_order, pending_order_file):
    """Save pending order to JSON file."""
    with open(pending_order_file, 'w') as f:
        json.dump(pending_order, f)
    logger.info(f"Saved pending_order to {pending_order_file}")


def apply_pending_order_to_df(df, pending_order, pending_order_file, symbol, interval):
    """Apply pending trade details to the last row of the DataFrame and remove the file."""
    for key in ['Actual_Buy', 'Actual_Sell', 'Order_ID', 'Trade_Qty', 'Trade_Price', 'Trade_Fee', 'Trade_Time']:
        if key in pending_order:
            df.loc[df.index[-1], key] = pending_order[key]
    logger.info(f"Applied pending trade details to {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}")
    os.remove(pending_order_file)
    logger.info(f"Removed pending_order file {pending_order_file}")


def fetch_balances_and_set(df, symbol):
    """Fetch account balances and set them in the DataFrame."""
    btc_balance, eth_balance, ada_balance, sol_balance, usdt_balance = Checking_Balance(session_demo)
    coin = symbol[:-4]
    coin_balance = {'BTC': btc_balance, 'ETH': eth_balance, 'ADA': ada_balance}.get(coin, 0)
    if usdt_balance is not None:
        df['USDT_Balance'] = usdt_balance
        logger.info(f"Set USDT_Balance to {usdt_balance} for {symbol}")
    else:
        df['USDT_Balance'] = None
        logger.warning(f"Failed to fetch USDT balance for {symbol}, setting to None")
    df[f'{coin}_Balance'] = coin_balance
    for col in ORDER_COLUMNS + ['Stoploss_Trigger']:
        if col not in df.columns:
            df[col] = 0 if col in ['Actual_Buy', 'Actual_Sell', 'Stoploss_Trigger', 'Trade_Qty'] else None


def place_buy_order_and_store_pending(symbol, interval, pending_order_file, close_price, stoploss, usdt_balance):
    """Place a market buy order using quoteCoin with risk_amount and store pending trade details."""
    try:
        if usdt_balance <= 0:
            logger.info(f"No USDT balance to buy for {symbol} {interval}")
            return None

        # Calculate loss_per_unit
        loss_per_unit = close_price - stoploss
        if loss_per_unit <= 0:
            logger.warning(f"Invalid loss_per_unit ({loss_per_unit}) for {symbol} {interval}, no buy order placed")
            return None

        # Calculate risk amount
        risk_amount = RISK_PERC * usdt_balance

        # Calculate base quantity
        quantity_base = risk_amount / loss_per_unit

        # Calculate estimated USDT cost
        estimated_usdt = quantity_base * close_price

        if estimated_usdt >= usdt_balance:
            buy_usdt = usdt_balance
        else:
            buy_usdt = estimated_usdt

        # Round down to 2 decimal places (USDT precision)
        rounded_qty = round_down(buy_usdt, 2)
        if rounded_qty <= 0:
            logger.info(f"Rounded quantity for {symbol} {interval} is 0, no buy order placed")
            return None

        logger.info(f"Attempting to place buy order for {symbol} {interval} with quantity {rounded_qty} (quoteCoin)")

        response = Placing_Market_Order(
            session=session_demo,
            category="spot",
            symbol=symbol,
            side="Buy",
            qty=str(rounded_qty),  # Use determined buy_usdt as quoteCoin quantity
            marketUnit="quoteCoin",
        )
        if response['retCode'] == 0:
            time.sleep(10)  # Wait for order to process
            order_id = response['result']['orderId']
            trade_details = get_trade_history(session_demo, category="spot", symbol=symbol, orderId=order_id)
            if trade_details:
                logger.info(
                    f"Stored pending buy trade details for {symbol} {interval}: "
                    f"Order ID {trade_details['Order_ID']}, Trade Qty {trade_details['Trade_Qty']}, "
                    f"Trade Fee {trade_details['Trade_Fee']}"
                )
                save_pending_order(trade_details, pending_order_file)
                return trade_details
            else:
                logger.warning(f"No trade execution details retrieved for {symbol} {interval} orderId {order_id}")
        else:
            logger.error(
                f"Failed to place Buy Order for {symbol} {interval}: {response.get('retMsg', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error placing Buy Order for {symbol} {interval}: {e}")
    return None


def place_sell_order_and_store_pending(symbol, interval, pending_order_file, df):
    """Place a market sell order using the base coin quantity from trade history minus trade fee and store pending trade details."""
    try:
        # Check if there's an active buy position with trade quantity
        buy_cum = df['Actual_Buy'].sum()
        sell_cum = df['Actual_Sell'].sum()
        is_in_position = buy_cum > sell_cum

        if not is_in_position:
            logger.info(f"No active buy position to sell for {symbol} {interval}")
            return None

        # Get the trade quantity and fee from the last buy order
        last_buy_row = df[df['Actual_Buy'] == 1].iloc[-1]
        trade_qty = last_buy_row.get('Trade_Qty', 0)
        trade_fee = last_buy_row.get('Trade_Fee', 0) or 0  # Handle None case
        available_qty = trade_qty
        if available_qty <= 0:
            logger.info(
                f"No valid available quantity (Trade_Qty {trade_qty} = {available_qty}) for {symbol} {interval} to sell")
            return None

        # Round down the quantity to the symbol's allowed precision
        precision = get_quantity_precision(symbol)
        rounded_qty = round_down(available_qty, precision)
        if rounded_qty <= 0:
            logger.info(f"Rounded quantity for {symbol} {interval} is 0, no sell order placed")
            return None

        sell_qty = round_down(rounded_qty * (1 - BROKERAGE_SELL), precision)
        if sell_qty <= 0:
            logger.info(f"Rounded sell quantity for {symbol} {interval} is 0 after fee, no sell order placed")
            return None

        logger.info(
            f"Attempting to place sell order for {symbol} {interval} with quantity {sell_qty} (baseCoin) after fee {trade_fee}")
        response = Placing_Market_Order(
            session=session_demo,
            category="spot",
            symbol=symbol,
            side="Sell",
            qty=str(sell_qty),  # Use available_qty after fee
            marketUnit="baseCoin",
        )
        if response['retCode'] == 0:
            time.sleep(10)  # Wait for order to process
            order_id = response['result']['orderId']
            trade_details = get_trade_history(session_demo, category="spot", symbol=symbol, orderId=order_id)
            if trade_details:
                logger.info(
                    f"Stored pending sell trade details for {symbol} {interval}: "
                    f"Order ID {trade_details['Order_ID']}, Trade Qty {trade_details['Trade_Qty']}, "
                    f"Trade Fee {trade_details['Trade_Fee']}"
                )
                save_pending_order(trade_details, pending_order_file)
                return trade_details
            else:
                logger.warning(f"No trade execution details retrieved for {symbol} {interval} orderId {order_id}")
        else:
            logger.error(
                f"Failed to place sell order for {symbol} {interval}: {response.get('retMsg', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error placing sell order for {symbol} {interval}: {e}")
    return None


def apply_trailing_stoploss(df):
    """Apply trailing stoploss logic to the last row if there is an open position."""
    buy_count = df['Actual_Buy'].sum()
    sell_count = df['Actual_Sell'].sum()
    if buy_count > sell_count:
        buy_indices = df[df['Actual_Buy'] == 1].index.tolist()
        buy_index = buy_indices[-1]
        initial_sl = df.at[buy_index, 'Stoploss']
        post_buy_lm = df.loc[buy_index + 1:, 'LM_Low_window_1_CS']
        non_zero_lm = post_buy_lm[post_buy_lm != 0]
        if not non_zero_lm.empty:
            max_lm = non_zero_lm.max()
            trailing_sl = max(initial_sl, max_lm)
        else:
            trailing_sl = initial_sl
        df.at[df.index[-1], 'Stoploss'] = trailing_sl
        if df.at[df.index[-1], 'close'] <= trailing_sl:
            df.at[df.index[-1], 'Stoploss_Trigger'] = 1
        else:
            df.at[df.index[-1], 'Stoploss_Trigger'] = 0
    return df


def process_and_save_df(df, symbol, interval, folder, pending_order, pending_order_file, is_realtime=False):
    """Process DataFrame with indicators and strategy, collect potential order, and save."""
    df = compute_indicators(df, symbol, interval)
    if df is not None:
        df = strategy_1(df, symbol, interval)
        if pending_order:
            apply_pending_order_to_df(df, pending_order, pending_order_file, symbol, interval)
            # Call strategy_1 again after applying pending order
            df = strategy_1(df, symbol, interval)
        df = apply_trailing_stoploss(df)
        # Check if this is the initial run
        existing_files = [f for f in os.listdir(folder) if f.startswith('historical_data') and f.endswith('.parquet')]
        is_initial_run = len(existing_files) == 0
        if is_initial_run:
            logger.info(f"Initial run for {symbol} {interval}, no orders will be collected.")
            potential_order = None
        else:
            potential_order = None
            if is_realtime:
                if df.iloc[-1]['Buy_Signal'] == 1 and df.iloc[-1]['USDT_Balance'] > 0:
                    # Collect buy details instead of placing
                    potential_order = {
                        'symbol': symbol,
                        'interval': interval,
                        'action': 'buy',
                        'details': {
                            'close_price': df.iloc[-1]['close'],
                            'stoploss': df.iloc[-1]['Stoploss'],
                            'usdt_balance': df.iloc[-1]['USDT_Balance']
                        }
                    }
                    logger.info(f"Collected potential buy order for {symbol} {interval}")
                elif df.iloc[-1]['Stoploss_Trigger'] == 1 and df.iloc[-1]['Buy_Signal'] != 1:
                    # Collect sell details instead of placing
                    buy_cum = df['Actual_Buy'].sum()
                    sell_cum = df['Actual_Sell'].sum()
                    if buy_cum > sell_cum:
                        last_buy_row = df[df['Actual_Buy'] == 1].iloc[-1]
                        trade_qty = last_buy_row.get('Trade_Qty', 0)
                        trade_fee = last_buy_row.get('Trade_Fee', 0) or 0
                        available_qty = trade_qty
                        if available_qty > 0:
                            potential_order = {
                                'symbol': symbol,
                                'interval': interval,
                                'action': 'sell',
                                'details': {
                                    'available_qty': available_qty
                                }
                            }
                            logger.info(f"Collected potential sell order for {symbol} {interval}")
        # Clear Stoploss for sells without buy signal
        if df.iloc[-1]['Actual_Sell'] == 1 and df.iloc[-1]['Buy_Signal'] != 1:
            df.loc[df.index[-1], 'Stoploss'] = None
            logger.info(
                f"Cleared Stoploss to None for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']} due to executed sell order without Buy_Signal")
        save_parquet(df, folder)
    else:
        potential_order = None
    return df, potential_order


def execute_order(potential_order):
    """Execute a single order with refetched balances."""
    if potential_order is None:
        return
    symbol = potential_order['symbol']
    interval = potential_order['interval']
    action = potential_order['action']
    details = potential_order['details']

    folder = f"{symbol}_{interval}"
    _, pending_order_file = load_pending_order(folder)  # Get file path for storing pending

    # Refetch current balances
    btc_balance, eth_balance, ada_balance, sol_balance, usdt_balance = Checking_Balance(session_demo)
    coin = symbol[:-4]
    coin_balance = {'BTC': btc_balance, 'ETH': eth_balance, 'ADA': ada_balance}.get(coin, 0)

    if action == 'buy':
        close_price = details['close_price']
        stoploss = details['stoploss']
        placed_order = place_buy_order_and_store_pending(symbol, interval, pending_order_file, close_price, stoploss,
                                                         usdt_balance)
        if placed_order:
            logger.info(
                f"Executed buy order for {symbol} {interval} at timestamp {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.info(f"Failed to execute buy order for {symbol} {interval}")
    elif action == 'sell':
        # Reload DataFrame to get latest state (alternative to passing df)
        existing_files = [f for f in os.listdir(folder) if f.startswith('historical_data') and f.endswith('.parquet')]
        if not existing_files:
            logger.error(f"No parquet files found for {symbol} {interval}, cannot execute sell order")
            return
        latest_file = max(existing_files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
        df = pd.read_parquet(os.path.join(folder, latest_file))
        placed_order = place_sell_order_and_store_pending(symbol, interval, pending_order_file, df)
        if placed_order:
            logger.info(
                f"Executed sell order for {symbol} {interval} at timestamp {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.info(f"Failed to execute sell order for {symbol} {interval}")


def update_data(symbol, interval):
    """Fetch, process, and save incremental candlestick data for a symbol and interval. Return potential order."""
    folder = f"{symbol}_{interval}"
    interval_delta = get_interval_delta(interval)
    current_time = datetime.now(timezone.utc) - interval_delta
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # Load pending order if exists
    pending_order, pending_order_file = load_pending_order(folder)

    # Find latest parquet file
    os.makedirs(folder, exist_ok=True)
    existing_files = [f for f in os.listdir(folder) if f.startswith('historical_data') and f.endswith('.parquet')]

    if not existing_files:
        print(f"No parquet files found for {symbol} {interval}. Fetching initial data.")
        start_time = (current_time - timedelta(days=200)).strftime('%Y-%m-%d %H:%M:%S')
        df = get_candlestick_data(session_demo, symbol, start_time, current_time_str, interval)
        if not df.empty:
            df['timestamp'] = df['timestamp'].apply(format_date)
            df = df.iloc[:-1]  # Remove last row (potentially partial)
            fetch_balances_and_set(df, symbol)
            df, potential_order = process_and_save_df(df, symbol, interval, folder, pending_order, pending_order_file)
            return potential_order
        return None

    # Load latest file and get last timestamp
    latest_file = max(existing_files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
    df = pd.read_parquet(os.path.join(folder, latest_file))
    if df.empty:
        print(f"Latest parquet file for {symbol} {interval} is empty. Fetching initial data.")
        start_time = (current_time - timedelta(days=200)).strftime('%Y-%m-%d %H:%M:%S')
        df = get_candlestick_data(session_demo, symbol, start_time, current_time_str, interval)
        if not df.empty:
            df['timestamp'] = df['timestamp'].apply(format_date)
            df = df.iloc[:-1]
            fetch_balances_and_set(df, symbol)
            df, potential_order = process_and_save_df(df, symbol, interval, folder, pending_order, pending_order_file)
            return potential_order
        return None

    last_timestamp = datetime.strptime(df.iloc[-1]['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    if last_timestamp >= current_time:
        print(f"Data for {symbol} {interval} is up to date.")
        return None

    # Calculate time difference and fetch in chunks
    max_candles = 1000
    interval_seconds = get_interval_delta(interval).total_seconds()
    total_seconds = (current_time - last_timestamp).total_seconds()
    total_candles = int(total_seconds / interval_seconds)

    print(f"Fetching new data for {symbol} {interval} from {last_timestamp} to {current_time_str} "
          f"({total_candles} candles needed)...")

    new_start_time = last_timestamp
    potential_order = None
    while new_start_time < current_time:
        chunk_end_time = new_start_time + timedelta(seconds=interval_seconds * (max_candles - 1))
        if chunk_end_time > current_time:
            chunk_end_time = current_time

        new_start_time_str = new_start_time.strftime('%Y-%m-%d %H:%M:%S')
        chunk_end_time_str = chunk_end_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"Fetching chunk from {new_start_time_str} to {chunk_end_time_str}...")
        new_df = get_candlestick_data(session_demo, symbol, new_start_time_str, chunk_end_time_str, interval)

        if not new_df.empty:
            new_df['timestamp'] = new_df['timestamp'].apply(format_date)
            new_df = new_df[~new_df['timestamp'].isin(df['timestamp'])]  # Filter duplicates
            if not new_df.empty:
                fetch_balances_and_set(new_df, symbol)
                last_candle_time = datetime.strptime(new_df.iloc[-1]['timestamp'], '%Y-%m-%d %H:%M:%S').replace(
                    tzinfo=timezone.utc)
                is_last_candle_complete = last_candle_time <= current_time
                end_index = len(new_df) if is_last_candle_complete else len(new_df) - 1

                for i in range(end_index):
                    single_df = new_df.iloc[[i]]
                    updated_df = pd.concat([df, single_df], ignore_index=True)
                    current_time_check = datetime.now(timezone.utc) - interval_delta
                    last_candle_time = datetime.strptime(updated_df.iloc[-1]['timestamp'], '%Y-%m-%d %H:%M:%S').replace(
                        tzinfo=timezone.utc)
                    is_realtime = last_candle_time >= current_time_check - interval_delta
                    updated_df, potential_order = process_and_save_df(updated_df, symbol, interval, folder,
                                                                      pending_order, pending_order_file, is_realtime)
                    df = updated_df
                    pending_order = None  # Clear after applying

                if end_index < len(new_df):
                    print(f"Ignored last row for {symbol} {interval} with timestamp: {new_df.iloc[-1]['timestamp']}")
                else:
                    print(f"All candles processed for {symbol} {interval} up to {new_df.iloc[-1]['timestamp']}")

                new_start_time = datetime.strptime(new_df.iloc[end_index - 1]['timestamp'],
                                                   '%Y-%m-%d %H:%M:%S').replace(
                    tzinfo=timezone.utc) + interval_delta if end_index > 0 else new_start_time + interval_delta
            else:
                print(f"No new unique data in chunk for {symbol} {interval}.")
                new_start_time = chunk_end_time + interval_delta
        else:
            print(
                f"No new data fetched for chunk {new_start_time_str} to {chunk_end_time_str} for {symbol} {interval}.")
            new_start_time = chunk_end_time + interval_delta

    print(f"Completed fetching {total_candles} candles for {symbol} {interval}.")
    return potential_order


def wait_until_next_update(intervals):
    """Wait until the next update based on the shortest interval."""
    now = datetime.now(timezone.utc)
    min_interval = min([get_interval_delta(interval) for interval in intervals], default=timedelta(minutes=1))
    next_update = now.replace(second=0, microsecond=0) + min_interval
    if next_update <= now:
        next_update += min_interval
    sleep_time = (next_update - now).total_seconds()
    print(f"Waiting for {sleep_time:.2f} seconds until {next_update}...")
    time.sleep(sleep_time)


def sort_intervals(intervals):
    """Sort intervals by their duration in descending order."""

    def interval_to_minutes(interval):
        if interval == "D":
            return 24 * 60
        return int(interval)

    return sorted(intervals, key=interval_to_minutes, reverse=True)


def interval_to_minutes(interval):
    """Convert interval to minutes for sorting."""
    if interval == "D":
        return 24 * 60
    return int(interval)


def main():
    """Main function to run the data update loop with parallel computation and sequential order execution."""
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]  # Example with multiple symbols
    intervals = ["1", "5", "15", "30", "60"]  # Example intervals
    intervals = sort_intervals(intervals)
    print(f"Processing intervals in order: {intervals}")

    num_processes = min(multiprocessing.cpu_count(), len(symbols) * len(intervals))
    print(f"Using {num_processes} processes for parallel execution.")

    # Create a dictionary to map symbols to their index for sorting
    symbol_priority = {symbol: index for index, symbol in enumerate(symbols)}

    while True:
        tasks = [(symbol, interval) for symbol in symbols for interval in intervals]
        with multiprocessing.Pool(processes=num_processes) as pool:
            potential_orders = pool.starmap(update_data, tasks)  # Collect list of potential_order dicts or None

        # Filter and sort non-None potential orders: by symbol order in symbols list, then interval duration descending
        pending_orders = [order for order in potential_orders if order is not None]
        pending_orders.sort(
            key=lambda o: (symbol_priority[o['symbol']], -interval_to_minutes(o['interval']))
        )

        # Execute orders sequentially
        for order in pending_orders:
            execute_order(order)

        wait_until_next_update(intervals)
        time.sleep(1)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()