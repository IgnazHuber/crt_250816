from datetime import datetime, timezone
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_candlestick_data(session, symbol, start_time, end_time, interval, category="spot"):
    """
    Fetch candlestick data for a given symbol and interval from Bybit API.

    Args:
        session: Bybit HTTP session
        symbol (str): Trading pair, e.g., 'BTCUSDT'
        start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format
        end_time (str): End time in 'YYYY-MM-DD HH:MM:SS' format
        interval (str): Kline interval, e.g., '1', '5', '15', 'D'
        category (str): Trading category, default 'spot'

    Returns:
        pd.DataFrame: Candlestick data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                     or empty DataFrame if failed
    """
    try:
        # Convert start_time and end_time to millisecond timestamps
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Log the timestamps being sent to the API
        logger.info(
            f"Fetching data for {symbol} {interval} from {start_time} ({start_ms} ms) to {end_time} ({end_ms} ms)")

        # Make API call with explicit limit to avoid fetching too many candles
        response = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            startTime=start_ms,
            endTime=end_ms,
            limit=1000  # Explicitly set limit to 1000 candles
        )

        if response['retCode'] == 0:
            df = pd.DataFrame(response['result']['list'],
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            if df.empty:
                logger.warning(f"No data returned for {symbol} {interval} from {start_time} to {end_time}")
                return pd.DataFrame()

            # Convert timestamp to datetime and filter by requested time range
            df['timestamp'] = df['timestamp'].astype(float).apply(
                lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc)
            )
            # Filter rows to ensure they are within the requested time range
            df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

            # Log the actual time range of the returned data
            if not df.empty:
                logger.info(f"Received {len(df)} candles from {df['timestamp'].iloc[-1]} to {df['timestamp'].iloc[0]}")
            else:
                logger.warning(f"Filtered DataFrame is empty for {symbol} {interval}")

            # Format timestamp to string
            df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            df[['open', 'high', 'low', 'close', 'volume', 'turnover']] = df[
                ['open', 'high', 'low', 'close', 'volume', 'turnover']].astype(float)
            return df[::-1].reset_index(drop=True)  # Reverse to chronological order
        else:
            logger.error(f"Error fetching data for {symbol} {interval}: {response['retMsg']}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Exception fetching data for {symbol} {interval}: {e}")
        return pd.DataFrame()