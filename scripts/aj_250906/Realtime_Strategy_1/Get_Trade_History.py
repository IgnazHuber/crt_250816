# Get_Trade_History.py
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_trade_history(session, category, symbol, orderId):
    """
    Fetch trade execution history for a specific Order ID and return trade details.

    Args:
        session: Bybit HTTP session
        category (str): Trading category, e.g., 'spot', 'linear'
        symbol (str): Trading pair, e.g., 'BTCUSDT'
        orderId (str): Specific Order ID to fetch execution details for

    Returns:
        dict: Trade details or None if failed
    """
    try:
        # Query trade execution history
        response = session.get_executions(
            category=category,
            symbol=symbol,
            orderId=orderId
        )
        if response and response['retCode'] == 0:
            executions = response['result']['list']
            if executions:
                execution = executions[0]  # Take the first execution (spot market typically has one per order)
                actual_key = 'Actual_Buy' if execution['side'] == 'Buy' else 'Actual_Sell'
                return {
                    actual_key: 1,
                    'Order_ID': execution['orderId'],
                    'Trade_Qty': float(execution['execQty']),  # Base coin quantity executed
                    'Trade_Price': float(execution['execPrice']),
                    'Trade_Fee': float(execution['execFee']) if execution['execFee'] else None,
                    'Trade_Time': pd.to_datetime(
                        int(execution['execTime']) / 1000, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                    'Trade_ID': execution['execId']
                }
            else:
                logger.warning(f"No execution details returned for {symbol} orderId {orderId}")
                return None
        else:
            logger.error(
                f"Failed to fetch execution history for {symbol}: {response.get('retMsg', 'Unknown error') if response else 'No response'}")
            return None
    except Exception as e:
        logger.error(f"Error fetching execution history for {symbol} orderId {orderId}: {e}")
        return None