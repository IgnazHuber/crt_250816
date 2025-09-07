# Strategy_1_2.py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def strategy_1(df, symbol, interval):
    """Check for bullish divergence in the second-last row, add Buy_Signal, Stoploss, and monitor Stoploss_Trigger."""
    try:
        if len(df) < 2:
            print(f"DataFrame for {symbol} {interval} has fewer than 2 rows, cannot check bullish divergence.")
            df['Buy_Signal'] = 0
            df['Stoploss'] = 0
            df['Stoploss_Trigger'] = 0
            return df

        # Initialize columns if not present
        if 'Buy_Signal' not in df.columns:
            df['Buy_Signal'] = 0
        if 'Stoploss' not in df.columns:
            df['Stoploss'] = 0
        if 'Stoploss_Trigger' not in df.columns:
            df['Stoploss_Trigger'] = 0
        if 'Actual_Buy' not in df.columns:
            df['Actual_Buy'] = 0
        if 'Actual_Sell' not in df.columns:
            df['Actual_Sell'] = 0

        # Compute if in position based on previous rows (excluding current row)
        buy_cum_prev = df['Actual_Buy'].iloc[:-1].sum()
        sell_cum_prev = df['Actual_Sell'].iloc[:-1].sum()
        is_in_position = buy_cum_prev > sell_cum_prev

        # Check for new Buy_Signal regardless of position
        second_last_row = df.iloc[-2]
        buy_signal_set = False
        if 'CBullD_gen' in second_last_row and second_last_row['CBullD_gen'] == 1:
            df.loc[df.index[-1], 'Buy_Signal'] = 1
            df.loc[df.index[-1], 'Stoploss'] = second_last_row['CBullD_Lower_Low_gen']
            buy_signal_set = True
            logger.info(
                f"CBullD_gen detected, Buy Signal generated for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}, "
                f"Stoploss set to {second_last_row['CBullD_Lower_Low_gen']}"
            )
        elif 'CBullD_neg_MACD' in second_last_row and second_last_row['CBullD_neg_MACD'] == 1:
            df.loc[df.index[-1], 'Buy_Signal'] = 1
            df.loc[df.index[-1], 'Stoploss'] = second_last_row['CBullD_Lower_Low_neg_MACD']
            buy_signal_set = True
            logger.info(
                f"CBullD_neg_MACD detected, Buy Signal generated for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}, "
                f"Stoploss set to {second_last_row['CBullD_Lower_Low_neg_MACD']}"
            )
        elif 'HBullD_gen' in second_last_row and second_last_row['HBullD_gen'] == 1:
            df.loc[df.index[-1], 'Buy_Signal'] = 1
            df.loc[df.index[-1], 'Stoploss'] = second_last_row['HBullD_Higher_Low_gen']
            buy_signal_set = True
            logger.info(
                f"HBullD_gen detected, Buy Signal generated for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}, "
                f"Stoploss set to {second_last_row['HBullD_Higher_Low_gen']}"
            )
        elif 'HBullD_neg_MACD' in second_last_row and second_last_row['HBullD_neg_MACD'] == 1:
            df.loc[df.index[-1], 'Buy_Signal'] = 1
            df.loc[df.index[-1], 'Stoploss'] = second_last_row['HBullD_Higher_Low_neg_MACD']
            buy_signal_set = True
            logger.info(
                f"HBullD_neg_MACD detected, Buy Signal generated for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}, "
                f"Stoploss set to {second_last_row['HBullD_Higher_Low_neg_MACD']}"
            )
        else:
            logger.info(
                f"No bullish divergence detected for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}"
            )

        # Monitor stoploss only if in position or a buy just occurred in the current row, and no sell has occurred in the current row
        if (is_in_position or df['Actual_Buy'].iloc[-1] == 1) and df['Actual_Sell'].iloc[-1] != 1:
            # If no new Buy_Signal, propagate the previous Stoploss
            if not buy_signal_set:
                df.loc[df.index[-1], 'Stoploss'] = df['Stoploss'].iloc[-2]
            # Check if close price is below Stoploss
            if df['close'].iloc[-1] < df['Stoploss'].iloc[-1]:
                df.loc[df.index[-1], 'Stoploss_Trigger'] = 1
                logger.info(
                    f"Stoploss triggered for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']}, "
                    f"Close {df['close'].iloc[-1]} < Stoploss {df['Stoploss'].iloc[-1]}"
                )
        else:
            if df['Buy_Signal'].iloc[-1] == 0:
                df.loc[df.index[-1], 'Stoploss'] = 0
            df.loc[df.index[-1], 'Stoploss_Trigger'] = 0
            logger.info(
                f"Skipping stoploss monitoring for {symbol} {interval} at timestamp {df.iloc[-1]['timestamp']} "
                f"due to {'existing sell order' if df['Actual_Sell'].iloc[-1] == 1 else 'no active position or buy signal'}"
            )

        return df
    except Exception as e:
        print(f"Error in strategy_1 for {symbol} {interval}: {e}")
        df['Buy_Signal'] = 0
        df['Stoploss'] = 0
        df['Stoploss_Trigger'] = 0
        return df