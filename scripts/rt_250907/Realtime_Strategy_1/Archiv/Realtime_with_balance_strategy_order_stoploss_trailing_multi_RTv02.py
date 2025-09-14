# Realtime_with_balance_strategy_order_stoploss_trailing_multi_RTv02.py
# Hinweis: Additiv zu v01, keine Features entfernt. Default jetzt threadsicher (ThreadPool).
import argparse
import time
import pandas as pd
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import json
import math
import logging
import threading
import signal
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Projekt-Imports (unverändert) ---
from config import BYBIT_DEMO_API_KEY, BYBIT_DEMO_API_SECRET
from pybit.unified_trading import HTTP
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
# from CBearDivg_analysis_vectorized import CBearDivg_analysis
# from HBearDivg_analysis_vectorized import HBearDivg_analysis
# from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
# from Goldenratio_vectorized import calculate_golden_ratios
# from Support_Resistance_vectorized import calculate_support_levels
# from Trendline_Up_Support_vectorized import calc_TL_Up_Support
# from Trendline_Up_Resistance_vectorized import calc_TL_Up_Resistance
# from Trendline_Down_Resistance_vectorized import calc_TL_Down_Resistance
from Get_Account_Balance import Checking_Balance
from Placing_Market_Order import Placing_Market_Order
from Get_candlestick_data import get_candlestick_data
from Get_Trade_History import get_trade_history
from Strategy_1 import strategy_1

# --- Multiprocessing-Option (nur wenn explizit gewählt) ---
import multiprocessing

# =========================
# Grundeinstellungen
# =========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

LOG = logging.getLogger("RTv02")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)

# Balance-Handling
BALANCE_MODE = "auto"      # "auto" | "on" | "off"
BALANCE_AVAILABLE = True   # wird bei Fehlern auf False gesetzt


# Risiko/Fees (unverändert)
RISK_PERC = 0.01
BROKERAGE_BUY = 0.001
BROKERAGE_SELL = 0.001

# Bybit-HTTP-Session (global; wird in Threads wiederverwendet)
session_demo = HTTP(api_key=BYBIT_DEMO_API_KEY, api_secret=BYBIT_DEMO_API_SECRET, demo=True)

ORDER_COLUMNS = [
    'Order_ID','Order_Type','Order_Qty','Order_Price','Order_Status','Order_Fees',
    'Order_Created_Time','Actual_Buy','Actual_Sell','Trade_Qty','Trade_Price','Trade_Fee','Trade_Time'
]

SYMBOL_PRECISION = {'BTCUSDT': 6, 'ETHUSDT': 3, 'ADAUSDT': 1}

def get_quantity_precision(symbol: str) -> int:
    return SYMBOL_PRECISION.get(symbol, 4)

def round_down(value: float, decimals: int) -> float:
    if value <= 0:
        return 0.0
    f = 10 ** decimals
    return math.floor(value * f) / f

def get_interval_delta(interval: str) -> timedelta:
    if interval == "D":
        return timedelta(days=1)
    return timedelta(minutes=int(interval))

def format_date(date_value) -> str:
    if isinstance(date_value, str):
        try:
            date_obj = pd.to_datetime(date_value)
        except ValueError:
            return date_value
    else:
        date_obj = date_value
    return date_obj.strftime('%Y-%m-%d %H:%M:%S')

def compute_indicators(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame | None:
    try:
        df = Initialize_RSI_EMA_MACD(df)
        Candlestick_Type(df)
        Level_1_Max_Min(df)

        # Intervallabhängige Toleranzen (wie v01)
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
        LOG.exception("Indicator-Berechnung fehlgeschlagen (%s %s): %s", symbol, interval, e)
        return None

def save_parquet(df: pd.DataFrame, folder: Path) -> Path | None:
    if df is None or df.empty or len(df) < 1:
        LOG.warning("DataFrame leer, kein Save.")
        return None
    folder.mkdir(parents=True, exist_ok=True)
    last_ts = str(df.iloc[-1]['timestamp'])
    file_ts = last_ts.replace(':', '-').replace(' ', '_')
    p = folder / f"historical_data_{file_ts}.parquet"
    try:
        df.to_parquet(p, index=False)
        LOG.info("Parquet gespeichert: %s (rows=%d)", p, len(df))
        return p
    except Exception as e:
        LOG.exception("Parquet-Speichern fehlgeschlagen (%s): %s", p, e)
        return None

def load_pending_order(folder: Path):
    f = folder / 'pending_order.json'
    if f.exists():
        try:
            with open(f, 'r') as fh:
                pending = json.load(fh)
            LOG.info("pending_order geladen: %s", f)
            return pending, f
        except Exception as e:
            LOG.error("pending_order defekt (%s): %s -> Datei wird entfernt", f, e)
            try: f.unlink()
            except Exception: pass
    return None, f

def save_pending_order(pending_order: dict, f: Path):
    with open(f, 'w') as fh:
        json.dump(pending_order, fh)
    LOG.info("pending_order gespeichert: %s", f)

def apply_pending_order_to_df(df: pd.DataFrame, pending_order: dict, f: Path, symbol: str, interval: str):
    for key in ['Actual_Buy','Actual_Sell','Order_ID','Trade_Qty','Trade_Price','Trade_Fee','Trade_Time']:
        if key in pending_order:
            df.loc[df.index[-1], key] = pending_order[key]
    LOG.info("pending_order angewandt auf %s %s @ %s", symbol, interval, df.iloc[-1]['timestamp'])
    try:
        f.unlink()
        LOG.info("pending_order entfernt: %s", f)
    except Exception:
        pass

def _set_balance_columns(df: pd.DataFrame, symbol: str, coin_balance, usdt):
    df['USDT_Balance'] = usdt
    coin = symbol[:-4]
    df[f'{coin}_Balance'] = coin_balance
    for col in ORDER_COLUMNS + ['Stoploss_Trigger']:
        if col not in df.columns:
            df[col] = 0 if col in ['Actual_Buy','Actual_Sell','Stoploss_Trigger','Trade_Qty'] else None

def fetch_balances_and_set(df: pd.DataFrame, symbol: str) -> bool:
    global BALANCE_AVAILABLE
    try:
        if BALANCE_MODE == "off" or (BALANCE_MODE == "auto" and not BALANCE_AVAILABLE):
            _set_balance_columns(df, symbol, None, None)
            return False

        btc, eth, ada, sol, usdt = Checking_Balance(session_demo)
        coin = symbol[:-4]
        coin_balance = {'BTC': btc, 'ETH': eth, 'ADA': ada, 'SOL': sol}.get(coin, None)
        _set_balance_columns(df, symbol, coin_balance, usdt)
        BALANCE_AVAILABLE = True
        LOG.info("Balances gesetzt (%s): USDT=%s", symbol, usdt)
        return True

    except InvalidRequestError as e:
        LOG.warning("Balance-API deaktiviert (InvalidRequestError): %s", e)
        _set_balance_columns(df, symbol, None, None)
        BALANCE_AVAILABLE = False
        return False
    except Exception as e:
        LOG.warning("Balance-API Problem (%s): %s -> setze None und fahre fort", type(e).__name__, e)
        _set_balance_columns(df, symbol, None, None)
        BALANCE_AVAILABLE = False
        return False

def place_buy_order_and_store_pending(symbol, interval, pending_file: Path, close_price, stoploss, usdt_balance):
    try:
        if not usdt_balance or usdt_balance <= 0:
            LOG.info("Kein USDT-Balance für Kauf %s %s", symbol, interval)
            return None
        loss_per_unit = close_price - stoploss
        if loss_per_unit <= 0:
            LOG.warning("loss_per_unit <= 0 (%s %s) -> kein Kauf", symbol, interval)
            return None
        risk_amount = RISK_PERC * usdt_balance
        quantity_base = risk_amount / loss_per_unit
        est_usdt = quantity_base * close_price
        buy_usdt = usdt_balance if est_usdt >= usdt_balance else est_usdt
        rounded_qty = round_down(buy_usdt, 2)
        if rounded_qty <= 0:
            LOG.info("Gerundete USDT=0 (%s %s) -> kein Kauf", symbol, interval)
            return None

        LOG.info("Sende Buy (quoteCoin): %s %s qty=%s", symbol, interval, rounded_qty)
        resp = Placing_Market_Order(
            session=session_demo, category="spot", symbol=symbol,
            side="Buy", qty=str(rounded_qty), marketUnit="quoteCoin"
        )
        if resp.get('retCode') == 0:
            time.sleep(10)
            order_id = resp['result']['orderId']
            td = get_trade_history(session_demo, category="spot", symbol=symbol, orderId=order_id)
            if td:
                LOG.info("Buy ausgeführt %s %s: ID=%s qty=%s fee=%s",
                         symbol, interval, td.get('Order_ID'), td.get('Trade_Qty'), td.get('Trade_Fee'))
                save_pending_order(td, pending_file)
                return td
            else:
                LOG.warning("Keine Ausführungsdetails erhalten (%s %s)", symbol, interval)
        else:
            LOG.error("Buy fehlgeschlagen %s %s: %s", symbol, interval, resp.get('retMsg', 'Unknown'))
    except Exception as e:
        LOG.exception("Buy-Fehler %s %s: %s", symbol, interval, e)
    return None

def place_sell_order_and_store_pending(symbol, interval, pending_file: Path, df: pd.DataFrame):
    try:
        buy_cum = df['Actual_Buy'].sum()
        sell_cum = df['Actual_Sell'].sum()
        if not (buy_cum > sell_cum):
            LOG.info("Keine offene Long-Position zum Verkaufen (%s %s)", symbol, interval)
            return None
        last_buy_row = df[df['Actual_Buy'] == 1].iloc[-1]
        trade_qty = last_buy_row.get('Trade_Qty', 0) or 0
        trade_fee = last_buy_row.get('Trade_Fee', 0) or 0
        if trade_qty <= 0:
            LOG.info("Keine gültige Menge für Sell (%s %s)", symbol, interval)
            return None
        prec = get_quantity_precision(symbol)
        sell_qty = round_down(round_down(trade_qty, prec) * (1 - BROKERAGE_SELL), prec)
        if sell_qty <= 0:
            LOG.info("Sell-Menge nach Fee=0 (%s %s)", symbol, interval)
            return None

        LOG.info("Sende Sell (baseCoin): %s %s qty=%s fee=%s", symbol, interval, sell_qty, trade_fee)
        resp = Placing_Market_Order(
            session=session_demo, category="spot", symbol=symbol,
            side="Sell", qty=str(sell_qty), marketUnit="baseCoin"
        )
        if resp.get('retCode') == 0:
            time.sleep(10)
            order_id = resp['result']['orderId']
            td = get_trade_history(session_demo, category="spot", symbol=symbol, orderId=order_id)
            if td:
                LOG.info("Sell ausgeführt %s %s: ID=%s qty=%s fee=%s",
                         symbol, interval, td.get('Order_ID'), td.get('Trade_Qty'), td.get('Trade_Fee'))
                save_pending_order(td, pending_file)
                return td
            else:
                LOG.warning("Keine Ausführungsdetails (Sell) %s %s", symbol, interval)
        else:
            LOG.error("Sell fehlgeschlagen %s %s: %s", symbol, interval, resp.get('retMsg', 'Unknown'))
    except Exception as e:
        LOG.exception("Sell-Fehler %s %s: %s", symbol, interval, e)
    return None

def apply_trailing_stoploss(df: pd.DataFrame) -> pd.DataFrame:
    buy_count = df['Actual_Buy'].sum()
    sell_count = df['Actual_Sell'].sum()
    if buy_count > sell_count:
        buy_idx = df[df['Actual_Buy'] == 1].index.tolist()[-1]
        initial_sl = df.at[buy_idx, 'Stoploss']
        post_lm = df.loc[buy_idx + 1:, 'LM_Low_window_1_CS']
        non_zero = post_lm[post_lm != 0]
        trailing_sl = max(initial_sl, non_zero.max()) if not non_zero.empty else initial_sl
        df.at[df.index[-1], 'Stoploss'] = trailing_sl
        df.at[df.index[-1], 'Stoploss_Trigger'] = 1 if df.at[df.index[-1], 'close'] <= trailing_sl else 0
    return df

def process_and_save_df(df: pd.DataFrame, symbol: str, interval: str, folder: Path,
                        pending_order: dict | None, pending_file: Path, is_realtime: bool = False):
    df = compute_indicators(df, symbol, interval)
    potential_order = None
    if df is not None:
        df = strategy_1(df, symbol, interval)
        if pending_order:
            apply_pending_order_to_df(df, pending_order, pending_file, symbol, interval)
            df = strategy_1(df, symbol, interval)
        df = apply_trailing_stoploss(df)

        existing = [p for p in folder.glob("historical_data*.parquet")]
        is_initial = len(existing) == 0

        if is_initial:
            LOG.info("Initialer Lauf %s %s: keine Orders sammeln", symbol, interval)
        else:
            if is_realtime:
                if df.iloc[-1]['Buy_Signal'] == 1 and (df.iloc[-1]['USDT_Balance'] or 0) > 0:
                    potential_order = {
                        'symbol': symbol, 'interval': interval, 'action': 'buy',
                        'details': {
                            'close_price': float(df.iloc[-1]['close']),
                            'stoploss': float(df.iloc[-1]['Stoploss']),
                            'usdt_balance': float(df.iloc[-1]['USDT_Balance'])
                        }
                    }
                    LOG.info("Pot. BUY gesammelt %s %s", symbol, interval)
                elif df.iloc[-1]['Stoploss_Trigger'] == 1 and df.iloc[-1]['Buy_Signal'] != 1:
                    buy_cum = df['Actual_Buy'].sum()
                    sell_cum = df['Actual_Sell'].sum()
                    if buy_cum > sell_cum:
                        last_buy = df[df['Actual_Buy'] == 1].iloc[-1]
                        available_qty = float(last_buy.get('Trade_Qty', 0) or 0)
                        if available_qty > 0:
                            potential_order = {
                                'symbol': symbol, 'interval': interval, 'action': 'sell',
                                'details': {'available_qty': available_qty}
                            }
                            LOG.info("Pot. SELL gesammelt %s %s", symbol, interval)

        if df.iloc[-1]['Actual_Sell'] == 1 and df.iloc[-1]['Buy_Signal'] != 1:
            df.loc[df.index[-1], 'Stoploss'] = None
            LOG.info("Stoploss auf None gesetzt (Sell ohne Buy_Signal) %s %s @ %s",
                     symbol, interval, df.iloc[-1]['timestamp'])

        save_parquet(df, folder)
    return df, potential_order

def execute_order(potential_order: dict, outdir: Path):
    if not potential_order:
        return
    symbol = potential_order['symbol']
    interval = potential_order['interval']
    action = potential_order['action']
    details = potential_order['details']
    folder = outdir / f"{symbol}_{interval}"
    _, pending_file = load_pending_order(folder)

    # Rebalances vor Order
    btc, eth, ada, sol, usdt = Checking_Balance(session_demo)

    if action == 'buy':
        placed = place_buy_order_and_store_pending(
            symbol, interval, pending_file,
            details['close_price'], details['stoploss'], usdt
        )
        if placed:
            LOG.info("BUY ausgeführt %s %s @ %s", symbol, interval,
                     datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    elif action == 'sell':
        existing = [p for p in folder.glob("historical_data*.parquet")]
        if not existing:
            LOG.error("Kein Parquet für SELL (%s %s) -> Abbruch", symbol, interval)
            return
        latest = max(existing, key=lambda p: p.stat().st_ctime)
        df = pd.read_parquet(latest)
        placed = place_sell_order_and_store_pending(symbol, interval, pending_file, df)
        if placed:
            LOG.info("SELL ausgeführt %s %s @ %s", symbol, interval,
                     datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))

def update_data(symbol: str, interval: str, outdir: Path) -> dict | None:
    """Ein Update-Zyklus für (symbol, interval). Gibt ggf. eine potenzielle Order zurück."""
    try:
        folder = outdir / f"{symbol}_{interval}"
        interval_delta = get_interval_delta(interval)
        current_time = datetime.now(timezone.utc) - interval_delta
        current_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

        pending_order, pending_file = load_pending_order(folder)

        folder.mkdir(parents=True, exist_ok=True)
        existing = [p for p in folder.glob("historical_data*.parquet")]

        if not existing:
            print(f"No parquet files found for {symbol} {interval}. Fetching initial data.")
            start_time = (current_time - timedelta(days=200)).strftime('%Y-%m-%d %H:%M:%S')
            df = get_candlestick_data(session_demo, symbol, start_time, current_str, interval)
            if not df.empty:
                df['timestamp'] = df['timestamp'].apply(format_date)
                df = df.iloc[:-1]
                fetch_balances_and_set(df, symbol)
                _, pot = process_and_save_df(df, symbol, interval, folder, pending_order, pending_file)
                return pot
            return None

        latest = max(existing, key=lambda p: p.stat().st_ctime)
        df = pd.read_parquet(latest)
        if df.empty:
            print(f"Latest parquet file for {symbol} {interval} is empty. Fetching initial data.")
            start_time = (current_time - timedelta(days=200)).strftime('%Y-%m-%d %H:%M:%S')
            df = get_candlestick_data(session_demo, symbol, start_time, current_str, interval)
            if not df.empty:
                df['timestamp'] = df['timestamp'].apply(format_date)
                df = df.iloc[:-1]
                fetch_balances_and_set(df, symbol)
                _, pot = process_and_save_df(df, symbol, interval, folder, pending_order, pending_file)
                return pot
            return None

        last_ts = datetime.strptime(str(df.iloc[-1]['timestamp']), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        if last_ts >= current_time:
            print(f"Data for {symbol} {interval} is up to date.")
            return None

        max_candles = 1000
        interval_seconds = get_interval_delta(interval).total_seconds()
        total_seconds = (current_time - last_ts).total_seconds()
        total_candles = int(total_seconds / interval_seconds)

        print(f"Fetching new data for {symbol} {interval} from {last_ts} to {current_str} ({total_candles} candles needed)...")

        new_start_time = last_ts
        potential_order = None
        while new_start_time < current_time:
            chunk_end_time = new_start_time + timedelta(seconds=interval_seconds * (max_candles - 1))
            if chunk_end_time > current_time:
                chunk_end_time = current_time

            s = new_start_time.strftime('%Y-%m-%d %H:%M:%S')
            e = chunk_end_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Fetching chunk from {s} to {e}...")
            new_df = get_candlestick_data(session_demo, symbol, s, e, interval)

            if not new_df.empty:
                new_df['timestamp'] = new_df['timestamp'].apply(format_date)
                new_df = new_df[~new_df['timestamp'].isin(df['timestamp'])]
                if not new_df.empty:
                    fetch_balances_and_set(new_df, symbol)
                    last_candle_time = datetime.strptime(str(new_df.iloc[-1]['timestamp']), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    is_last_complete = last_candle_time <= current_time
                    end_index = len(new_df) if is_last_complete else len(new_df) - 1

                    for i in range(end_index):
                        single_df = new_df.iloc[[i]]
                        updated_df = pd.concat([df, single_df], ignore_index=True)
                        current_time_check = datetime.now(timezone.utc) - interval_delta
                        last_candle_time = datetime.strptime(str(updated_df.iloc[-1]['timestamp']), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                        is_realtime = last_candle_time >= current_time_check - interval_delta
                        updated_df, potential_order = process_and_save_df(
                            updated_df, symbol, interval, folder, pending_order, pending_file, is_realtime
                        )
                        df = updated_df
                        pending_order = None

                    if end_index < len(new_df):
                        print(f"Ignored last row for {symbol} {interval} with timestamp: {new_df.iloc[-1]['timestamp']}")
                    else:
                        print(f"All candles processed for {symbol} {interval} up to {new_df.iloc[-1]['timestamp']}")

                    new_start_time = datetime.strptime(str(new_df.iloc[end_index - 1]['timestamp']), '%Y-%m-%d %H:%M:%S').replace(
                        tzinfo=timezone.utc) + get_interval_delta(interval) if end_index > 0 else new_start_time + get_interval_delta(interval)
                else:
                    print(f"No new unique data in chunk for {symbol} {interval}.")
                    new_start_time = chunk_end_time + get_interval_delta(interval)
            else:
                print(f"No new data fetched for chunk {s} to {e} for {symbol} {interval}.")
                new_start_time = chunk_end_time + get_interval_delta(interval)

        print(f"Completed fetching {total_candles} candles for {symbol} {interval}.")
        return potential_order
    except Exception as e:
        LOG.exception("update_data Fehler (%s %s): %s", symbol, interval, e)
        return None

def wait_until_next_update(intervals: list[str], stop_event: threading.Event, stopfile: Path):
    """Unterbrechbare Wartezeit bis zum nächsten kleinsten Intervall."""
    now = datetime.now(timezone.utc)
    min_interval = min([get_interval_delta(i) for i in intervals], default=timedelta(minutes=1))
    next_update = now.replace(second=0, microsecond=0) + min_interval
    if next_update <= now:
        next_update += min_interval
    sleep_total = (next_update - now).total_seconds()
    LOG.info("Warte %.2fs bis %s (Stop-Datei: %s)", sleep_total, next_update, stopfile)
    # in kleinen Schritten schlafen, damit STOP schnell greift
    slept = 0.0
    step = 0.5
    while slept < sleep_total:
        if stop_event.is_set() or stopfile.exists():
            break
        time.sleep(step)
        slept += step

def sort_intervals(intervals: list[str]) -> list[str]:
    def to_minutes(interval: str):
        return 24 * 60 if interval == "D" else int(interval)
    return sorted(intervals, key=to_minutes, reverse=True)

def interval_to_minutes(interval: str) -> int:
    return 24 * 60 if interval == "D" else int(interval)

def parse_args():
    p = argparse.ArgumentParser(description="Realtime trading loop (RTv02)")
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,ADAUSDT",
                   help="Kommagetrennt, z.B. BTCUSDT,ETHUSDT")
    p.add_argument("--intervals", type=str, default="1,5,15,30,60", help="Kommagetrennt, z.B. 1,5,15,30,60,D")
    p.add_argument("--executor", choices=["thread", "process"], default="thread",
                   help="Parallelisierung: thread (Default) oder process")
    p.add_argument("--outdir", type=str, default=".", help="Ausgabeverzeichnis (Ordner pro Symbol_Intervall)")
    p.add_argument("--stopfile", type=str, default="STOP", help="Wenn diese Datei existiert -> sauber beenden")
    p.add_argument("--once", action="store_true", help="Nur einen Update-Zyklus laufen")
    p.add_argument("--loglevel", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return p.parse_args()

def main():
    args = parse_args()

    global BALANCE_MODE
    BALANCE_MODE = args.balance
    LOG.info("Balance-Mode: %s", BALANCE_MODE)


    logging.getLogger().setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    intervals = sort_intervals([i.strip() for i in args.intervals.split(",") if i.strip()])
    outdir = Path(args.outdir).resolve()
    stopfile = Path(args.stopfile).resolve()

    print(f"Output-Verzeichnis: {outdir}")
    print(f"Processing intervals in order: {intervals}")

    num_workers = min(max(1, (os.cpu_count() or 2)), max(1, len(symbols) * len(intervals)))
    print(f"Using {num_workers} workers ({args.executor}).")

    symbol_priority = {s: idx for idx, s in enumerate(symbols)}
    stop_event = threading.Event()

    def handle_sig(sig, frame):
        LOG.warning("Signal %s empfangen -> Stop", sig)
        stop_event.set()
    try:
        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)
    except Exception:
        pass  # Windows kann SIGTERM u.U. nicht setzen

    # Optional: für Process-Executor Windows-Startmethode
    if args.executor == "process":
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # Ein Durchlauf = alle (symbol, interval) abarbeiten, dann Orders sequentiell ausführen
    def one_cycle():
        tasks = [(sym, itv) for sym in symbols for itv in intervals]

        pending_orders = []
        if args.executor == "thread":
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futs = {ex.submit(update_data, s, i, outdir): (s, i) for s, i in tasks}
                for fut in as_completed(futs):
                    try:
                        res = fut.result()
                        if res is not None:
                            pending_orders.append(res)
                    except Exception as e:
                        s, i = futs[fut]
                        LOG.exception("Task-Fehler (%s %s): %s", s, i, e)
        else:
            # Nur auf Wunsch: Prozesspool (kann Pickle-Probleme verursachen, daher Default=thread)
            with multiprocessing.Pool(processes=num_workers) as pool:
                try:
                    results = pool.starmap(update_data, [(s, i, outdir) for s, i in tasks])
                    pending_orders = [r for r in results if r is not None]
                except Exception as e:
                    LOG.exception("ProcessPool-Fehler: %s", e)
                    pending_orders = []

        # Sortierung: Symbol-Reihenfolge, dann Intervall absteigend
        pending_orders.sort(key=lambda o: (symbol_priority[o['symbol']], -interval_to_minutes(o['interval'])))
        for order in pending_orders:
            execute_order(order, outdir)

    # Hauptloop
    try:
        while not stop_event.is_set() and not stopfile.exists():
            one_cycle()
            if args.once:
                break
            wait_until_next_update(intervals, stop_event, stopfile)
    finally:
        LOG.info("Beende sauber. (Stop-Datei vorhanden? %s)", stopfile.exists())

if __name__ == "__main__":
    main()
