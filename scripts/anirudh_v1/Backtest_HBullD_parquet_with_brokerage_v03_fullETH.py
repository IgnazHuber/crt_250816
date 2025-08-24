import pandas as pd
import os
from datetime import datetime
import numpy as np
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, Future

# Plot/Init-Imports belassen (Features nicht entfernen)
import finplot as fplt  # noqa: F401
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD  # noqa: F401


# ---------- Laufzeit ----------
class Timings:
    def __init__(self):
        self.sections = {}

    def add(self, key, seconds):
        self.sections[key] = self.sections.get(key, 0.0) + seconds

    @staticmethod
    def _fmt(s): return f"{s:,.3f}s"

    def print_aggregated_only(self, total_seconds):
        print(f"\nGesamtlaufzeit: {total_seconds:,.3f}s\n")
        print("================ Laufzeit-Übersicht (aggregiert) ================")
        total = 0.0
        for k, v in sorted(self.sections.items(), key=lambda x: -x[1]):
            print(f"{k:<30s}: {self._fmt(v)}")
            total += v
        print(f"{'-'*30}: {'-'*10}")
        print(f"{'Summe gemessener Abschnitte':<30s}: {self._fmt(total)}")


# ---------- Pandas Anzeige ----------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ---------- Nur benötigte Spalten ----------
NEEDED_COLS = [
    'date', 'open', 'close', 'EMA_50', 'EMA_200',
    'HBullD_gen', 'HBullD_Lower_Low_RSI_gen', 'HBullD_Higher_Low_RSI_gen', 'HBullD_Higher_Low_gen',
    'HBullD_neg_MACD', 'HBullD_Lower_Low_RSI_neg_MACD', 'HBullD_Higher_Low_RSI_neg_MACD', 'HBullD_Higher_Low_neg_MACD',
    'CBullD_gen', 'CBullD_Lower_Low_RSI_gen', 'CBullD_Higher_Low_RSI_gen', 'CBullD_Lower_Low_gen',
    'CBullD_neg_MACD', 'CBullD_Lower_Low_RSI_neg_MACD', 'CBullD_Higher_Low_RSI_neg_MACD', 'CBullD_Lower_Low_neg_MACD',
    'CBullD_x2', 'CBullD_x2_Lower_Low',
    'LM_Low_window_1_CS'
]


# ---------- Effizientes Tail-Lesen aus Parquet ----------
def read_last_n_rows_parquet(path, n=100, columns=None):
    """
    Liest die letzten n Zeilen effizient aus Parquet (nur benötigte Rowgroups/Spalten).
    Identisch zu pd.read_parquet(...)[-n:], aber ohne Voll-Load.
    """
    import pyarrow.parquet as pq
    t0 = time.perf_counter()
    pf = pq.ParquetFile(path, memory_map=True)
    md = pf.metadata
    num_row_groups = md.num_row_groups

    rows_needed = n
    row_groups_to_read = []
    for rg in range(num_row_groups - 1, -1, -1):
        rg_rows = md.row_group(rg).num_rows
        row_groups_to_read.append(rg)
        rows_needed -= rg_rows
        if rows_needed <= 0:
            break
    row_groups_to_read.reverse()

    tables = [pf.read_row_group(rg, columns=columns) for rg in row_groups_to_read]
    tbl = tables[0].concat(tables[1:]) if len(tables) > 1 else tables[0]
    if tbl.num_rows > n:
        tbl = tbl.slice(tbl.num_rows - n, n)
    df = tbl.to_pandas(types_mapper=None, use_threads=True, split_blocks=True)
    return df, (time.perf_counter() - t0)


# ---------- Fachlogik (unverändert) ----------
def get_last_nonzero_lm_low_fast(series, dates):
    arr = series.to_numpy()
    nz = np.flatnonzero(arr != 0)
    if nz.size == 0:
        return 0, 0
    idx = nz[-1]
    return arr[idx], dates.iloc[idx]

def Buy_Signal_Check(df):
    if df['HBullD_gen'].iloc[-2] == 1 and 70 > df['HBullD_Lower_Low_RSI_gen'].iloc[-2] > 40 and 70 > df['HBullD_Higher_Low_RSI_gen'].iloc[-2] > 40 and df['EMA_50'].iloc[-2] > df['EMA_200'].iloc[-2]:
        return 1, df['HBullD_Higher_Low_gen'].iloc[-2]
    elif df['HBullD_neg_MACD'].iloc[-2] == 1 and 70 > df['HBullD_Lower_Low_RSI_neg_MACD'].iloc[-2] > 40 and 70 > df['HBullD_Higher_Low_RSI_neg_MACD'].iloc[-2] > 40 and df['EMA_50'].iloc[-2] > df['EMA_200'].iloc[-2]:
        return 1, df['HBullD_Higher_Low_neg_MACD'].iloc[-2]
    elif ((df['CBullD_gen'].iloc[-2] == 1 and df['CBullD_neg_MACD'].iloc[-2] == 1) or df['CBullD_gen'].iloc[-2] == 1) and 55 > df['CBullD_Higher_Low_RSI_gen'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_gen'].iloc[-2] > 15:
        return 1, df['CBullD_Lower_Low_gen'].iloc[-2]
    elif df['CBullD_neg_MACD'].iloc[-2] == 1 and 55 > df['CBullD_Higher_Low_RSI_neg_MACD'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_neg_MACD'].iloc[-2] > 15:
        return 1, df['CBullD_Lower_Low_neg_MACD'].iloc[-2]
    elif df['CBullD_gen'].iloc[-2] == 1 and 55 > df['CBullD_Higher_Low_RSI_gen'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_gen'].iloc[-2] > 15:
        return 1, df['CBullD_Lower_Low_gen'].iloc[-2]
    elif df['CBullD_x2'].iloc[-2] == 1:
        return 1, df['CBullD_x2_Lower_Low'].iloc[-2]
    else:
        return 0, 0

def Stoploss_Trigger_Check(df, Stoploss):
    return 1 if df['close'].iloc[-1] < Stoploss else 0


# ---------- Prefetch-Iterator mit kleinem Fenster ----------
def prefetch_tail_iterator(folder, files, tail, columns, workers, window, timings):
    """
    Liest Dateien mit einem begrenzten Prefetch-Fenster vor (Overlapping I/O),
    liefert (index, DataFrame) in Eingangsreihenfolge.
    """
    start = time.perf_counter()
    n = len(files)
    if n == 0:
        return
    # Worker begrenzen: nicht größer als Fenster
    max_workers = max(1, min(workers, window))

    # Producer-Funktion
    def submit(ex: ThreadPoolExecutor, idx: int) -> Future:
        fpath = os.path.join(folder, files[idx])
        return ex.submit(read_last_n_rows_parquet, fpath, tail, columns)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        next_to_submit = 0
        next_to_yield = 0

        # initial füllen
        while next_to_submit < n and len(futures) < window:
            futures[next_to_submit] = submit(ex, next_to_submit)
            next_to_submit += 1

        while next_to_yield < n:
            fut = futures.pop(next_to_yield)
            df, dt = fut.result()
            timings.add("Parquet lesen (gesamt)", dt)
            yield next_to_yield, df
            next_to_yield += 1

            # neues in die Pipeline nachschieben
            if next_to_submit < n:
                futures[next_to_submit] = submit(ex, next_to_submit)
                next_to_submit += 1

    timings.add("Prefetch-Orchestrierung", time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description="HBullD Backtest (Tail-I/O, Prefetch-Pipeline)")
    parser.add_argument("--folder", default=r"c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet",
                        help="Ordner mit Parquet-Dateien")
    parser.add_argument("--out", default=r"c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet\\backtest_results_ETH_4hour_100perc_with_brokerage.csv",
                        help="CSV-Ausgabedatei")
    parser.add_argument("--tail", type=int, default=100, help="Anzahl letzter Zeilen je Datei")
    parser.add_argument("--workers", type=int, default=0,
                        help="max. Threads fürs Lesen (0=auto)")
    parser.add_argument("--prefetch-window", type=int, default=4,
                        help="Größe des Prefetch-Fensters (empf. 2–8)")
    parser.add_argument("--summary-only", action="store_true", default=True,
                        help="Nur aggregierte Laufzeit-Zusammenfassung ausgeben")
    parser.add_argument("--print-io", action="store_true",
                        help="Frühere I/O-Prints (Start/Enddaten, Kapital) zusätzlich ausgeben")
    args = parser.parse_args()

    # Brokerage-Parameter (unverändert)
    Risk_percentage = 1.00
    Brokerage = 0.1
    Brokerage_buy = 1 + (Brokerage/100)
    Brokerage_sell = 1 - (Brokerage/100)

    timings = Timings()
    t_total_start = time.perf_counter()

    # Dateiliste
    t0 = time.perf_counter()
    parquet_files = sorted([f for f in os.listdir(args.folder) if f.endswith('.parquet')])
    timings.add("Dateiliste erstellen/sortieren", time.perf_counter() - t0)
    n = len(parquet_files)

    # Arrays
    t_init = time.perf_counter()
    Buy_Signal = [0]*n
    Buy_Signal_date = [0]*n
    Actual_Buy = [0]*n
    First_buy_date = [0]*n
    Stoploss = [0]*n
    Stoploss_Trigger = [0]*n
    Stoploss_Trigger_date = [0]*n
    Actual_Sell = [0]*n
    Current_Capital_Value = [0]*n
    Available_Capital_for_trade = [0]*n
    Buy_Quantity = [0]*n
    Total_Buy_Quantity = [0]*n
    loss_per_unit = [0]*n
    serial_date = [0]*n
    LM_Low_window_1_CS_last = [0]*n
    LM_Low_window_1_CS_last_date = [0]*n
    timings.add("Arrays initialisieren", time.perf_counter() - t_init)

    # Workers bestimmen
    if args.workers and args.workers > 0:
        max_workers = args.workers
    else:
        cpu = os.cpu_count() or 4
        # kleine Obergrenze, um Thrash zu vermeiden
        max_workers = min(8, max(2, cpu))

    window = max(1, args.prefetch_window)

    # Prefetch-Iterator liefert (i, df) in Ordnung
    trade_exec_total = 0.0
    signal_total = 0.0
    param_total = 0.0

    for i, df in prefetch_tail_iterator(
        args.folder, parquet_files, args.tail, NEEDED_COLS, max_workers, window, timings
    ):
        serial_date[i] = df['date'].iloc[-1]

        if i == 0:
            Current_Capital_Value[i] = 10000
            Available_Capital_for_trade[i] = Current_Capital_Value[i]

        # Trade-Execution
        t_trade = time.perf_counter()
        if i > 0:
            if Buy_Signal[i-1] == 1 and Stoploss_Trigger[i-1] != 1:
                loss_per_unit[i] = ((Brokerage_buy*df['close'].iloc[-2]) - (Brokerage_sell*Stoploss[i-1]))
                Buy_Quantity[i] = (Risk_percentage * Available_Capital_for_trade[i - 1]) / loss_per_unit[i] if loss_per_unit[i] > 0 else 0
                if (Buy_Quantity[i] * df['open'].iloc[-1] * Brokerage_buy) > Available_Capital_for_trade[i-1]:
                    Buy_Quantity[i] = Available_Capital_for_trade[i-1] / (df['open'].iloc[-1] * Brokerage_buy)
                if Buy_Quantity[i] < 0:
                    Buy_Quantity[i] = 0
                Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] - (Buy_Quantity[i] * df['open'].iloc[-1] * Brokerage_buy)
                Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1] + Buy_Quantity[i]
                if Buy_Quantity[i] > 0:
                    Actual_Buy[i] = 1
                    if Total_Buy_Quantity[i-1] == 0:
                        First_buy_date[i] = df['date'].iloc[-1]

            if Stoploss_Trigger[i-1] == 1 and Buy_Signal[i-1] != 1 and Total_Buy_Quantity[i-1] > 0:
                Sold_Quantity = Total_Buy_Quantity[i-1]
                Available_Capital_for_trade[i] = Available_Capital_for_trade[i-1] + (Sold_Quantity * df['open'].iloc[-1] * Brokerage_sell)
                Stoploss[i] = 0
                Actual_Sell[i] = 1

            if Actual_Buy[i] != 1 and Actual_Sell[i] != 1:
                Available_Capital_for_trade[i] = Available_Capital_for_trade[i - 1]
                Total_Buy_Quantity[i] = Total_Buy_Quantity[i-1]
        trade_exec_total += (time.perf_counter() - t_trade)

        # Signal-Monitoring
        t_sig = time.perf_counter()
        Buy_Signal[i], Stoploss[i] = Buy_Signal_Check(df)
        if Buy_Signal[i] == 1:
            Buy_Signal_date[i] = df['date'].iloc[-1]
        else:
            Stoploss[i] = Stoploss[i - 1]

        Stoploss_Trigger[i] = Stoploss_Trigger_Check(df, Stoploss[i])
        if Stoploss_Trigger[i] == 1 and Total_Buy_Quantity[i] > 0:
            Stoploss_Trigger_date[i] = df['date'].iloc[-1]
        signal_total += (time.perf_counter() - t_sig)

        # Parameter-Update / Stoploss-Logik
        t_param = time.perf_counter()
        if First_buy_date[i] != 0:
            First_buy_date[i] = First_buy_date[i]
        elif Actual_Sell[i] == 1:
            First_buy_date[i] = 0
        else:
            First_buy_date[i] = First_buy_date[i-1]

        LM_Low, LM_Date = get_last_nonzero_lm_low_fast(df['LM_Low_window_1_CS'], df['date'])
        LM_Low_window_1_CS_last[i], LM_Low_window_1_CS_last_date[i] = LM_Low, LM_Date

        if Buy_Signal[i] != 1:
            Stoploss[i] = Stoploss[i-1]
        if Buy_Signal[i] == 1 and Actual_Sell[i] != 1 and Total_Buy_Quantity[i-1] > 0:
            Stoploss[i] = min(Stoploss[i], Stoploss[i - 1])
        if Buy_Signal[i] != 1 and Actual_Sell[i] != 1 and Total_Buy_Quantity[i-1] > 0:
            if LM_Date > First_buy_date[i] and LM_Low > Stoploss[i - 1]:
                Stoploss[i] = LM_Low

        Current_Capital_Value[i] = Available_Capital_for_trade[i] + (Total_Buy_Quantity[i] * df['close'].iloc[-1])
        param_total += (time.perf_counter() - t_param)

    # Abschnittszeiten
    timings.add("Trade-Execution (gesamt)", trade_exec_total)
    timings.add("Signal-Monitoring (gesamt)", signal_total)
    timings.add("Parameter-Update/Stoploss-Logik (gesamt)", param_total)

    # DataFrame + CSV (unverändert)
    t_df = time.perf_counter()
    out_df = pd.DataFrame({
        'date': serial_date,
        'Buy_Signal': Buy_Signal,
        'Buy_Signal_Date': Buy_Signal_date,
        'Stoploss_Trigger': Stoploss_Trigger,
        'Stoploss_Trigger_Date': Stoploss_Trigger_date,
        'Actual_Buy': Actual_Buy,
        'Buy_Quantity': Buy_Quantity,
        'Total_Buy_Quantity': Total_Buy_Quantity,
        'Actual_Sell': Actual_Sell,
        'Available_Capital_for_trade': Available_Capital_for_trade,
        'Current_Capital_Value': Current_Capital_Value,
        'Stoploss': Stoploss,
        'LM_Low_window_1_CS_last': LM_Low_window_1_CS_last,
        'LM_Low_window_1_CS_last_date': LM_Low_window_1_CS_last_date,
        'First_buy_date': First_buy_date
    })
    timings.add("DataFrame zusammenbauen", time.perf_counter() - t_df)

    t_csv = time.perf_counter()
    out_df.to_csv(args.out, index=False)
    timings.add("CSV schreiben", time.perf_counter() - t_csv)

    # Nur Zusammenfassung
    total_seconds = time.perf_counter() - t_total_start
    timings.print_aggregated_only(total_seconds)


if __name__ == "__main__":
    main()
