import pandas as pd
import os
from datetime import datetime
import numpy as np
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hinweis: finplot & Initialize_RSI_EMA_MACD bleiben wie gehabt vorhanden,
# der Plot-Teil ist weiter auskommentiert, also keine Features entfernt.
import finplot as fplt  # noqa: F401
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD  # noqa: F401

# ---------- Präzise Laufzeitmessung ----------
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

def read_last_n_rows_parquet(path, n=100, columns=None):
    """
    Liest die letzten n Zeilen einer Parquet-Datei effizient:
    - Nur ausgewählte Spalten (columns)
    - Rowgroups rückwärts, dann tail auf n
    Identisch zu pd.read_parquet(...).tail(n), aber ohne Voll-Load.
    """
    import pyarrow.parquet as pq
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
    return df

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

def main():
    parser = argparse.ArgumentParser(description="HBullD Backtest, schnell & deterministisch")
    parser.add_argument("--folder", default=r"c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10",
                        help="Ordner mit Parquet-Dateien")
    parser.add_argument("--out", default=r"c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10\\backtest_results_ETH_4hour_100perc_with_brokerage_10.csv",
                        help="CSV-Ausgabedatei")
    parser.add_argument("--tail", type=int, default=100, help="Anzahl letzter Zeilen je Datei")
    parser.add_argument("--workers", type=int, default=0,
                        help="Anzahl parallel lesender Threads (0=auto)")
    parser.add_argument("--summary-only", action="store_true", default=True,
                        help="Nur aggregierte Laufzeit-Zusammenfassung ausgeben")
    parser.add_argument("--print-io", action="store_true",
                        help="Frühere I/O-Prints (Start/Enddaten, Kapital) zusätzlich ausgeben")
    args = parser.parse_args()

    # Parameter (Brokerage etc.) unverändert
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

    # ---------- PARALLELES LADEN ----------
    # Auto-Workers: I/O-bound → mehr Threads sinnvoll
    if args.workers and args.workers > 0:
        max_workers = args.workers
    else:
        # Da Arrow den GIL freigibt, sind 2*CPU-Kerne meist ok; cap 32
        cpu = os.cpu_count() or 4
        max_workers = min(32, max(4, 2*cpu))

    def _load(i_file):
        fname = parquet_files[i_file]
        fpath = os.path.join(args.folder, fname)
        t_read = time.perf_counter()
        df = read_last_n_rows_parquet(fpath, n=args.tail, columns=NEEDED_COLS)
        # aggregiertes Lesen als Gesamtzeit erfassen
        elapsed = time.perf_counter() - t_read
        return i_file, df, elapsed

    t_read_all = time.perf_counter()
    dfs = [None]*n
    if n > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_load, i): i for i in range(n)}
            for fut in as_completed(futures):
                i, df_i, dt = fut.result()
                dfs[i] = df_i
                timings.add("Parquet lesen (gesamt)", dt)
    timings.add("Paralleles Laden orchestrieren", time.perf_counter() - t_read_all)

    # ---------- HAUPTSCHLEIFE (deterministisch in Dateireihenfolge) ----------
    t_trade_exec_total = 0.0
    t_signal_total = 0.0
    t_param_total = 0.0

    for i in range(n):
        df = dfs[i]
        if df is None:
            raise RuntimeError(f"Fehlender DataFrame für Index {i}")

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
        t_trade_exec_total += (time.perf_counter() - t_trade)

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
        t_signal_total += (time.perf_counter() - t_sig)

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
        t_param_total += (time.perf_counter() - t_param)

    # Abschnittszeiten verbuchen
    timings.add("Trade-Execution (gesamt)", t_trade_exec_total)
    timings.add("Signal-Monitoring (gesamt)", t_signal_total)
    timings.add("Parameter-Update/Stoploss-Logik (gesamt)", t_param_total)

    # DataFrame + CSV
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

    # Optional: alte Konsolenprints nur bei --print-io
    if args.print_io:
        print(out_df['date'].iloc[0])
        print(out_df['Current_Capital_Value'].iloc[-1])
        print(out_df['date'].iloc[-1])

    # Gesamtlaufzeit & nur aggregierte Übersicht
    total_seconds = time.perf_counter() - t_total_start
    timings.print_aggregated_only(total_seconds)

if __name__ == "__main__":
    main()
