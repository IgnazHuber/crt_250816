import pandas as pd
import os
import finplot as fplt
import glob
from datetime import datetime
import numpy as np
import time
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD

# ---------- Präzise Laufzeitmessung ----------
class Timings:
    def __init__(self):
        self.sections = {}
        self.per_file = {}

    def add(self, key, seconds):
        self.sections[key] = self.sections.get(key, 0.0) + seconds

    def add_file(self, fname, key, seconds):
        d = self.per_file.setdefault(fname, {})
        d[key] = d.get(key, 0.0) + seconds

    @staticmethod
    def _fmt(s): return f"{s:,.3f}s"

    def print_summary(self):
        print("\n================ Laufzeit-Übersicht (aggregiert) ================")
        total = 0.0
        for k, v in sorted(self.sections.items(), key=lambda x: -x[1]):
            print(f"{k:<30s}: {self._fmt(v)}")
            total += v
        print(f"{'-'*30}: {'-'*10}")
        print(f"{'Summe gemessener Abschnitte':<30s}: {self._fmt(total)}")
        print("=================================================================\n")

    def print_per_file(self, limit=10):
        if not self.per_file:
            return
        print("================ Laufzeit je Datei (Top/erste N) ================")
        count = 0
        for fname, secs in self.per_file.items():
            print(f"[{fname}]")
            subtotal = 0.0
            for k, v in sorted(secs.items(), key=lambda x: -x[1]):
                print(f"  {k:<26s}: {self._fmt(v)}")
                subtotal += v
            print(f"  {'-'*26}: {'-'*10}")
            print(f"  {'Subtotal':<26s}: {self._fmt(subtotal)}\n")
            count += 1
            if count >= limit:
                break
        if len(self.per_file) > limit:
            print(f"... ({len(self.per_file)-limit} weitere Dateien nicht angezeigt)")
        print("=================================================================\n")

timings = Timings()

# ---------- Pandas Anzeige ----------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

Risk_percentage = 1.00  # %
Brokerage = 0.1         # in %
Brokerage_buy = 1 + (Brokerage/100)
Brokerage_sell = 1 - (Brokerage/100)

# Pfade (unverändert)
folder_path = r'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10'
output_combined_file = r'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10\\backtest_results_ETH_4hour_100perc_with_brokerage_10.csv'
input_file = r'c:\\Projekte\\Anirudh\\ETH\\output_4hour_parquet_10\\eth_4hour_candlesticks_all.csv'

# ---------- Nur benötigte Spalten (Projektion) ----------
NEEDED_COLS = [
    'date', 'open', 'close', 'EMA_50', 'EMA_200',
    'HBullD_gen', 'HBullD_Lower_Low_RSI_gen', 'HBullD_Higher_Low_RSI_gen', 'HBullD_Higher_Low_gen',
    'HBullD_neg_MACD', 'HBullD_Lower_Low_RSI_neg_MACD', 'HBullD_Higher_Low_RSI_neg_MACD', 'HBullD_Higher_Low_neg_MACD',
    'CBullD_gen', 'CBullD_Lower_Low_RSI_gen', 'CBullD_Higher_Low_RSI_gen', 'CBullD_Lower_Low_gen',
    'CBullD_neg_MACD', 'CBullD_Lower_Low_RSI_neg_MACD', 'CBullD_Higher_Low_RSI_neg_MACD', 'CBullD_Lower_Low_neg_MACD',
    'CBullD_x2', 'CBullD_x2_Lower_Low',
    'LM_Low_window_1_CS'
]

# ---------- Effizientes Tail-Lesen letzter N Zeilen aus Parquet ----------
def read_last_n_rows_parquet(path, n=100, columns=None):
    """
    Liest die letzten n Zeilen einer Parquet-Datei effizient:
    - Nur ausgewählte Spalten (columns)
    - Nur letzte Rowgroups (falls n in letzte RG passt), sonst mehrere von hinten
    """
    import pyarrow.parquet as pq
    t0 = time.perf_counter()
    pf = pq.ParquetFile(path, memory_map=True)
    md = pf.metadata
    num_row_groups = md.num_row_groups

    rows_needed = n
    row_groups_to_read = []
    # von hinten nach vorne, bis wir >= n Zeilen haben
    for rg in range(num_row_groups-1, -1, -1):
        rg_rows = md.row_group(rg).num_rows
        row_groups_to_read.append(rg)
        rows_needed -= rg_rows
        if rows_needed <= 0:
            break
    row_groups_to_read.reverse()

    # lesen
    t_read = time.perf_counter()
    tables = [pf.read_row_group(rg, columns=columns) for rg in row_groups_to_read]
    tbl = tables[0].concat(tables[1:]) if len(tables) > 1 else tables[0]
    # final tail
    if tbl.num_rows > n:
        tbl = tbl.slice(tbl.num_rows - n, n)
    df = tbl.to_pandas(types_mapper=None, use_threads=True, split_blocks=True)
    timings.add("Parquet lesen (gesamt)", time.perf_counter() - t0)
    return df

# ---------- Fachlogik (unverändert) ----------
def get_last_nonzero_lm_low_fast(series, dates):
    """Schneller: letzte Position != 0 via NumPy, dann Wert + Datum zurückgeben."""
    arr = series.to_numpy()
    nz = np.flatnonzero(arr != 0)
    if nz.size == 0:
        return 0, 0
    idx = nz[-1]
    return arr[idx], dates.iloc[idx]

def Buy_Signal_Check(df):
    if df['HBullD_gen'].iloc[-2] == 1 and 70 > df['HBullD_Lower_Low_RSI_gen'].iloc[-2] > 40 and 70 > df['HBullD_Higher_Low_RSI_gen'].iloc[-2] > 40 and df['EMA_50'].iloc[-2] > df['EMA_200'].iloc[-2]:
        Buy_Signal = 1; Stoploss = df['HBullD_Higher_Low_gen'].iloc[-2]
    elif df['HBullD_neg_MACD'].iloc[-2] == 1 and 70 > df['HBullD_Lower_Low_RSI_neg_MACD'].iloc[-2] > 40 and 70 > df['HBullD_Higher_Low_RSI_neg_MACD'].iloc[-2] > 40 and df['EMA_50'].iloc[-2] > df['EMA_200'].iloc[-2]:
        Buy_Signal = 1; Stoploss = df['HBullD_Higher_Low_neg_MACD'].iloc[-2]
    elif ((df['CBullD_gen'].iloc[-2] == 1 and df['CBullD_neg_MACD'].iloc[-2] == 1) or df['CBullD_gen'].iloc[-2] == 1) and 55 > df['CBullD_Higher_Low_RSI_gen'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_gen'].iloc[-2] > 15:
        Buy_Signal = 1; Stoploss = df['CBullD_Lower_Low_gen'].iloc[-2]
    elif df['CBullD_neg_MACD'].iloc[-2] == 1 and 55 > df['CBullD_Higher_Low_RSI_neg_MACD'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_neg_MACD'].iloc[-2] > 15:
        Buy_Signal = 1; Stoploss = df['CBullD_Lower_Low_neg_MACD'].iloc[-2]
    elif df['CBullD_gen'].iloc[-2] == 1 and 55 > df['CBullD_Higher_Low_RSI_gen'].iloc[-2] > 30 and 55 > df['CBullD_Lower_Low_RSI_gen'].iloc[-2] > 15:
        Buy_Signal = 1; Stoploss = df['CBullD_Lower_Low_gen'].iloc[-2]
    elif df['CBullD_x2'].iloc[-2] == 1:
        Buy_Signal = 1; Stoploss = df['CBullD_x2_Lower_Low'].iloc[-2]
    else:
        Buy_Signal = 0; Stoploss = 0
    return Buy_Signal, Stoploss

def Stoploss_Trigger_Check(df, Stoploss):
    return 1 if df['close'].iloc[-1] < Stoploss else 0

# ================================== MAIN ==================================
t_total_start = time.perf_counter()

# Dateiliste
t0 = time.perf_counter()
parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])
timings.add("Dateiliste erstellen/sortieren", time.perf_counter() - t0)

# Arrays
t_init = time.perf_counter()
n = len(parquet_files)
Buy_Signal = [0]*n
Buy_Signal_date = [0]*n
Actual_Buy = [0]*n
First_buy_date = [0]*n
Stoploss = [0]*n
Stoploss_Trigger = [0]*n
Stoploss_Trigger_date = [0]*n
Last_LM_window_1_CS = [0]*n
Last_LM_window_1_CS_date = [0]*n
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

# Hauptschleife
for i, file_name in enumerate(parquet_files):
    file_path = os.path.join(folder_path, file_name)

    # Nur letzte 100 Zeilen + benötigte Spalten laden
    t_read = time.perf_counter()
    df = read_last_n_rows_parquet(file_path, n=100, columns=NEEDED_COLS)
    timings.add_file(file_name, "Parquet lesen", time.perf_counter() - t_read)

    serial_date[i] = df['date'].iloc[-1]

    # Init Kapital
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
    timings.add("Trade-Execution (gesamt)", time.perf_counter() - t_trade)
    timings.add_file(file_name, "Trade-Execution", time.perf_counter() - t_trade)

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
    timings.add("Signal-Monitoring (gesamt)", time.perf_counter() - t_sig)
    timings.add_file(file_name, "Signal-Monitoring", time.perf_counter() - t_sig)

    # Parameter-Update / Stoploss-Logik
    t_param = time.perf_counter()
    if First_buy_date[i] != 0:
        First_buy_date[i] = First_buy_date[i]
    elif Actual_Sell[i] == 1:
        First_buy_date[i] = 0
    else:
        First_buy_date[i] = First_buy_date[i-1]

    # Schnellste Variante für "letztes != 0"
    LM_Low_window_1_CS_last[i], LM_Low_window_1_CS_last_date[i] = get_last_nonzero_lm_low_fast(
        df['LM_Low_window_1_CS'], df['date']
    )

    if Buy_Signal[i] != 1:
        Stoploss[i] = Stoploss[i-1]
    if Buy_Signal[i] == 1 and Actual_Sell[i] != 1 and Total_Buy_Quantity[i-1] > 0:
        Stoploss[i] = min(Stoploss[i], Stoploss[i - 1])
    if Buy_Signal[i] != 1 and Actual_Sell[i] != 1 and Total_Buy_Quantity[i-1] > 0:
        if LM_Low_window_1_CS_last_date[i] > First_buy_date[i] and LM_Low_window_1_CS_last[i] > Stoploss[i - 1]:
            Stoploss[i] = LM_Low_window_1_CS_last[i]

    Current_Capital_Value[i] = Available_Capital_for_trade[i] + (Total_Buy_Quantity[i] * df['close'].iloc[-1])
    timings.add("Parameter-Update/Stoploss-Logik (gesamt)", time.perf_counter() - t_param)
    timings.add_file(file_name, "Parameter-Update/Stoploss-Logik", time.perf_counter() - t_param)

# Combine into a DataFrame (unverändert)
t_df = time.perf_counter()
df = pd.DataFrame({
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

# CSV schreiben (unverändert)
t_csv = time.perf_counter()
df.to_csv(output_combined_file, index=False)
timings.add("CSV schreiben", time.perf_counter() - t_csv)

print(serial_date[0])
print(Current_Capital_Value[len(df)-1])
print(serial_date[len(df)-1])

# (Plot-Teil belassen, aber auskommentiert, um keine Features zu entfernen)

# Gesamtlaufzeit
t_total_end = time.perf_counter()
print(f"\nGesamtlaufzeit: {t_total_end - t_total_start:,.3f}s")
timings.print_summary()
timings.print_per_file(limit=10)
