import pandas as pd
import warnings
import os
import numpy as np
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from Goldenratio_vectorized import calculate_golden_ratios
from Support_Resistance_vectorized import calculate_support_levels
import multiprocessing as mp
import uuid

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented*')

# Paths
csv_file_path = r'C:\Anirudh\Python\IBKR\Final_Version\ETH\eth_2hour_candlesticks_all.csv'
output_dir = r'C:\Anirudh\Python\IBKR\Incremental\ETH\output_2hour_parquet'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Preprocessing: Read the full CSV once and save to a temporary parquet in output_dir for faster loading in processes
df_full = pd.read_csv(csv_file_path, header=0, parse_dates=['date'])
temp_parquet_path = os.path.join(output_dir, 'temp_df.parquet')  # Changed to store in output_dir
df_full.to_parquet(temp_parquet_path, index=False, engine='pyarrow')
len_df = len(df_full)

def process_i(i):
    # Create unique temp parquet for this process
    temp_parquet_path = os.path.join(output_dir, f'temp_df_{uuid.uuid4().hex}.parquet')
    pd.read_csv(csv_file_path, header=0, parse_dates=['date']).head(i).to_parquet(temp_parquet_path, index=False, engine='pyarrow')

    # Load from this parquet
    df = pd.read_parquet(temp_parquet_path)

    # Initialize indicators
    Initialize_RSI_EMA_MACD(df)
    Level_1_Max_Min(df)
    Candlestick_Type(df)
    CBullDivg_analysis(df, 0.05, 3.25)
    CBullDivg_x2_analysis(df, 0.05, 3.25)
    HBullDivg_analysis(df, 0.05, 3.25)
    CBearDivg_analysis(df, 0.05, 3.25)
    HBearDivg_analysis(df, 0.05, 3.25)

    df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
    df = calculate_golden_ratios(df)

    # Save output
    last_date = df['date'].iloc[-1]
    last_date_sanitized = str(last_date).replace('/', '-').replace(':', '-').replace(' ', '_')
    output_file = os.path.join(output_dir, f'output_{last_date_sanitized}.parquet')
    df.tail(400).to_parquet(output_file, index=False, engine='pyarrow')

    # Clean up unique temp file
    os.remove(temp_parquet_path)

if __name__ == '__main__':
    # Parallelize the processing using multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
    # with mp.Pool(6) as pool:
        pool.map(process_i, range(200, len_df - 1))

    # Optional: Clean up temporary parquet file
    os.remove(temp_parquet_path)
