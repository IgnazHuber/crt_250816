import pandas as pd
from pathlib import Path

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
from Trendline_Up_Support_vectorized import calc_TL_Up_Support
from Trendline_Up_Resistance_vectorized import calc_TL_Up_Resistance
from Trendline_Down_Resistance_vectorized import calc_TL_Down_Resistance

csv_path = Path('test_data.csv')
if not csv_path.exists():
    raise SystemExit('test_data.csv missing; please generate it first')

df_full = pd.read_csv(csv_path, parse_dates=['date'])


def pipeline_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    Initialize_RSI_EMA_MACD(df)
    Level_1_Max_Min(df)
    Candlestick_Type(df)
    CBullDivg_analysis(df, 0.1, 3.25)
    CBullDivg_x2_analysis(df, 0.1, 3.25)
    HBullDivg_analysis(df, 0.1, 3.25)
    CBearDivg_analysis(df, 0.1, 3.25)
    HBearDivg_analysis(df, 0.1, 3.25)
    df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
    df = calc_TL_Up_Support(df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
    df = calc_TL_Up_Resistance(df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
    df = calc_TL_Down_Resistance(df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
    df = calculate_golden_ratios(df)
    return df


def pipeline_hourly(df: pd.DataFrame, sens: float) -> pd.DataFrame:
    df = df.copy()
    Initialize_RSI_EMA_MACD(df)
    Level_1_Max_Min(df)
    Candlestick_Type(df)
    CBullDivg_analysis(df, sens, 3.25)
    CBullDivg_x2_analysis(df, sens, 3.25)
    HBullDivg_analysis(df, sens, 3.25)
    CBearDivg_analysis(df, sens, 3.25)
    HBearDivg_analysis(df, sens, 3.25)
    df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
    df = calculate_golden_ratios(df)
    return df


from pandas.testing import assert_frame_equal

def compare(label: str, tail: int | None, pipeline_func, *pipeline_args):
    base_processed = pipeline_func(df_full, *pipeline_args)
    indices = [200, 210, 280]
    print(f'Checking {label} ...')
    for i in indices:
        if i >= len(df_full):
            continue
        slice_df = df_full.iloc[:i]
        old = pipeline_func(slice_df, *pipeline_args)
        if tail:
            old_slice = old.tail(tail).reset_index(drop=True)
            new_slice = base_processed.iloc[:i].tail(tail).reset_index(drop=True)
        else:
            old_slice = old.reset_index(drop=True)
            new_slice = base_processed.iloc[:i].reset_index(drop=True)
        try:
            assert_frame_equal(old_slice, new_slice, check_dtype=False, rtol=1e-6, atol=1e-8)
        except AssertionError as exc:
            print(f'  [FAIL] mismatch at i={i}: {exc}')
            return False
    print('  [OK] matches for tested slices')
    return True

all_ok = True
all_ok &= compare('daily', None, pipeline_daily)
all_ok &= compare('1h', 400, pipeline_hourly, 0.01)
all_ok &= compare('4h', 400, pipeline_hourly, 0.05)

if not all_ok:
    raise SystemExit(1)
print('All comparisons passed.')
