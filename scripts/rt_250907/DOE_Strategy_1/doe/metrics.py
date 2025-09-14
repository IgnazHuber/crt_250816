import pandas as pd
import numpy as np


def compute_metrics_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    # Basic guard
    if df.empty:
        return {
            'trades': 0,
            'final_capital': np.nan,
            'total_return_pct': np.nan,
            'max_drawdown_pct': np.nan,
            'win_rate_pct': np.nan,
            'profit_factor': np.nan,
            'avg_trade_pnl': np.nan,
        }

    initial_capital = float(df['Current_Capital_Value'].iloc[0])
    final_capital = float(df['Current_Capital_Value'].iloc[-1])
    total_return_pct = (final_capital / initial_capital - 1.0) * 100.0 if initial_capital else np.nan

    # Max drawdown on equity curve
    eq = df['Current_Capital_Value'].values.astype(float)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_drawdown_pct = float(dd.min() * 100.0) if len(dd) else np.nan

    # Trades and outcomes
    sells = df['Actual_Sell'].values.astype(int)
    trade_count = int(sells.sum())
    profits = df.get('Trade_Profit', pd.Series([0]*len(df))).values.astype(float)
    losses = df.get('Trade_Loss', pd.Series([0]*len(df))).values.astype(float)
    realized = profits - losses
    realized_trades = realized[sells == 1]
    if trade_count > 0 and realized_trades.size > 0:
        wins = (realized_trades > 0).sum()
        win_rate_pct = wins / trade_count * 100.0
        gross_profit = realized_trades[realized_trades > 0].sum()
        gross_loss = -realized_trades[realized_trades < 0].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
        avg_trade_pnl = realized_trades.mean()
    else:
        win_rate_pct = np.nan
        profit_factor = np.nan
        avg_trade_pnl = np.nan

    return {
        'trades': trade_count,
        'final_capital': final_capital,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate_pct': win_rate_pct,
        'profit_factor': profit_factor,
        'avg_trade_pnl': avg_trade_pnl,
    }

