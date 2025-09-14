import numpy as np


def _ema_filter(ema20, ema50, ema200, variant: str):
    if variant == 'v1':
        return ema20 > ema50
    if variant == 'v2':
        return (ema20 > ema50) & (ema50 > ema200)
    if variant == 'v3':
        return (ema20 > ema50) & (ema50 > ema200)  # strict chain as numeric inequalities already imply order
    return np.ones_like(ema20, dtype=bool)


def _in_range(val, low_high):
    x, y = low_high
    return (val > x) & (val < y)


def compute_signals(context: dict, spec: dict | None):
    """
    Returns (buy_signals_pre, initial_stoploss_pre, divg_slope_pre)
    - All arrays are 1D numpy arrays aligned with context arrays
    """
    # Extract arrays
    closes = context['closes']
    ema20 = context['ema20_prev']
    ema50 = context['ema50_prev']
    ema200 = context['ema200_prev']

    # HBull
    hb_gen = context['hb_gen']
    hb_neg = context['hb_neg_macd']
    hb_ll_rsi_gen = context['hb_ll_rsi_gen']
    hb_hl_rsi_gen = context['hb_hl_rsi_gen']
    hb_ll_rsi_neg = context['hb_ll_rsi_neg']
    hb_hl_rsi_neg = context['hb_hl_rsi_neg']
    hb_ll = context['hb_ll_gen']
    hb_hl = context['hb_hl_gen']
    hb_ll_n = context['hb_ll_neg']
    hb_hl_n = context['hb_hl_neg']
    hb_gap_g = context['hb_date_gap_gen']
    hb_gap_n = context['hb_date_gap_neg']

    # CBull
    cb_gen = context['cb_gen']
    cb_neg = context['cb_neg_macd']
    cb_ll_rsi_gen = context['cb_ll_rsi_gen']
    cb_hl_rsi_gen = context['cb_hl_rsi_gen']
    cb_ll_rsi_neg = context['cb_ll_rsi_neg']
    cb_hl_rsi_neg = context['cb_hl_rsi_neg']
    cb_ll = context['cb_ll_gen']
    cb_hl = context['cb_hl_gen']
    cb_ll_n = context['cb_ll_neg']
    cb_hl_n = context['cb_hl_neg']
    cb_gap_g = context['cb_date_gap_gen']
    cb_gap_n = context['cb_date_gap_neg']

    # EMA filter
    ema_cond = _ema_filter(ema20, ema50, ema200, (spec or {}).get('ema_variant', 'v1'))

    div_family = (spec or {}).get('div_family', 'hb_any')
    r_ll_rng = (spec or {}).get('rsi_lower_low_range', [15, 55])
    r_hl_rng = (spec or {}).get('rsi_higher_low_range', [30, 70])
    gap_rng = (spec or {}).get('date_gap_range', [10, 90])
    slope_rng = (spec or {}).get('slope_range', [0, 999999])

    buy = np.zeros_like(closes, dtype=int)
    isl = np.zeros_like(closes, dtype=float)

    # Compute divergence slope candidates (safe divisions)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = 10000 * np.divide(np.abs(hb_ll - hb_hl), (hb_ll * hb_gap_g), out=np.zeros_like(hb_ll), where=(hb_ll * hb_gap_g != 0))
        d2 = 10000 * np.divide(np.abs(hb_ll_n - hb_hl_n), (hb_ll_n * hb_gap_n), out=np.zeros_like(hb_ll_n), where=(hb_ll_n * hb_gap_n != 0))
        d3 = 10000 * np.divide(np.abs(cb_ll - cb_hl), (cb_hl * cb_gap_g), out=np.zeros_like(cb_hl), where=(cb_hl * cb_gap_g != 0))
        d4 = 10000 * np.divide(np.abs(cb_ll_n - cb_hl_n), (cb_hl_n * cb_gap_n), out=np.zeros_like(cb_hl_n), where=(cb_hl_n * cb_gap_n != 0))

    if div_family == 'hb_any':
        # Either hb_gen or hb_neg
        cond_gen = (hb_gen == 1) & _in_range(hb_ll_rsi_gen, r_ll_rng) & _in_range(hb_hl_rsi_gen, r_hl_rng) & _in_range(hb_gap_g, gap_rng) & _in_range(d1, slope_rng) & (closes > hb_hl)
        cond_neg = (hb_neg == 1) & _in_range(hb_ll_rsi_neg, r_ll_rng) & _in_range(hb_hl_rsi_neg, r_hl_rng) & _in_range(hb_gap_n, gap_rng) & _in_range(d2, slope_rng) & (closes > hb_hl_n)
        cond = (cond_gen | cond_neg) & ema_cond
        isl = np.where(cond_gen, hb_hl, np.where(cond_neg, hb_hl_n, 0))
        ds = np.where(cond_gen, d1, np.where(cond_neg, d2, 0))
        buy = np.where(cond, 1, 0)
        ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
        return buy, isl, ds

    if div_family == 'cb_gen':
        cond = (cb_gen == 1) & _in_range(cb_ll_rsi_gen, r_ll_rng) & _in_range(cb_hl_rsi_gen, r_hl_rng) & _in_range(cb_gap_g, gap_rng) & _in_range(d3, slope_rng) & (closes > cb_ll) & ema_cond
        buy = np.where(cond, 1, 0)
        isl = np.where(cond, cb_ll, 0)
        ds = np.where(cond, d3, 0)
        ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
        return buy, isl, ds

    if div_family == 'cb_neg':
        cond = (cb_neg == 1) & _in_range(cb_ll_rsi_neg, r_ll_rng) & _in_range(cb_hl_rsi_neg, r_hl_rng) & _in_range(cb_gap_n, gap_rng) & _in_range(d4, slope_rng) & (closes > cb_ll_n) & ema_cond
        buy = np.where(cond, 1, 0)
        isl = np.where(cond, cb_ll_n, 0)
        ds = np.where(cond, d4, 0)
        ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
        return buy, isl, ds

    # Fallback: no rules
    return np.zeros_like(closes, dtype=int), np.zeros_like(closes, dtype=float), np.zeros_like(closes, dtype=float)

