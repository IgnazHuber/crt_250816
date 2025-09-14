import itertools
from typing import Iterable, List, Optional
import os
import json
try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # Fallback if needed
    _toml = None


def x_y_pairs(step: int = 10, low: int = 10, high: int = 90):
    """Generate ordered pairs (x, y) with low <= x < y <= high at a given step size."""
    values = list(range(low, high + 1, step))
    for x in values:
        for y in values:
            if y > x:
                yield x, y


def build_specs(
    limit: Optional[int] = None,
    max_specs: Optional[int] = None,
    families: Optional[List[str]] = None,
    ema_variants: Optional[List[str]] = None,
    step: int = 10,
    low: int = 10,
    high: int = 90,
):
    """
    Build a manageable Design of Experiments (DOE) grid.

    - EMA variants:
      v1: EMA20 > EMA50
      v2: EMA20 > EMA50 and EMA50 > EMA200
      v3: EMA20 > EMA50 > EMA200 (strict chain)

    - Divergence families:
      hb_any: HBullD_gen == 1 or HBullD_neg_MACD == 1
      cb_gen: CBullD_gen == 1
      cb_neg: CBullD_neg_MACD == 1

    - Thresholds (RSI ranges, gap and slope): pairs (x,y) with low <= x < y <= high (default 10..90)

    Parameters
    - limit: per (ema_variant, family) cap on pair combinations (inner product); defaults to 60 if None
    - max_specs: global cap across all generated specs (applied after building per-combo)
    - families: optional list to restrict divergence families
    - ema_variants: optional list to restrict EMA variants
    - step/low/high: control the pair grid granularity and bounds

    Returns a list of spec dicts.
    """
    ema_variants = list(ema_variants) if ema_variants else ['v1', 'v2', 'v3']
    families = list(families) if families else ['hb_any', 'cb_gen', 'cb_neg']
    pairs = list(x_y_pairs(step=step, low=low, high=high))

    # To keep first run small, restrict to a subset unless limit is None
    if limit is None:
        limit = 60  # reasonable per-combo cap

    specs = []
    stop_at = max_specs if isinstance(max_specs, int) and max_specs > 0 else None
    for ema_v in ema_variants:
        for fam in families:
            # Inner product of (LL range) x (HL range), limited per combo
            for (x1, y1), (x2, y2) in itertools.islice(itertools.product(pairs, pairs), 0, limit):
                spec = {
                    'ema_variant': ema_v,
                    'div_family': fam,
                    # RSI ranges for Lower_Low and Higher_Low
                    'rsi_lower_low_range': [x1, y1],
                    'rsi_higher_low_range': [x2, y2],
                    # Optional date gap and slope ranges; use same as higher-low by default
                    'date_gap_range': [x2, y2],
                    'slope_range': [x2, y2],
                }
                specs.append(spec)
                if stop_at is not None and len(specs) >= stop_at:
                    return specs
    return specs


def _read_config(path: str) -> dict:
    with open(path, 'rb') as f:
        if path.lower().endswith(('.toml', '.doe')) and _toml is not None:
            return _toml.load(f)
        # Fallback: attempt JSON if TOML not available
        data = f.read()
    try:
        return json.loads(data.decode('utf-8'))
    except Exception as e:
        raise ValueError(f"Unsupported DOE config format for {path}. Install Python 3.11+ for TOML or provide JSON.")


def build_specs_from_config(config_path: str) -> list[dict]:
    """
    Build specs from a TOML/JSON config file.

    Supported schema (TOML example):

    [grid]
    ema_variants = ["v1", "v2"]
    families = ["hb_any", "cb_gen"]
    step = 20
    low = 10
    high = 90
    limit = 3        # per (ema,family)
    max_specs = 25   # global cap

    # Or explicit specs list:
    [[specs]]
    ema_variant = "v1"
    div_family = "hb_any"
    rsi_lower_low_range = [10, 30]
    rsi_higher_low_range = [30, 50]
    date_gap_range = [30, 50]
    slope_range = [30, 50]
    """
    cfg = _read_config(config_path)

    # If explicit specs provided, use them as-is (with optional max_specs)
    specs_list = cfg.get('specs')
    grid = cfg.get('grid')
    out: list[dict] = []

    def _cap(out_list: list[dict]):
        mx = None
        if isinstance(grid, dict) and isinstance(grid.get('max_specs'), int):
            mx = grid['max_specs']
        if isinstance(cfg.get('max_specs'), int):
            mx = cfg['max_specs']
        return out_list[:mx] if mx and mx > 0 else out_list

    if isinstance(specs_list, list) and specs_list:
        for s in specs_list:
            spec = {
                'ema_variant': s['ema_variant'],
                'div_family': s['div_family'],
                'rsi_lower_low_range': list(s['rsi_lower_low_range']),
                'rsi_higher_low_range': list(s['rsi_higher_low_range']),
                'date_gap_range': list(s.get('date_gap_range', s['rsi_higher_low_range'])),
                'slope_range': list(s.get('slope_range', s['rsi_higher_low_range'])),
            }
            out.append(spec)
        return _cap(out)

    if isinstance(grid, dict):
        ema_variants = grid.get('ema_variants') or ['v1', 'v2', 'v3']
        families = grid.get('families') or ['hb_any', 'cb_gen', 'cb_neg']
        step = int(grid.get('step', 10))
        low = int(grid.get('low', 10))
        high = int(grid.get('high', 90))
        limit = int(grid.get('limit', 60))
        built = build_specs(limit=limit, max_specs=grid.get('max_specs'), families=families, ema_variants=ema_variants, step=step, low=low, high=high)
        return _cap(built)

    # If nothing matched, error
    raise ValueError(f"DOE config {config_path} missing 'grid' or 'specs' section")
