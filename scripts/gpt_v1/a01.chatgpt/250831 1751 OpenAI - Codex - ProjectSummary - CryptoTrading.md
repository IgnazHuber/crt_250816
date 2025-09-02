# Project Summary — CryptoTrading (OpenAI Codex)

Date/Time: 2025-08-31 17:51

This repository implements an end-to-end research workflow for divergence-based signal discovery, DOE (Design of Experiments), signal validation via backtesting, and automated parameter optimization, with reporting to interactive HTML and Excel workbooks. The main entry point provides an interactive menu to run analyses, DOE, backtests, and a secondary sweep optimizer. Recent work adds scaffolds for robustness diagnostics (WFCV, PBO/DS), execution realism, risk sizing, regime hints, and a portfolio layer.

## Core Modules

- `Mainframe_RTv250829_mplfinance.py`: Interactive driver. Menu options a–i, asset and time-span pickers, plotting, DOE, backtesting, and secondary sweep integration.
- `Backtest_Divergences.py`: Vectorized backtest core and Excel export helpers.
- Indicator analyses (vectorized):
  - `CBullDivg_Analysis_vectorized.py`
  - `CBullDivg_x2_analysis_vectorized.py`
  - `HBearDivg_analysis_vectorized.py`
  - `HBullDivg_analysis_vectorized.py`
- Signal prep: `Initialize_RSI_EMA_MACD.py`, `Local_Maximas_Minimas.py`

## Menu (Mainframe)

- a: Classic Bullish (CBullDivg)
- b: Extended Bullish (CBullDivg_x2)
- c: Hidden Bearish (HBearDivg)
- d: Hidden Bullish (HBullDivg)
- e: All analyses (a–d)
- f: DOE (Design of Experiments) over `(candle_percent, macd_percent)`
- g: Backtest markers (validate signals for current span)
- h: Backtest DOE markers (combine any `*doe_markers_*.csv` in `results/`)
- i: Secondary sweep optimizer (Stage‑B trading params)

Notes:
- Asset picker via Windows file dialog (fallback to console). Timespan picker via Windows Forms calendar; intraday-aware. Filenames and plot titles are prefixed with an asset tag (e.g., `btc_1day`).

## Today’s Additions (Scaffolds)

- `validation_cv.py`: Purged Walk‑Forward CV with embargo; per‑fold KPIs and robust score μ − λ·σ; XLSX + HTML with equity snapshot.
- `execution_simulator.py`: Execution realism knobs — latency (bars), simple limit/stop “touch” entry filter; respects fee/slippage; XLSX + HTML.
- `stats_overfit.py`: PBO/Deflated‑Sharpe surrogates across purged folds; IS/OOS KPIs, Spearman(IS,OOS), OOS μ/σ; XLSX + HTML.
- `risk_position.py`: Kelly fraction and “10-loss streak” ruin proxy derived from backtest trades; XLSX + equity preview HTML.
- `regime_robustness.py`: Volatility proxy (30‑bar) and placeholder for regime conditioning; XLSX preview + HTML.
- `portfolio_layer.py`: Multi-asset correlation matrix and simple discrete-weight portfolio sampling; XLSX + HTML with heatmap and best equity curve.

These modules are imported in the main driver with graceful fallbacks (they don’t break existing flows if unavailable). Dedicated menu wiring can be added later if desired.

## DOE Workflow (Option f)

- Inputs: `doe_parameters_example.csv` lists `(candle_percent, macd_percent)` grid.
- Parallelization: `ProcessPoolExecutor` with deterministic aggregation; logs timing.
- Counts: By‑type aggregation (Classic/Hidden/Total) per grid cell; saved markers per cell into `results/`.
- Plots:
  - Default: 2×2 facet heatmaps (CBullDivg, CBullDivg_x2, HBullDivg, HBearDivg) with cell text “C:…/H:…”.
  - Optional: Total heatmap and per‑type HTML pages.
  - Overlays (optional): Robust Top‑N and Pareto points from WFCV on marker counts; optional robust backtest KPI CV overlays.
- DOE Summary XLSX: `DOE_Summary`, `Info` (asset/span/config), `All_Markers`, per‑type pivots/heatmaps, and “Delta‑to‑baseline” matrices; optional DOE backtest CV table.

## Backtesting (Options g/h)

- Entries at next bar open after marker; conservative ordering for stop vs TP.
- Risk‑based sizing on realized equity; optional conservative sizing using actual entry fill distance.
- Fee and slippage applied per side on notional; equity updated on exits.
- Constraints: max concurrency, per-side caps, time stop, single-position mode.
- Exports: Trades (with fills), EquityCurve chart, KPIs; XLSX workbook.

## Secondary Sweep Optimizer (Option i)

- Stage‑B sweep across trading params for shortlisted indicator cells.
- Grid sources: CSV > ENV > built-ins. Default CSV: `secondary_sweep_ranges.csv` (list schema; per-frequency overrides supported). Built‑in defaults exist for candle%, macd%, risk%, stop%, tp%.
- Walk‑forward CV per tuple with robust scoring and constraints (MinTrades, MaxDD, MinPF). Coarse‑to‑fine refinement (±20%).
- Single‑asset and multi‑asset modes; combined report with Best‑By‑Asset and Top‑Assets pages.
- Console summaries: top‑3 tuples (single) or assets (multi) with Score/PnL/PF/DD/Trades/FinalEq.

## Plotting Enhancements

- Variant grouping and legend: V1/V2/V3 → Type → Direction; Classic+Hidden combined group toggles; proxy traces ensure clean legend entries.
- Variant comparison: Outline markers show Additional/Missing vs V1; quick per-variant visibility buttons (Only/Hide V1/V2/V3).
- Extra panels: Optional Volume, ATR%, OBV, VWAP with adjustable heights; RSI(30/70) and MACD(0) guides.
- Performance: Switch to ScatterGL over `PLOT_GL_THRESHOLD` (default 20k points).

## Configuration (selected ENV)

- DOE: `DOE_MAX_WORKERS`, `DOE_WF_SPLITS`, `DOE_ROBUST_LAMBDA`, `DOE_TOPN_OVERLAY`, `DOE_BT_WF_SPLITS`, `DOE_BT_ROBUST_KPI`
- Backtest: `BT_FEE_PCT`, `BT_SLIPPAGE_PCT`, `BT_MAX_POS_VAL_PCT`, `BT_EQUITY_CAP`, `BT_TIME_STOP_BARS`, `BT_SINGLE_POSITION`
- Secondary Sweep: `SS_RANGES_CSV`, `SS_GRID_*`, `SS_WF_SPLITS`, `SS_LAMBDA`, `SS_MIN_TRADES`, `SS_MAX_DD`, `SS_MIN_PF`, `SS_SCORE_METHOD`, `SS_MAX_WORKERS`, `SS_TOPN`, `SS_CTF_ENABLE`, `SS_CTF_TOPN`, `SS_ASSETS`, `SS_ASSETS_GLOB`, `SS_SPAN_START`, `SS_SPAN_END`
- Execution: `EXEC_LATENCY_BARS`, `EXEC_ORDER_MODEL` (market/limit/stop), `EXEC_TICK_PCT`, `EXEC_SLIPPAGE_PCT`
- PBO/DS/WFCV: `WFCV_SPLITS`, `WFCV_EMBARGO_PCT`, `PBO_SPLITS`, `PBO_EMBARGO_PCT`

## Inputs and Outputs

- Inputs: CSV/Parquet market files with columns including `date, open, high, low, close, volume` (+ computed indicators from `Initialize_RSI_EMA_MACD.py`).
- Outputs (prefixed by asset tag):
  - HTML charts: analysis plot, DOE total/facet/per‑type.
  - CSV/XLSX: markers, DOE summary, backtest reports, secondary sweep reports, combined sweep, and diagnostic scaffolds.
  - Directory: `results/` (created automatically).

## Quick Start

1) Launch `Mainframe_RTv250829_mplfinance.py` and pick an asset file.
2) Choose a menu option:
   - f: DOE over a grid from `doe_parameters_example.csv`
   - g: Backtest current markers (ensure you ran a–e first)
   - h: Backtest any DOE marker CSVs found in `results/`
   - i: Secondary sweep; optionally select multiple assets
3) Review `results/` for HTML and XLSX outputs; console prints top summaries.

## Safety & Correctness Assumptions

- Sizing uses realized equity at entry; conservative sizing uses actual entry fill for stop distance.
- Fees/slippage applied on both sides; fills incorporate slippage in stops/TP.
- CV folds skip empty-signal segments; still produce Info‑only workbooks if no trades.

## Roadmap

- Wire new scaffolds (WFCV, PBO/DS, execution, risk, regime, portfolio) into the main menu.
- Adaptive defaults for very short spans (auto reduce folds/thresholds).
- Robust DOE directly on backtest KPIs across indicators.
- Portfolio‑level optimizer with correlation and risk constraints.
- Live forward monitoring and alerting on drift.

## File Pointers

- Main driver: `Mainframe_RTv250829_mplfinance.py`
- Summary (prior): `PROJECT_SUMMARY.md`
- DOE grid example: `doe_parameters_example.csv`
- Secondary sweep ranges: `secondary_sweep_ranges.csv`

