# Project Summary: Divergence Analysis, DOE, Backtesting, and Optimizer

This project provides an end‑to‑end pipeline to detect divergence markers, explore parameter spaces (DOE), validate signals via backtesting, and automatically optimize trading parameters across assets and timeframes. Results are exported to rich Excel workbooks and interactive HTML plots, with asset/frequency tagging in filenames and within reports.

## High‑Level Components

- Main driver: `Mainframe_RTv250829_mplfinance.py`
  - Options a–i for analyses, DOE, backtesting, and optimization
  - Asset picker, timespan picker (date/time aware), asset/frequency tagging
  - Interactive Plotly HTML charts and Excel reports
- Backtester: `Backtest_Divergences.py`
  - Risk‑based sizing, conservative sizing (entry fill), trailing stop, take‑profit
  - Fees & slippage applied on both sides (per‑side %)
  - Concurrency caps, single‑position mode, time stop
  - Equity curve, drawdown, KPIs, theoretical max loss per trade
- DOE visuals & summaries (inside main)
  - ProcessPool parallelization, by‑type facet heatmaps, total heatmap
  - Robust overlays (counts and backtest CV): top‑N by score and Pareto front
  - DOE summary XLSX with Info sheet (asset, span, robust settings)
- Secondary sweep optimizer: `Secondary_Sweep_Optimizer.py` (option i)
  - Stage‑B sweep of trading params for shortlisted indicator cells
  - CSV/ENV/defaults precedence for parameter grids; per‑frequency overrides
  - Walk‑forward CV with robust scoring; coarse‑to‑fine refinement
  - Multi‑asset mode + combined workbook with dashboards

## Menu Options (Mainframe)

- a: Classic Bullish (CBullDivg)
- b: Extended Bullish (CBullDivg_x2)
- c: Hidden Bearish (HBearDivg)
- d: Hidden Bullish (HBullDivg)
- e: All analyses (a–d)
- f: DOE (Design of Experiments)
- g: Backtest markers (validate signals)
- h: Backtest DOE markers (combine any `*doe_markers_*.csv` in `results/`)
- i: Secondary sweep optimizer (auto‑tune Stage‑B)

Each run opens an asset picker (default falls back to BTC daily), then a timespan picker. Option g/h/i echo and enforce the selected span.

## Key Enhancements

### Asset & Timespan Handling
- Asset label and tag extracted from filename (e.g., `btc_1day_candlesticks_all.csv` → tag: `btc_1day`).
- Titles in HTML charts and filenames are prefixed with the asset tag.
- Timespan selection via date/time dialog; span embedded in filenames and stored in reports.

### DOE
- ProcessPool parallelization (faster CPU‑bound exploration) with timing logs.
- By‑type facet heatmaps (2×2) plus total heatmap; consistent scales.
- Robust overlays:
  - Marker counts CV (walk‑forward): score = mean − λ·std
  - Backtest CV per DOE cell: robust KPI overlays (e.g., Total_PnL), top‑N and Pareto
- DOE XLSX
  - DOE_Summary, Info (asset, span, robust configs), All_Markers
  - Per‑analysis heatmaps (optionally per‑type pages), deltas vs baseline
  - Overview: Top‑N parameter pairs, stacked charts, matrices

### Backtesting
- Entry: next bar open post marker; conservative ordering (stop before TP).
- Sizing: risk% of realized equity at entry; conservative option uses entry_fill for stop distance.
- Costs: slippage (adverse) and per‑side fee% on notional.
- Controls: max concurrent positions, per‑direction caps, time stop, single‑position mode.
- Trades sheet: Entry/Exit fills, price audit, Risk_Cash, Equity_At_Entry, Theoretical_Max_Loss_$.
- EquityCurve: dual‑axis chart (Price/Equity) with explicit axes and legend.

### Secondary Sweep Optimizer (Option i)
- Stage‑B sweep of trading params over shortlisted indicator cells:
  - Walk‑forward CV (folds), robust scoring, constraints (MinTrades, MaxDD, MinPF)
  - CSV/ENV/defaults parameter grids with per‑frequency overrides
  - Coarse‑to‑fine refinement (±20% around top tuples, clamped)
  - Parallel execution with progress indicator
- Reports (per asset):
  - Overview (top‑N), All_Results, Pareto (PnL vs MaxDD chart), Info (spans/configs)
  - Adds Mean_Final_Equity to KPIs
- Combined report (multi‑asset):
  - Best_By_Asset (best robust per asset; also written as CSV)
  - Top_Assets (bar chart of top N assets by Score)
  - Best_By_Frequency (best robust per frequency + chart)
  - Reports (paths to per‑asset sweep files)
- Console summary (end of run):
  - Single asset: top‑3 parameter sets (Score, PnL, PF, DD%, Trades, FinalEq)
  - Combined: top‑3 assets by Score (Score, PnL, PF, DD%, FinalEq)

## Configuration Precedence

CSV > ENV > built‑in defaults.
- CSV: `secondary_sweep_ranges.csv` (see formats below)
- ENV: `SS_GRID_*` and others (reference at bottom)
- Defaults (built‑in):
  - candle%: 0.01, 0.02, 0.05, 0.10, 0.20, 0.50
  - macd%: 1, 2, 3, 4, 5
  - risk%: 3, 5, 8, 12, 18, 25
  - stop%: 2, 3, 5, 7, 10
  - tp%: 3, 5, 7, 10, 15, 20, 30, 50

## CSV Formats for Secondary Sweep

Use one schema per CSV.

List form (current default)

param,values,frequency
candle_percent,"0.01,0.02,0.05,0.10,0.20,0.50",all
macd_percent,"1,2,3,4,5",all
risk_pct,"3,5,8,12,18,25",all
stop_pct,"2,3,5,7,10",all
tp_pct,"3,5,7,10,15,20,30,50",all

Add per‑frequency rows (override “all” for matches):

param,values,frequency
candle_percent,"0.005,0.01,0.02,0.05,0.10,0.20",1h
macd_percent,"0.5,1,2,3",1h
risk_pct,"1,2,3,5,8",1h
stop_pct,"1,2,3,5,7",1h
tp_pct,"2,3,5,7,10",1h
candle_percent,"0.01,0.02,0.05,0.10,0.20,0.50",1day
macd_percent,"1,2,3,4,5",1day
risk_pct,"3,5,8,12,18,25",1day
stop_pct,"2,3,5,7,10",1day
tp_pct,"3,5,7,10,15,20,30,50",1day
candle_percent,"0.02,0.05,0.10,0.20,0.50,1.00",1week
macd_percent,"1,2,3,4,6,8,10",1week
risk_pct,"3,5,8,12,15",1week
stop_pct,"3,5,7,10,15",1week
tp_pct,"5,10,15,20,30,50",1week

Range form (alternative)

param,min,max,step,frequency
candle_percent,0.005,0.20,auto,1h
macd_percent,0.5,3.0,0.5,1h
risk_pct,1,8,auto,1h
stop_pct,1,7,auto,1h
tp_pct,2,10,auto,1h
candle_percent,0.01,0.50,auto,1day
macd_percent,1,5,1,1day
risk_pct,3,25,auto,1day
stop_pct,2,10,auto,1day
tp_pct,3,50,auto,1day
candle_percent,0.02,1.00,auto,1week
macd_percent,1,10,auto,1week
risk_pct,3,15,auto,1week
stop_pct,3,15,auto,1week
tp_pct,5,50,auto,1week

Notes:
- `auto` generates ~6 points; candle_percent uses log spacing, others linear.
- Frequency is matched from the asset tag (e.g., `eth_4h` → `4h`).

## Outputs & Naming

- Filenames: prefixed with `Asset_Tag` (e.g., `btc_1day_…`).
- HTML: `…_plot.html`, `…_doe_heatmap_total.html`, `…_doe_heatmaps_by_type.html`, per‑type heatmaps.
- DOE XLSX: `{tag}_doe_summary_… .xlsx` with Info (asset/tag, Start/End, robust configs).
- Backtest XLSX: `{tag}_backtest[_doe_combined]_s…-e…_… .xlsx`.
- Secondary sweep (per asset): `{tag}_secondary_sweep_… .xlsx`.
- Combined sweep: `combined_secondary_sweep_… .xlsx` (+ `_best_by_asset.csv`).

## Performance

- DOE: ProcessPoolExecutor with sorted aggregation, timing logs, and consistent outputs.
- Backtest: vectorized core + per‑trade simulation; equity updates on exits.
- Secondary sweep: ThreadPoolExecutor across parameter tuples; progress indicators for coarse and refine stages; optional SS_MAX_WORKERS.

## Safety & Correctness

- Risk sizing uses realized equity at entry; open PnL not counted.
- Conservative sizing option uses entry_fill for stop distance (safer sizing).
- Fees/slippage applied on notional entry/exit; stop/TP fills include slippage.
- If a fold has no signals, it’s skipped; if no folds yield signals, Info‑only workbook is created.

## Quick Start Examples

Single asset (option i):
1) Run option i; pick timespan; answer No to multi‑asset.
2) Optionally adjust envs:
   - `SS_WF_SPLITS=3` `SS_LAMBDA=1.0`
   - `SS_GRID_RISK="3,5,8,12,18,25"` (or use CSV)
3) Check `results/{tag}_secondary_sweep_… .xlsx` and console top‑3 summary.

Multi‑asset (option i):
- Env: `SS_ASSETS="C:\\data\\btc_1day.csv;C:\\data\\eth_1day.csv"`
- Or: `SS_ASSETS_GLOB="C:\\data\\daily\\*_candlesticks_*.csv"`
- Optional: `SS_SPAN_START=2019-01-01` `SS_SPAN_END=2022-12-31`
- Run option i; a combined workbook and CSV are generated, plus per‑asset workbooks.

## Environment Variables (Selected)

- Common
  - `DOE_MAX_WORKERS`, `DOE_WF_SPLITS`, `DOE_ROBUST_LAMBDA`, `DOE_TOPN_OVERLAY`, `DOE_BT_WF_SPLITS`, `DOE_BT_ROBUST_KPI`
- Backtest
  - `BT_FEE_PCT`, `BT_SLIPPAGE_PCT`, `BT_MAX_POS_VAL_PCT`, `BT_EQUITY_CAP`, `BT_TIME_STOP_BARS`, `BT_SINGLE_POSITION`
- Secondary Sweep
  - `SS_RANGES_CSV` (default `secondary_sweep_ranges.csv`)
  - `SS_GRID_CANDLE`, `SS_GRID_MACD`, `SS_GRID_RISK`, `SS_GRID_STOP`, `SS_GRID_TP`
  - `SS_WF_SPLITS`, `SS_LAMBDA`, `SS_MIN_TRADES`, `SS_MAX_DD`, `SS_MIN_PF`, `SS_SCORE_METHOD`
  - `SS_MAX_WORKERS`, `SS_TOPN`
  - `SS_CTF_ENABLE` (true), `SS_CTF_TOPN` (5)
  - `SS_ASSETS`, `SS_ASSETS_GLOB`, `SS_SPAN_START`, `SS_SPAN_END`

## Future Ideas

- Adaptive defaults for very short spans (auto reduce folds/thresholds).
- Robust DOE directly on backtest KPIs across indicators.
- Portfolio‑level optimizer and correlation management.
- Live forward‑monitoring and alerting on performance drift.

