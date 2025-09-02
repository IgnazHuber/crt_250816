# Crypto Trading Divergence Analysis System
## Project Description - Version 250831

### Overview
This project is a comprehensive cryptocurrency trading analysis system focused on **divergence detection and backtesting**. The main framework (`Mainframe_RTv250829_mplfinance.py`) serves as an orchestrator for technical analysis, pattern recognition, backtesting, and statistical validation of trading strategies based on bullish and bearish divergences in price and technical indicators.

### Core Architecture

#### Main Framework (`Mainframe_RTv250829_mplfinance.py`)
The central orchestrator (~37,000 tokens) that coordinates all system components:

**Key Features:**
- Interactive Windows Forms-based date picker for backtest period selection
- Multi-variant analysis comparison (V1/V2 parameter sets)
- Parallel processing support using ThreadPoolExecutor and ProcessPoolExecutor
- Rich HTML report generation with Plotly visualizations
- Excel export functionality with advanced formatting
- Design of Experiments (DOE) support for parameter optimization

**Core Functions:**
- `select_backtest_timespan()`: Interactive timespan selection with PowerShell calendar
- `plot_with_plotly()`: Advanced visualization with legend grouping and marker plotting
- Main execution flow with comprehensive error handling and logging

### Technical Analysis Modules

#### Divergence Detection Engines
1. **`CBullDivg_Analysis_vectorized.py`**: Classic Bullish Divergence detection
   - Analyzes price lows vs RSI/MACD patterns
   - Vectorized NumPy implementation for performance
   - Configurable tolerance parameters for pattern matching

2. **`CBullDivg_x2_analysis_vectorized.py`**: Extended Classic Bullish Divergence analysis
   - Enhanced pattern detection algorithms
   - Additional confirmation signals

3. **`HBearDivg_analysis_vectorized.py`**: Hidden Bearish Divergence detection
   - Identifies continuation patterns in bearish trends
   - Price highs vs indicator analysis

4. **`HBullDivg_analysis_vectorized.py`**: Hidden Bullish Divergence detection
   - Continuation patterns in bullish trends
   - Complementary to classic divergence analysis

#### Technical Indicators (`Initialize_RSI_EMA_MACD.py`)
High-precision calculation engine for:
- **RSI (Relative Strength Index)**: 14-period default with vectorized computation
- **EMA (Exponential Moving Averages)**: Multiple timeframes (12, 20, 26, 50, 100, 200)
- **MACD (Moving Average Convergence Divergence)**: With 9-period signal line
- **Precision Control**: Uses Decimal class for high-precision calculations to minimize accumulation errors

#### Local Extrema Detection (`Local_Maximas_Minimas.py`)
- Identifies significant price highs and lows
- Configurable window parameters for peak/valley detection
- Essential for divergence pattern recognition

### Backtesting & Risk Management

#### Core Backtesting Engine (`Backtest_Divergences.py`)
**BacktestParams Configuration:**
- Starting capital and risk percentage per trade
- Trailing stops and take profit levels
- Transaction fees and slippage modeling
- Position sizing constraints
- Conservative sizing options
- Maximum open positions control

**Features:**
- Long and short position support
- Multiple concurrent positions
- Time-based exit rules
- Equity curve generation
- Comprehensive trade logging

#### Parameter Optimization (`Secondary_Sweep_Optimizer.py`)
- Multi-parameter sweep optimization
- Parallel execution of parameter combinations
- Statistical analysis of optimization results
- Overfitting detection mechanisms

### Advanced Analytics Modules

#### Cross-Validation Framework (`validation_cv.py`)
**Purged Walk-Forward Cross-Validation:**
- Time-series aware validation methodology
- Embargo periods to prevent data leakage
- Fold-based performance aggregation
- Robust scoring with mean-variance penalties
- HTML and Excel reporting

#### Statistical Validation (`stats_overfit.py`)
**Probability of Backtest Overfitting (PBO) & Deflated Sharpe Analysis:**
- In-sample vs Out-of-sample performance comparison
- Statistical significance testing
- Overfitting probability calculations
- Deflated Sharpe ratio computations

#### Execution Simulation (`execution_simulator.py`)
- Real-world execution modeling
- Latency and slippage simulation
- Market impact analysis
- Order fill probability modeling

#### Risk Management (`risk_position.py`)
- Portfolio-level risk assessment
- Position sizing algorithms
- Drawdown control mechanisms
- Risk-adjusted performance metrics

#### Regime Analysis (`regime_robustness.py`)
- Market regime identification
- Strategy performance across different market conditions
- Robustness testing framework

#### Data Quality Assurance (`data_integrity.py`)
- Data completeness validation
- Outlier detection and handling
- Missing data imputation strategies
- Data quality reporting

#### Portfolio Management (`portfolio_layer.py`)
- Multi-asset portfolio allocation
- Correlation analysis
- Diversification optimization
- Portfolio-level backtesting

### Technical Infrastructure

#### Dependencies & Libraries
**Core Scientific Computing:**
- `numpy`: Vectorized numerical computations
- `pandas`: Time series data manipulation
- `plotly`: Interactive visualizations
- `openpyxl`: Excel file generation and formatting

**Concurrency & Performance:**
- `concurrent.futures`: Parallel processing
- Vectorized operations throughout all modules
- Memory-efficient data structures

**Optional Integrations:**
- All advanced analytics modules use graceful imports with fallbacks
- Modular architecture allows partial functionality if dependencies missing

#### Data Flow Architecture
1. **Input**: OHLC price data with timestamp
2. **Technical Analysis**: RSI, EMA, MACD calculation
3. **Pattern Detection**: Local extrema identification
4. **Divergence Analysis**: Multiple divergence type detection
5. **Signal Generation**: Trading signals based on patterns
6. **Backtesting**: Historical performance simulation
7. **Validation**: Statistical robustness testing
8. **Reporting**: Comprehensive HTML/Excel outputs

### Output & Reporting

#### Visualization Features
- Multi-subplot Plotly charts (Price/RSI/MACD)
- Interactive markers for divergence signals
- Equity curves with trade overlays
- Parameter heatmaps for optimization results
- Legend grouping by signal type (Classic/Hidden)

#### Export Capabilities
- **Excel Reports**: Multi-sheet workbooks with formatting, charts, and conditional coloring
- **HTML Dashboards**: Interactive reports with embedded charts
- **CSV Data**: Raw results for further analysis
- **PNG/Base64**: Chart exports for documentation

### Project Structure & Versioning
```
scripts/claude_v4/
├── Mainframe_RTv250829_mplfinance.py  # Main orchestrator
├── Technical Analysis/
│   ├── CBullDivg_Analysis_vectorized.py
│   ├── CBullDivg_x2_analysis_vectorized.py
│   ├── HBearDivg_analysis_vectorized.py
│   ├── HBullDivg_analysis_vectorized.py
│   ├── Initialize_RSI_EMA_MACD.py
│   └── Local_Maximas_Minimas.py
├── Backtesting/
│   ├── Backtest_Divergences.py
│   └── Secondary_Sweep_Optimizer.py
├── Advanced Analytics/
│   ├── validation_cv.py
│   ├── stats_overfit.py
│   ├── execution_simulator.py
│   ├── risk_position.py
│   ├── regime_robustness.py
│   ├── data_integrity.py
│   └── portfolio_layer.py
├── Configuration/
│   ├── settings.json
│   ├── doe_parameters_example.csv
│   └── secondary_sweep_ranges.csv
└── Documentation/
    ├── 250831 CryptoTrading - Parameters.docx
    └── a01-a44.chatgpt/ (Development history)
```

### Development Methodology
- **Iterative Development**: 44+ development iterations (a01-a44.chatgpt)
- **Multi-LLM Collaboration**: ChatGPT, Gemini, Grok contributions
- **Modular Design**: Independent modules with clean interfaces
- **Error Resilience**: Comprehensive exception handling throughout
- **Performance Optimization**: Vectorized operations and parallel processing

### Use Cases & Applications
1. **Cryptocurrency Trading**: Primary focus on crypto market analysis
2. **Strategy Development**: Divergence-based trading strategy creation
3. **Risk Assessment**: Portfolio risk management and position sizing
4. **Academic Research**: Statistical validation of trading hypotheses
5. **Parameter Optimization**: Systematic strategy parameter tuning
6. **Performance Analysis**: Comprehensive backtesting and validation

### German Language Support
The system includes comprehensive German documentation and user interface elements, reflecting its likely European development origin. Error messages, parameter descriptions, and report headers include German translations.

---

**Generated**: August 31, 2025  
**Analysis Tool**: Claude Code v4  
**Project Version**: RTv250829  
**Total Codebase Size**: ~37,000+ lines of Python code