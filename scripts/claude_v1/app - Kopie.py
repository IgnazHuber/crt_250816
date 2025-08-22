import polars as pl
import numpy as np
from fastapi import FastAPI, UploadFile, File
import os
import pyarrow as pa
import pyarrow.compute as pc
from datetime import datetime

app = FastAPI()

# Konfiguriere Polars für maximale Performance
pl.Config.set_streaming_chunk_size(1_000_000)  # Größere Chunks für Streaming
# Thread-Konfiguration wird von Polars automatisch gehandhabt
# Optional: Setze POLARS_MAX_THREADS als Umgebungsvariable, z. B. set POLARS_MAX_THREADS=8

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = str(np.random.randint(1e9))
    file_path = f"data/{session_id}_{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # Lazy-Scan der Datei mit frühem Filter  
    df = pl.scan_parquet(file_path).select(["date", "open", "high", "low", "close"])
    return {"success": True, "session_id": session_id, "info": {"file_loaded": True}}

@app.post("/api/analyze")
async def analyze(data: dict):
    session_id = data["session_id"]
    variants = data["variants"]
    file_path = f"data/{session_id}_*.parquet"
    
    # LazyFrame für Analyse
    df = pl.scan_parquet(file_path).select(["date", "open", "high", "low", "close"]).cache()
    
    # Frühzeitiger Zeitraum-Filter (letzte 6 Monate, anpassbar)
    six_months_ago = datetime.now().timestamp() - 180 * 24 * 3600
    df = df.filter(pl.col("date").cast(pl.Int64) / 1000 >= six_months_ago)
    
    # EMA-Berechnung (optimiert mit Arrow)
    def calculate_ema(df, window):
        return df.with_columns(
            pl.col("close").ewm_mean(span=window, adjust=False).alias(f"ema{window}")
        ).cache()
    
    # EMA für alle Fenster
    for variant in variants:
        df = calculate_ema(df, variant["window"])
    
    # RSI und MACD (optimiert mit Arrow)
    df = df.with_columns(
        delta=pl.col("close").diff(),
        ema12=pl.col("close").ewm_mean(span=12, adjust=False),
        ema26=pl.col("close").ewm_mean(span=26, adjust=False)
    ).with_columns(
        macd=pl.col("ema12") - pl.col("ema26"),
        gain=pl.col("delta").clip(lower_bound=0),
        loss=pl.col("delta").abs().clip(lower_bound=0, upper_bound=None).filter(pl.col("delta") < 0)
    ).with_columns(
        avg_gain=pl.col("gain").ewm_mean(span=14, adjust=False),
        avg_loss=pl.col("loss").ewm_mean(span=14, adjust=False)
    ).with_columns(
        rsi=pl.when(pl.col("avg_loss") != 0)
              .then(100 - (100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))))
              .otherwise(50)
    ).cache()

    # Optimierte Divergenz-Detektion mit Polars
    results = {}
    for variant in variants:
        window = variant["window"]
        candle_tol = variant["candleTol"] / 100
        macd_tol = variant["macdTol"]
        
        # Vectorized divergence detection using rolling operations
        df_analysis = df.with_columns([
            # Rolling min/max for price patterns
            pl.col("close").rolling_min(window_size=window*2+1, center=True).alias("price_min"),
            pl.col("rsi").rolling_min(window_size=window*2+1, center=True).alias("rsi_min"),
            pl.col("macd").rolling_min(window_size=window*2+1, center=True).alias("macd_min"),
            
            # Price tolerance check
            (pl.col("close") - pl.col("close").rolling_min(window_size=window*2+1, center=True)).abs() / pl.col("close") < candle_tol).alias("price_tol_ok"),
            
            # MACD tolerance check  
            (pl.col("macd") - pl.col("macd").rolling_min(window_size=window*2+1, center=True)).abs() < macd_tol).alias("macd_tol_ok")
        ]).with_columns([
            # Classic divergence: price at local min but RSI not at min
            (pl.col("close") == pl.col("price_min") & 
             pl.col("rsi") != pl.col("rsi_min") &
             pl.col("price_tol_ok") &
             pl.col("macd_tol_ok")).alias("classic_div"),
            
            # Hidden divergence: price not at min but RSI at min
            (pl.col("close") != pl.col("price_min") &
             pl.col("rsi") == pl.col("rsi_min") &
             pl.col("price_tol_ok") &
             pl.col("macd_tol_ok")).alias("hidden_div")
        ])
        
        # Collect only divergence points (much smaller dataset)
        classic_df = df_analysis.filter(pl.col("classic_div")).select([
            "date", "close", "rsi", "macd"
        ]).collect(streaming=True)
        
        hidden_df = df_analysis.filter(pl.col("hidden_div")).select([
            "date", "close", "rsi", "macd"  
        ]).collect(streaming=True)
        
        # Convert to result format
        classic_divs = []
        for row in classic_df.iter_rows(named=True):
            classic_divs.append({
                "div_id": len(classic_divs) + 1,
                "date": int(row["date"]),
                "type": "bullish",
                "strength": min(1.0, abs(row["rsi"] - row["rsi"]) / 10),
                "low": row["close"],
                "rsi": row["rsi"],
                "macd": row["macd"],
                "window": window
            })
            
        hidden_divs = []
        for row in hidden_df.iter_rows(named=True):
            hidden_divs.append({
                "div_id": len(hidden_divs) + 1,
                "date": int(row["date"]),
                "type": "bearish", 
                "strength": min(1.0, abs(row["rsi"] - row["rsi"]) / 10),
                "low": row["close"],
                "rsi": row["rsi"],
                "macd": row["macd"],
                "window": window
            })
        
        results[variant["id"]] = {
            "classic": classic_divs,
            "hidden": hidden_divs,
            "total": len(classic_divs) + len(hidden_divs)
        }
    
    # Sample data for chart (limit to recent data to avoid memory issues)
    df_filtered = df.select([
        "date", "open", "high", "low", "close", "rsi", "macd"
    ] + [f"ema{v['window']}" for v in variants if f"ema{v['window']}" in df.columns])
    
    # Only collect last 10000 points for charting
    df_filtered = df_filtered.tail(10000).collect(streaming=True)
    
    chart_data = {
        "dates": df_filtered["date"].to_list(),
        "open": df_filtered["open"].to_list(),
        "high": df_filtered["high"].to_list(),
        "low": df_filtered["low"].to_list(),
        "close": df_filtered["close"].to_list(),
        "rsi": df_filtered["rsi"].to_list(),
        "macd_histogram": df_filtered["macd"].to_list(),
        "ema20": df_filtered.get_column("ema20").to_list() if "ema20" in df_filtered.columns else None,
        "ema50": df_filtered.get_column("ema50").to_list() if "ema50" in df_filtered.columns else None,
        "ema100": df_filtered.get_column("ema100").to_list() if "ema100" in df_filtered.columns else None,
        "ema200": df_filtered.get_column("ema200").to_list() if "ema200" in df_filtered.columns else None
    }
    
    return {"success": True, "chartData": chart_data, "results": results}