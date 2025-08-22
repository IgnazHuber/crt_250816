import polars as pl
import numpy as np
from fastapi import FastAPI, UploadFile, File
import os
from datetime import datetime

app = FastAPI()

# Konfiguriere Polars für maximale Performance
pl.Config.set_streaming_chunk_size(1_000_000)  # Größere Chunks für Streaming
# Thread-Konfiguration wird von Polars automatisch gehandhabt
# Optional: Setze POLARS_MAX_THREADS=8

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = str(np.random.randint(1e9))
    file_path = f"data/{session_id}_{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    df = pl.scan_parquet(file_path).select(["date", "open", "high", "low", "close"])
    return {"success": True, "session_id": session_id, "info": {"file_loaded": True}}

@app.post("/api/analyze")
async def analyze(data: dict):
    session_id = data["session_id"]
    variants = data["variants"]
    file_path = f"data/{session_id}_*.parquet"
    
    # LazyFrame für Analyse, frühzeitig begrenzen
    six_months_ago = datetime.now().timestamp() - 180 * 24 * 3600
    df = pl.scan_parquet(file_path).select(["date", "open", "high", "low", "close"]).filter(
        pl.col("date").cast(pl.Int64) / 1000 >= six_months_ago
    ).tail(10000).cache()
    
    # EMA-Berechnung
    def calculate_ema(df, window):
        return df.with_columns(
            pl.col("close").ewm_mean(span=window, adjust=False).alias(f"ema{window}")
        )
    
    # EMA für alle Fenster
    for variant in variants:
        df = calculate_ema(df, variant["window"])
    
    # RSI und MACD
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
    )
    
    # Optimierte Divergenz-Detektion
    def detect_divergences(df, window, candle_tol, macd_tol):
        df = df.with_columns(
            price_low=pl.col("close").rolling_min(window * 2 + 1, center=True),
            rsi_low=pl.col("rsi").rolling_min(window * 2 + 1, center=True),
            macd_low=pl.col("macd").rolling_min(window * 2 + 1, center=True)
        ).with_columns(
            price_idx=pl.col("close").eq(pl.col("price_low")),
            rsi_idx=pl.col("rsi").eq(pl.col("rsi_low")),
            macd_idx=pl.col("macd").eq(pl.col("macd_low")),
            price_diff=pl.col("close").sub(pl.col("price_low")).truediv(pl.col("price_low")),
            macd_diff=pl.col("macd").sub(pl.col("macd_low")).abs()
        )
        
        classic_df = df.filter(
            (pl.col("price_idx") & ~pl.col("rsi_idx") & 
             (pl.col("price_diff") < candle_tol) & 
             (pl.col("macd_diff") < macd_tol))
        ).select([
            pl.col("date").cast(pl.Int64).alias("date"),
            pl.lit("bullish").alias("type"),
            (pl.col("rsi") - pl.col("rsi_low")).abs().truediv(10).clip(upper_bound=1.0).alias("strength"),
            pl.col("close").alias("low"),
            pl.col("rsi"),
            pl.col("macd"),
            pl.lit(window).alias("window")
        ])
        
        hidden_df = df.filter(
            (~pl.col("price_idx") & pl.col("rsi_idx") & 
             (pl.col("price_diff") < candle_tol) & 
             (pl.col("macd_diff") < macd_tol))
        ).select([
            pl.col("date").cast(pl.Int64).alias("date"),
            pl.lit("bearish").alias("type"),
            (pl.col("rsi") - pl.col("rsi_low")).abs().truediv(10).clip(upper_bound=1.0).alias("strength"),
            pl.col("close").alias("low"),
            pl.col("rsi"),
            pl.col("macd"),
            pl.lit(window).alias("window")
        ])
        
        return classic_df.collect(streaming=True), hidden_df.collect(streaming=True)
    
    results = {}
    for variant in variants:
        window = variant["window"]
        candle_tol = variant["candleTol"] / 100
        macd_tol = variant["macdTol"]
        
        classic_df, hidden_df = detect_divergences(df, window, candle_tol, macd_tol)
        classic_divs = [
            {"div_id": i + 1, "date": int(row["date"]), "type": row["type"], 
             "strength": row["strength"], "low": row["low"], "rsi": row["rsi"], 
             "macd": row["macd"], "window": row["window"]}
            for i, row in enumerate(classic_df.iter_rows(named=True))
        ]
        hidden_divs = [
            {"div_id": i + 1, "date": int(row["date"]), "type": row["type"], 
             "strength": row["strength"], "low": row["low"], "rsi": row["rsi"], 
             "macd": row["macd"], "window": row["window"]}
            for i, row in enumerate(hidden_df.iter_rows(named=True))
        ]
        
        results[variant["id"]] = {
            "classic": classic_divs,
            "hidden": hidden_divs,
            "total": len(classic_divs) + len(hidden_divs)
        }
    
    # Collect für Chart-Daten
    df_filtered = df.select([
        "date", "open", "high", "low", "close", "rsi", "macd"
    ] + [f"ema{v['window']}" for v in variants if f"ema{v['window']}" in df.columns]
    ).collect(streaming=True)
    
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