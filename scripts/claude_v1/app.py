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
    return {"success": True, "session_id": session_id, "info": {"rows": df.collect(streaming=True).shape[0]}}

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

    # Divergenz-Detektion (vereinfacht, optimiert)
    results = {}
    for variant in variants:
        window = variant["window"]
        candle_tol = variant["candleTol"] / 100
        macd_tol = variant["macdTol"]
        
        # Arrow-basierte Divergenz-Detektion
        df_arrow = df.collect(streaming=True).to_arrow()
        prices = df_arrow["close"].to_numpy()
        rsi = df_arrow["rsi"].to_numpy()
        macd = df_arrow["macd"].to_numpy()
        dates = df_arrow["date"].to_numpy()
        
        classic_divs = []
        hidden_divs = []
        for i in range(window, len(prices) - window):
            # Preis-Tiefs
            price_low = prices[i - window:i + window].min()
            price_idx = i - window + np.argmin(prices[i - window:i + window])
            
            # RSI-Tiefs
            rsi_low = rsi[i - window:i + window].min()
            rsi_idx = i - window + np.argmin(rsi[i - window:i + window])
            
            # MACD-Tiefs
            macd_low = macd[i - window:i + window].min()
            macd_idx = i - window + np.argmin(macd[i - window:i + window])
            
            # Klassische Divergenz (z. B. Preis fällt, RSI steigt)
            if (price_idx == i and rsi_idx != i and abs(prices[i] - price_low) / price_low < candle_tol and
                abs(macd[i] - macd_low) < macd_tol):
                classic_divs.append({
                    "div_id": len(classic_divs) + 1,
                    "date": int(dates[i]),
                    "type": "bullish",
                    "strength": min(1.0, abs(rsi[i] - rsi_low) / 10),
                    "low": prices[i],
                    "rsi": rsi[i],
                    "macd": macd[i],
                    "window": window
                })
            
            # Versteckte Divergenz (z. B. Preis steigt, RSI fällt)
            if (price_idx != i and rsi_idx == i and abs(prices[i] - price_low) / price_low < candle_tol and
                abs(macd[i] - macd_low) < macd_tol):
                hidden_divs.append({
                    "div_id": len(hidden_divs) + 1,
                    "date": int(dates[i]),
                    "type": "bearish",
                    "strength": min(1.0, abs(rsi[i] - rsi_low) / 10),
                    "low": prices[i],
                    "rsi": rsi[i],
                    "macd": macd[i],
                    "window": window
                })
        
        results[variant["id"]] = {
            "classic": classic_divs,
            "hidden": hidden_divs,
            "total": len(classic_divs) + len(hidden_divs)
        }
    
    # Collect mit Streaming
    df_filtered = df.select([
        "date", "open", "high", "low", "close", "rsi", "macd",
        "ema20", "ema50", "ema100", "ema200"
    ]).collect(streaming=True)
    
    chart_data = {
        "dates": df_filtered["date"].to_list(),
        "open": df_filtered["open"].to_list(),
        "high": df_filtered["high"].to_list(),
        "low": df_filtered["low"].to_list(),
        "close": df_filtered["close"].to_list(),
        "rsi": df_filtered["rsi"].to_list(),
        "macd_histogram": df_filtered["macd"].to_list(),
        "ema20": df_filtered.get_column("ema20", []).to_list(),
        "ema50": df_filtered.get_column("ema50", []).to_list(),
        "ema100": df_filtered.get_column("ema100", []).to_list(),
        "ema200": df_filtered.get_column("ema200", []).to_list()
    }
    
    return {"success": True, "chartData": chart_data, "results": results}