C:\Projekte\crt_250816\data\processed\btc_1day_candlesticks_all.parquet

.\.venv\Scripts\activate
python run_doe_grid.py `
  --input "C:\Projekte\crt_250816\data\raw\btc_1week_candlesticks_all.csv" `
  --w 2..8..3 `
  --ct 0.1..0.6..0.5 `
  --mt 1.25..5.25..2.0 `
  --out "C:\Projekte\crt_250816\results"

.\.venv\Scripts\activate
python run_doe_grid.py `
  --input "C:\Projekte\crt_250816\data\processed\btc_1day_candlesticks_all.parquet" `
  --w 2..8..3 `
  --ct 0.05..1..0.5 `
  --mt 1.25..5.25..2.0 `
  --out "C:\Projekte\crt_250816\results"

.\.venv\Scripts\activate
python run_doe_grid.py `
  --input "C:\Projekte\crt_250816\data\processed\btc_1day_candlesticks_all.parquet" `
  --w 5..5..0 `
  --ct 0.1..0.1..0.0 `
  --mt 1.25..5.25..2.0 `
  --out "C:\Projekte\crt_250816\results"