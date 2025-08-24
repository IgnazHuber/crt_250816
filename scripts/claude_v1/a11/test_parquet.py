import pandas as pd

# Ersetze 'deine_datei.parquet' mit dem Pfad zu deiner Parquet-Datei
file_path = 'C:/Projekte/crt_250816/data/processed/btc_1day_candlesticks_all.parquet'

try:
    df = pd.read_parquet(file_path, engine='pyarrow')
    print("✅ Datei erfolgreich geladen!")
    print("Spalten:", df.columns.tolist())
    print("Erste Zeilen:\n", df.head())
    # Prüfe die benötigten Spalten
    required = ['date', 'open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Fehlende Spalten: {missing}")
    else:
        print("✅ Alle benötigten Spalten vorhanden")
except Exception as e:
    print(f"❌ Fehler beim Laden: {e}")