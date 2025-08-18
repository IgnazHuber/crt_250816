"""
Mainframe für Technische Chartanalyse - Korrigierte Version
Basiert auf dem Original Mainframe_RT.py mit Error Handling
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import finplot as fplt

# Eigene Module importieren
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    print("Stelle sicher, dass alle Module im gleichen Verzeichnis sind:")
    print("- Initialize_RSI_EMA_MACD.py")
    print("- Local_Maximas_Minimas.py") 
    print("- CBullDivg_Analysis_vectorized.py")
    sys.exit(1)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_analyze_data(csv_file_path):
    """
    Lädt Daten und führt komplette Analyse durch wie im Original
    """
    try:
        # Datei prüfen
        if not Path(csv_file_path).exists():
            logger.error(f"Datei nicht gefunden: {csv_file_path}")
            return None
        
        logger.info(f"Lade Daten aus: {csv_file_path}")
        
        # CSV laden (genau wie im Original)
        df = pd.read_csv(csv_file_path, low_memory=False)
        logger.info(f"Daten geladen: {len(df)} Zeilen")
        
        # Basis-Validierung
        required_columns = ['date', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Fehlende Spalten: {missing_columns}")
            return None
        
        logger.info("Starte Indikator-Berechnung...")
        
        # Schritt 1: RSI, EMA, MACD berechnen (wie im Original)
        result = Initialize_RSI_EMA_MACD(df)
        if result is None:
            logger.error("Fehler bei Initialize_RSI_EMA_MACD")
            return None
        
        # Schritt 2: Lokale Maxima/Minima finden (wie im Original)
        Local_Max_Min(df)
        logger.info("Lokale Extrema berechnet")
        
        # Schritt 3: Bullish Divergenzen analysieren (wie im Original)
        CBullDivg_analysis(df, 5, 0.1, 3.25)
        logger.info("Bullish Divergenzen analysiert")
        
        # Statistiken ausgeben
        gen_count = (df['CBullD_gen'] == 1).sum() if 'CBullD_gen' in df.columns else 0
        neg_macd_count = (df['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in df.columns else 0
        
        logger.info(f"Gefundene Divergenzen - Classic: {gen_count}, Negative MACD: {neg_macd_count}")
        
        return df
        
    except Exception as e:
        logger.error(f"Fehler bei Datenanalyse: {e}")
        return None

def crea