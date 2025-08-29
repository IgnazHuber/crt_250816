"""
Mainframe für Technische Chartanalyse - Stabilisierte Version
Kritische Fehler behoben + grundlegendes Error Handling
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import finplot as fplt

# Eigene Module importieren
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD  # Korrigierter Import
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    #from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
    #from HBearDivg_analysis_vectorized import HBearDivg_analysis
    #from HBullDivg_analysis_vectorized import HBullDivg_analysis
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chartanalyse.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChartAnalyzer:
    """
    Hauptklasse für die Chartanalyse mit Bullish Divergenz Erkennung
    """
    
    def __init__(self, config=None):
        """
        Initialisierung mit optionaler Konfiguration
        """
        # Standard-Konfiguration
        self.config = config or {
            'divergence': {
                #'window': 5,
                #'candle_tolerance': 0.1,
                #'macd_tolerance': 3.25
                'window': 5,
                'candle_tolerance': 0.1,
                'macd_tolerance': 3.25
            },
            'visualization': {
                'background': "#FFFFFF",
                'crosshair_color': '#eefa'
            }
        }
        
        self.df = None
        logger.info("ChartAnalyzer initialisiert")
    
    def load_data(self, file_path):
        """
        Lädt Daten aus CSV oder Parquet Datei
        
        Args:
            file_path (str): Pfad zur Datendatei
            
        Returns:
            bool: True wenn erfolgreich geladen
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Datei nicht gefunden: {file_path}")
                return False
            
            # Dateierweiterung prüfen
            if file_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(file_path, low_memory=False)
                logger.info(f"CSV-Datei geladen: {file_path}")
            elif file_path.suffix.lower() == '.parquet':
                self.df = pd.read_parquet(file_path)
                logger.info(f"Parquet-Datei geladen: {file_path}")
            else:
                logger.error(f"Nicht unterstütztes Dateiformat: {file_path.suffix}")
                return False
            
            # Basis-Validierung
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                logger.error(f"Fehlende Spalten: {missing_columns}")
                return False
            
            logger.info(f"Daten geladen: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {e}")
            return False
    
    def calculate_indicators(self):
        """
        Berechnet alle technischen Indikatoren
        
        Returns:
            bool: True wenn erfolgreich berechnet
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False
            
            logger.info("Berechne technische Indikatoren...")
            
            # RSI, EMA, MACD berechnen
            result = Initialize_RSI_EMA_MACD(self.df)
            if result is None:
                logger.error("Fehler bei der Indikator-Berechnung")
                return False
            
            logger.info("Technische Indikatoren berechnet")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Indikator-Berechnung: {e}")
            return False
    
    def find_extrema(self):
        """
        Findet lokale Maxima und Minima
        
        Returns:
            bool: True wenn erfolgreich berechnet
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False
            
            logger.info("Suche lokale Extrema...")
            Local_Max_Min(self.df)
            logger.info("Lokale Extrema gefunden")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Extrema-Suche: {e}")
            return False
    
    def analyze_divergences(self):
        """
        Analysiert Bullish Divergenzen
        
        Returns:
            bool: True wenn erfolgreich analysiert
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False
            
            logger.info("Analysiere Bullish Divergenzen...")
            
            config = self.config['divergence']
            result = CBullDivg_analysis(
                self.df, 
                config['window'], 
                config['candle_tolerance'], 
                config['macd_tolerance']
            )
            
            if result is None:
                logger.error("Fehler bei Divergenz-Analyse")
                return False
            
            # Statistiken
            gen_count = (self.df['CBullD_gen'] == 1).sum()
            neg_macd_count = (self.df['CBullD_neg_MACD'] == 1).sum()
            
            logger.info(f"Divergenzen gefunden - Classic: {gen_count}, Negative MACD: {neg_macd_count}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Divergenz-Analyse: {e}")
            return False
    
    def create_chart(self):
        """
        Erstellt und zeigt das Chart
        
        Returns:
            bool: True wenn erfolgreich erstellt
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False
            
            logger.info("Erstelle Chart...")
            
            # Plot-Konfiguration
            config = self.config['visualization']
            fplt.background = fplt.odd_plot_background = config['background']
            fplt.cross_hair_color = config['crosshair_color']
            
            # 3-Panel Chart erstellen
            ax1, ax2, ax3 = fplt.create_plot('Technische Chartanalyse', rows=3)
            
            # Datum konvertieren
            self.df['date'] = pd.to_datetime(self.df['date'], format='mixed')
            
            # Candlestick Chart (Panel 1)
            candles = self.df[['date', 'open', 'close', 'high', 'low']]
            fplt.candlestick_ochl(candles, ax=ax1)
            
            # EMAs plotten
            if 'EMA_20' in self.df.columns:
                self.df.EMA_20.plot(ax=ax1, legend='20-EMA')
            if 'EMA_50' in self.df.columns:
                self.df.EMA_50.plot(ax=ax1, legend='50-EMA')
            if 'EMA_100' in self.df.columns:
                self.df.EMA_100.plot(ax=ax1, legend='100-EMA')
            if 'EMA_200' in self.df.columns:
                self.df.EMA_200.plot(ax=ax1, legend='200-EMA')
            
            # RSI (Panel 2)
            if 'RSI' in self.df.columns:
                fplt.plot(self.df.RSI, color='#000000', width=2, ax=ax2, legend='RSI')
                fplt.set_y_range(0, 100, ax=ax2)
                fplt.add_horizontal_band(0, 1, color='#000000', ax=ax2)
                fplt.add_horizontal_band(99, 100, color='#000000', ax=ax2)
            
            # MACD Histogram (Panel 3)
            if 'macd_histogram' in self.df.columns:
                macd_data = self.df[['date', 'open', 'close', 'macd_histogram']]
                fplt.volume_ocv(macd_data, ax=ax3, colorfunc=fplt.strength_colorfilter)
            
            # Divergenzen markieren
            self._plot_divergences(ax1, ax2, ax3)
            
            logger.info("Chart erstellt")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Chart-Erstellung: {e}")
            return False
    
    def _plot_divergences(self, ax1, ax2, ax3):
        """
        Plottet Divergenz-Markierungen auf den Charts
        """
        try:
            # Classic Bullish Divergenzen (CBullD_gen)
            if 'CBullD_gen' in self.df.columns:
                for i in range(2, len(self.df)):
                    if self.df['CBullD_gen'][i] == 1:
                        # Preis-Chart Markierungen
                        if pd.notna(self.df['CBullD_Lower_Low_date_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Lower_Low_date_gen'][i]), 
                                    self.df['CBullD_Lower_Low_gen'][i], 
                                    style='x', ax=ax1, color='red')
                        if pd.notna(self.df['CBullD_Higher_Low_date_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Higher_Low_date_gen'][i]), 
                                    self.df['CBullD_Higher_Low_gen'][i], 
                                    style='x', ax=ax1, color='blue')
                        
                        # RSI Markierungen
                        if pd.notna(self.df['CBullD_Lower_Low_RSI_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Lower_Low_date_gen'][i]), 
                                    self.df['CBullD_Lower_Low_RSI_gen'][i], 
                                    style='x', ax=ax2, color='red')
                        if pd.notna(self.df['CBullD_Higher_Low_RSI_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Higher_Low_date_gen'][i]), 
                                    self.df['CBullD_Higher_Low_RSI_gen'][i], 
                                    style='x', ax=ax2, color='blue')
                        
                        # MACD Markierungen
                        if pd.notna(self.df['CBullD_Lower_Low_MACD_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Lower_Low_date_gen'][i]), 
                                    self.df['CBullD_Lower_Low_MACD_gen'][i], 
                                    style='x', ax=ax3, color='red')
                        if pd.notna(self.df['CBullD_Higher_Low_MACD_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Higher_Low_date_gen'][i]), 
                                    self.df['CBullD_Higher_Low_MACD_gen'][i], 
                                    style='x', ax=ax3, color='blue')
            
            # Negative MACD Divergenzen
            if 'CBullD_neg_MACD' in self.df.columns:
                for i in range(2, len(self.df)):
                    if self.df['CBullD_neg_MACD'][i] == 1:
                        # Ähnliche Logik für negative MACD Divergenzen
                        # (Markierungen in anderen Farben/Stilen für Unterscheidung)
                        pass  # Implementierung analog zu oben
            
        except Exception as e:
            logger.error(f"Fehler beim Plotten der Divergenzen: {e}")
    
    def run_analysis(self, file_path):
        """
        Führt komplette Analyse durch: Daten laden -> Indikatoren -> Extrema -> Divergenzen -> Chart
        
        Args:
            file_path (str): Pfad zur Datendatei
            
        Returns:
            bool: True wenn komplette Analyse erfolgreich
        """
        try:
            logger.info("Starte komplette Chartanalyse...")
            
            # Schritt 1: Daten laden
            if not self.load_data(file_path):
                return False
            
            # Schritt 2: Technische Indikatoren berechnen
            if not self.calculate_indicators():
                return False
            
            # Schritt 3: Lokale Extrema finden
            if not self.find_extrema():
                return False
            
            # Schritt 4: Divergenzen analysieren
            if not self.analyze_divergences():
                return False
            
            # Schritt 5: Chart erstellen
            if not self.create_chart():
                return False
            
            logger.info("Chartanalyse erfolgreich abgeschlossen")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei kompletter Analyse: {e}")
            return False


def main():
    """
    Hauptfunktion - kann mit verschiedenen Datenquellen getestet werden
    """
    try:
        # Konfigurierbare Datenpfade (nicht mehr hardcoded)
        possible_data_files = [
            #r'C:\Projekte\crt_250816\data\raw\btc_1day_candlesticks_all.csv',  # Original
            r'C:\Projekte\crt_250816\data\raw\btc_1week_candlesticks_all.csv',  # Original
            'data/sp500_data.csv',  # Relative Pfade
            'data/test_data.parquet',
            'test_data.csv'
        ]
        
        # Ersten verfügbaren Datensatz finden
        data_file = None
        for file_path in possible_data_files:
            if Path(file_path).exists():
                data_file = file_path
                break
        
        if data_file is None:
            print("Keine Datendatei gefunden. Bitte Pfad anpassen oder Datei bereitstellen.")
            print("Gesuchte Dateien:")
            for file_path in possible_data_files:
                print(f"  - {file_path}")
            return
        
        # Chartanalyse durchführen
        analyzer = ChartAnalyzer()
        
        if analyzer.run_analysis(data_file):
            print("Analyse erfolgreich! Chart wird angezeigt...")
            fplt.show()
        else:
            print("Analyse fehlgeschlagen. Siehe chartanalyse.log für Details.")
            
    except KeyboardInterrupt:
        logger.info("Analyse vom Benutzer abgebrochen")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler in main(): {e}")


if __name__ == "__main__":
    main()