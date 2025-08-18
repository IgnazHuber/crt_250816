# -*- coding: utf-8 -*-
# file: Mainframe_RT_DOE.py
"""
Mainframe für Technische Chartanalyse - Stabilisierte Version + DOE-API

Erweiterungen (ohne Features zu entfernen):
- Parameter-Übergabe (window, candle_tolerance, macd_tolerance) via API & CLI
- Öffentliche API-Funktionen: analyze(), run(), main()
- Rückgabe der Divergenz-Zählungen für DOE (classic_count, neg_macd_count)
- Optionales Plot-Flag (enable_plot), Standard wie bisher bei CLI-Start mit Chart

Bestehende Features bleiben erhalten:
- Laden CSV/Parquet
- Initialize_RSI_EMA_MACD
- Local_Max_Min
- CBullDivg_analysis
- Finplot-Chart mit EMAs/RSI/MACD + Markierungen
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import finplot as fplt

# Eigene Module importieren
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD  # Korrigierter Import
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
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


@dataclass
class DivergenceParams:
    window: int = 5
    candle_tolerance: float = 0.5  # (vorheriger Stand aus Datei beibehalten)
    macd_tolerance: float = 3.25


class ChartAnalyzer:
    """
    Hauptklasse für die Chartanalyse mit Bullish Divergenz Erkennung
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        divergence_params: Optional[dict] = None,
        enable_plot: bool = True
    ):
        """
        Initialisierung mit optionaler Konfiguration + parametrisierbaren Divergenz-Parametern
        """
        # Standard-Konfiguration (unverändert lassen)
        self.config = config or {
            'divergence': {
                # Ursprüngliche Defaults beibehalten:
                'window': 5,
                'candle_tolerance': 0.5,
                'macd_tolerance': 3.25
            },
            'visualization': {
                'background': "#FFFFFF",
                'crosshair_color': '#eefa'
            }
        }
        # falls neue Parameter übergeben -> überschreiben
        if divergence_params:
            self.config['divergence'].update(divergence_params)

        self.enable_plot = enable_plot
        self.df: Optional[pd.DataFrame] = None
        logger.info("ChartAnalyzer initialisiert")

    # ---------------------------
    # Daten laden
    # ---------------------------
    def load_data(self, file_path: str) -> bool:
        """
        Lädt Daten aus CSV oder Parquet Datei
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
                # robust: verschiedene Engines probieren
                for engine in ("pyarrow", "fastparquet", None):
                    try:
                        self.df = (pd.read_parquet(file_path, engine=engine)
                                   if engine else pd.read_parquet(file_path))
                        break
                    except Exception:
                        continue
                if self.df is None:
                    logger.error(f"Parquet konnte nicht gelesen werden: {file_path}")
                    return False
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

    # ---------------------------
    # Indikatoren
    # ---------------------------
    def calculate_indicators(self) -> bool:
        """
        Berechnet alle technischen Indikatoren
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

    # ---------------------------
    # Extrema
    # ---------------------------
    def find_extrema(self) -> bool:
        """
        Findet lokale Maxima und Minima
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

    # ---------------------------
    # Divergenzen
    # ---------------------------
    def analyze_divergences(self) -> bool:
        """
        Analysiert Bullish Divergenzen
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False

            logger.info("Analysiere Bullish Divergenzen...")

            config = self.config['divergence']
            result = CBullDivg_analysis(
                self.df.copy(),
                int(config['window']),
                float(config['candle_tolerance']),
                float(config['macd_tolerance'])
            )

            if result is None:
                logger.error("Fehler bei Divergenz-Analyse")
                return False

            # Statistiken
            gen_count = int((self.df['CBullD_gen'] == 1).sum()) if 'CBullD_gen' in self.df.columns else 0
            neg_macd_count = int((self.df['CBullD_neg_MACD'] == 1).sum()) if 'CBullD_neg_MACD' in self.df.columns else 0

            logger.info(f"Divergenzen gefunden - Classic: {gen_count}, Negative MACD: {neg_macd_count}")
            return True

        except Exception as e:
            logger.error(f"Fehler bei Divergenz-Analyse: {e}")
            return False

    # ---------------------------
    # Chart
    # ---------------------------
    def create_chart(self) -> bool:
        """
        Erstellt und zeigt das Chart
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

    def _plot_divergences(self, ax1, ax2, ax3) -> None:
        """
        Plottet Divergenz-Markierungen auf den Charts
        """
        try:
            # Classic Bullish Divergenzen (CBullD_gen)
            if 'CBullD_gen' in self.df.columns:
                for i in range(2, len(self.df)):
                    if self.df['CBullD_gen'][i] == 1:
                        # Preis-Chart Markierungen
                        if 'CBullD_Lower_Low_date_gen' in self.df.columns and pd.notna(self.df['CBullD_Lower_Low_date_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Lower_Low_date_gen'][i]),
                                      self.df['CBullD_Lower_Low_gen'][i],
                                      style='x', ax=ax1, color='red')
                        if 'CBullD_Higher_Low_date_gen' in self.df.columns and pd.notna(self.df['CBullD_Higher_Low_date_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Higher_Low_date_gen'][i]),
                                      self.df['CBullD_Higher_Low_gen'][i],
                                      style='x', ax=ax1, color='blue')

                        # RSI Markierungen
                        if 'CBullD_Lower_Low_RSI_gen' in self.df.columns and pd.notna(self.df['CBullD_Lower_Low_RSI_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Lower_Low_date_gen'][i]),
                                      self.df['CBullD_Lower_Low_RSI_gen'][i],
                                      style='x', ax=ax2, color='red')
                        if 'CBullD_Higher_Low_RSI_gen' in self.df.columns and pd.notna(self.df['CBullD_Higher_Low_RSI_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Higher_Low_date_gen'][i]),
                                      self.df['CBullD_Higher_Low_RSI_gen'][i],
                                      style='x', ax=ax2, color='blue')

                        # MACD Markierungen
                        if 'CBullD_Lower_Low_MACD_gen' in self.df.columns and pd.notna(self.df['CBullD_Lower_Low_MACD_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Lower_Low_date_gen'][i]),
                                      self.df['CBullD_Lower_Low_MACD_gen'][i],
                                      style='x', ax=ax3, color='red')
                        if 'CBullD_Higher_Low_MACD_gen' in self.df.columns and pd.notna(self.df['CBullD_Higher_Low_MACD_gen'][i]):
                            fplt.plot(pd.to_datetime(self.df['CBullD_Higher_Low_date_gen'][i]),
                                      self.df['CBullD_Higher_Low_MACD_gen'][i],
                                      style='x', ax=ax3, color='blue')

            # Negative MACD Divergenzen (Marker-Struktur vorbereitet)
            if 'CBullD_neg_MACD' in self.df.columns:
                for i in range(2, len(self.df)):
                    if self.df['CBullD_neg_MACD'][i] == 1:
                        # Optional separate Markierung implementierbar
                        pass

        except Exception as e:
            logger.error(f"Fehler beim Plotten der Divergenzen: {e}")

    # ---------------------------
    # Komplettlauf
    # ---------------------------
    def run_analysis(self, file_path: str) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Daten laden -> Indikatoren -> Extrema -> Divergenzen -> optional Chart
        Gibt zusätzlich Zählungen zurück (für DOE).
        """
        try:
            logger.info("Starte komplette Chartanalyse...")

            # Schritt 1: Daten laden
            if not self.load_data(file_path):
                return False, None

            # Schritt 2: Technische Indikatoren berechnen
            if not self.calculate_indicators():
                return False, None

            # Schritt 3: Lokale Extrema finden
            if not self.find_extrema():
                return False, None

            # Schritt 4: Divergenzen analysieren
            if not self.analyze_divergences():
                return False, None

            # Statistiken
            gen_count = int((self.df['CBullD_gen'] == 1).sum()) if 'CBullD_gen' in self.df.columns else 0
            neg_macd_count = int((self.df['CBullD_neg_MACD'] == 1).sum()) if 'CBullD_neg_MACD' in self.df.columns else 0
            stats = {"classic": gen_count, "neg_macd": neg_macd_count}

            # Schritt 5: Chart erstellen (nur wenn gewünscht)
            if self.enable_plot:
                if not self.create_chart():
                    return False, stats

            logger.info("Chartanalyse erfolgreich abgeschlossen")
            return True, stats

        except Exception as e:
            logger.error(f"Fehler bei kompletter Analyse: {e}")
            return False, None


# =========================
# Öffentliche API-Funktionen für DOE
# =========================
def _autodetect_input() -> Optional[str]:
    candidates = [
        # Ursprüngliche Examples (beibehalten):
        r'C:\Projekte\crt_250816\data\raw\btc_1week_candlesticks_all.csv',
        'data/sp500_data.csv',
        'data/test_data.parquet',
        'test_data.csv'
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    # generische Suche
    for base in ("./data", "."):
        if not Path(base).is_dir():
            continue
        for name in os.listdir(base):
            if Path(name).suffix.lower() in ('.csv', '.parquet'):
                return str(Path(base) / name)
    return None


def analyze(
    window: int = DivergenceParams.window,
    candle_tolerance: float = DivergenceParams.candle_tolerance,
    macd_tolerance: float = DivergenceParams.macd_tolerance,
    input_path: Optional[str] = None,
    enable_plot: bool = False
) -> Dict[str, Any]:
    """
    DOE-tauglicher programmatischer Aufruf.
    Gibt ein Dict mit Counts zurück und (für Kompatibilität) keine Features werden entfernt.
    """
    if input_path is None:
        input_path = _autodetect_input()
    if input_path is None:
        raise FileNotFoundError("Keine Eingabedatei gefunden. Bitte input_path angeben oder Datei in ./data ablegen.")

    analyzer = ChartAnalyzer(
        divergence_params={
            "window": int(window),
            "candle_tolerance": float(candle_tolerance),
            "macd_tolerance": float(macd_tolerance),
        },
        enable_plot=enable_plot
    )
    ok, stats = analyzer.run_analysis(input_path)
    if not ok or stats is None:
        stats = {"classic": 0, "neg_macd": 0}

    # Kompatibles Rückgabeformat für DOE-Runner
    return {
        "classic_count": int(stats.get("classic", 0)),
        "neg_macd_count": int(stats.get("neg_macd", 0)),
        "details": analyzer.df if hasattr(analyzer, "df") else None
    }


def run(**kwargs) -> Dict[str, Any]:
    return analyze(**kwargs)


def main(
    window: int = DivergenceParams.window,
    candle_tolerance: float = DivergenceParams.candle_tolerance,
    macd_tolerance: float = DivergenceParams.macd_tolerance,
    input_path: Optional[str] = None,
    enable_plot: bool = True
) -> Dict[str, Any]:
    """
    Haupt-Einsprungpunkt als API (nicht zu verwechseln mit __main__).
    """
    return analyze(window, candle_tolerance, macd_tolerance, input_path, enable_plot)


# =========================
# CLI
# =========================
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mainframe RT DOE – Analyse & Chart")
    p.add_argument("--input", "-i", dest="input_path", type=str, default=None,
                   help="Pfad zu Eingabedaten (.csv | .parquet)")
    p.add_argument("--window", "-w", type=int, default=DivergenceParams.window,
                   help=f"Lookback-Fenster (default: {DivergenceParams.window})")
    p.add_argument("--candle-tolerance", "--ct", type=float, default=DivergenceParams.candle_tolerance,
                   help=f"Candle-Toleranz (default: {DivergenceParams.candle_tolerance})")
    p.add_argument("--macd-tolerance", "--mt", type=float, default=DivergenceParams.macd_tolerance,
                   help=f"MACD-Toleranz (default: {DivergenceParams.macd_tolerance})")
    p.add_argument("--no-plot", action="store_true", help="Chart rendering deaktivieren (für Batch/DOE)")
    return p.parse_args(argv)


def _legacy_possible_files() -> list:
    # Liste aus der ursprünglichen Datei beibehalten
    return [
        r'C:\Projekte\crt_250816\data\raw\btc_1week_candlesticks_all.csv',
        'data/sp500_data.csv',
        'data/test_data.parquet',
        'test_data.csv'
    ]


if __name__ == "__main__":
    # CLI-Verhalten: wie bisher Chart + Show, aber parametrisierbar
    args = _parse_args()
    input_path = args.input_path or _autodetect_input()
    if input_path is None:
        print("Keine Datendatei gefunden. Bitte Pfad angeben oder Datei bereitstellen.")
        print("Gesuchte Dateien (Beispiele):")
        for f in _legacy_possible_files():
            print(f"  - {f}")
        sys.exit(1)

    res = main(
        window=args.window,
        candle_tolerance=args.candle_tolerance,
        macd_tolerance=args.macd_tolerance,
        input_path=input_path,
        enable_plot=(not args.no_plot)
    )

    # kompakte Zusammenfassung
    print("\n=== Zusammenfassung ===")
    print(f"Classic Divergences:       {res.get('classic_count', 0)}")
    print(f"Negative MACD Divergences: {res.get('neg_macd_count', 0)}")

    # Chart anzeigen, wenn aktiv
    if not args.no_plot:
        try:
            fplt.show()
        except Exception as e:
            logger.error(f"Fehler bei fplt.show(): {e}")
