"""
Interaktive Chartanalyse mit Plotly und Dateiauswahl
Ersetzt finplot für bessere Interaktivität und moderne UI
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Eigene Module importieren
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas_optimized import Local_Max_Min_Optimized
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Verwaltet Konfiguration und Pfad-Speicherung
    """
    
    def __init__(self, config_file: str = "chart_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt Konfiguration aus JSON-Datei"""
        default_config = {
            "last_data_path": "",
            "preferred_formats": [".csv", ".parquet"],
            "divergence": {
                "window": 5,
                "candle_tolerance": 0.1,
                "macd_tolerance": 3.25
            },
            "visualization": {
                "theme": "plotly_dark",
                "colors": {
                    "bullish_divergence": "#00ff00",
                    "bearish_divergence": "#ff0000",
                    "ema_20": "#ff6b6b",
                    "ema_50": "#4ecdc4",
                    "ema_100": "#45b7d1",
                    "ema_200": "#96ceb4"
                }
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge mit default config
                    default_config.update(loaded_config)
                    return default_config
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Konfiguration: {e}")
        
        return default_config
    
    def save_config(self):
        """Speichert aktuelle Konfiguration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
    
    def update_last_path(self, path: str):
        """Aktualisiert den zuletzt verwendeten Pfad"""
        self.config["last_data_path"] = str(Path(path).parent)
        self.save_config()

class DataFileSelector:
    """
    GUI für Dateiauswahl mit Pfad-Speicherung
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.selected_file = None
    
    def select_file(self) -> Optional[str]:
        """
        Öffnet Dateiauswahl-Dialog
        
        Returns:
            Pfad zur ausgewählten Datei oder None
        """
        # Verstecke das Haupt-Tkinter-Fenster
        root = tk.Tk()
        root.withdraw()
        
        # Bestimme Startverzeichnis
        initial_dir = self.config_manager.config.get("last_data_path", "")
        if not initial_dir or not Path(initial_dir).exists():
            initial_dir = os.getcwd()
        
        # Dateiformate
        filetypes = [
            ("CSV Dateien", "*.csv"),
            ("Parquet Dateien", "*.parquet"),
            ("Alle unterstützten", "*.csv;*.parquet"),
            ("Alle Dateien", "*.*")
        ]
        
        try:
            file_path = filedialog.askopenfilename(
                title="Chartdaten auswählen",
                initialdir=initial_dir,
                filetypes=filetypes
            )
            
            if file_path:
                self.selected_file = file_path
                self.config_manager.update_last_path(file_path)
                logger.info(f"Datei ausgewählt: {file_path}")
                return file_path
            
        except Exception as e:
            logger.error(f"Fehler bei Dateiauswahl: {e}")
            messagebox.showerror("Fehler", f"Fehler bei Dateiauswahl: {e}")
        
        finally:
            root.destroy()
        
        return None

class InteractiveChartAnalyzer:
    """
    Hauptklasse für interaktive Chartanalyse mit Plotly
    """
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.file_selector = DataFileSelector(self.config_manager)
        self.df = None
        self.fig = None
    
    def load_data(self, file_path: str) -> bool:
        """
        Lädt Daten aus CSV oder Parquet Datei
        
        Args:
            file_path: Pfad zur Datendatei
            
        Returns:
            True wenn erfolgreich geladen
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Datei nicht gefunden: {file_path}")
                return False
            
            logger.info(f"Lade Daten: {file_path}")
            
            # Datei basierend auf Erweiterung laden
            if file_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix.lower() == '.parquet':
                self.df = pd.read_parquet(file_path)
            else:
                logger.error(f"Nicht unterstütztes Dateiformat: {file_path.suffix}")
                return False
            
            # Validierung
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                logger.error(f"Fehlende Spalten: {missing_columns}")
                return False
            
            # Datum konvertieren
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            logger.info(f"✅ Daten geladen: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")
            logger.info(f"Zeitraum: {self.df['date'].min()} bis {self.df['date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {e}")
            return False
    
    def calculate_all_indicators(self) -> bool:
        """
        Berechnet alle technischen Indikatoren und Divergenzen
        
        Returns:
            True wenn erfolgreich berechnet
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False
            
            logger.info("🔧 Berechne technische Indikatoren...")
            
            # Schritt 1: RSI, EMA, MACD
            result = Initialize_RSI_EMA_MACD(self.df)
            if result is None:
                logger.error("Fehler bei Indikator-Berechnung")
                return False
            
            # Schritt 2: Lokale Extrema (optimierte Version)
            logger.info("🔍 Suche lokale Extrema (optimiert)...")
            Local_Max_Min_Optimized(self.df)
            
            # Schritt 3: Divergenz-Analyse
            logger.info("📈 Analysiere Bullish Divergenzen...")
            config = self.config_manager.config['divergence']
            CBullDivg_analysis(
                self.df, 
                config['window'], 
                config['candle_tolerance'], 
                config['macd_tolerance']
            )
            
            # Statistiken
            if 'CBullD_gen' in self.df.columns:
                classic_count = (self.df['CBullD_gen'] == 1).sum()
                logger.info(f"✅ Classic Bullish Divergenzen: {classic_count}")
            
            if 'CBullD_neg_MACD' in self.df.columns:
                neg_macd_count = (self.df['CBullD_neg_MACD'] == 1).sum()
                logger.info(f"✅ Negative MACD Divergenzen: {neg_macd_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Indikator-Berechnung: {e}")
            return False
    
    def create_interactive_chart(self) -> bool:
        """
        Erstellt interaktives Plotly Chart
        
        Returns:
            True wenn erfolgreich erstellt
        """
        try:
            if self.df is None:
                logger.error("Keine Daten geladen")
                return False
            
            logger.info("📊 Erstelle interaktives Chart...")
            
            colors = self.config_manager.config['visualization']['colors']
            
            # 3-Panel Subplot erstellen
            self.fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=('Preis & EMAs', 'RSI', 'MACD Histogram'),
                row_heights=[0.6, 0.2, 0.2],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # ================== PANEL 1: CANDLESTICK & EMAs ==================
            
            # Candlestick Chart
            self.fig.add_trace(
                go.Candlestick(
                    x=self.df['date'],
                    open=self.df['open'],
                    high=self.df['high'],
                    low=self.df['low'],
                    close=self.df['close'],
                    name="Preis",
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
            
            # EMAs hinzufügen
            ema_columns = ['EMA_20', 'EMA_50', 'EMA_100', 'EMA_200']
            ema_colors = [colors['ema_20'], colors['ema_50'], colors['ema_100'], colors['ema_200']]
            
            for ema, color in zip(ema_columns, ema_colors):
                if ema in self.df.columns:
                    self.fig.add_trace(
                        go.Scatter(
                            x=self.df['date'],
                            y=self.df[ema],
                            mode='lines',
                            name=ema,
                            line=dict(color=color, width=1.5),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
            
            # ================== PANEL 2: RSI ==================
            
            if 'RSI' in self.df.columns:
                self.fig.add_trace(
                    go.Scatter(
                        x=self.df['date'],
                        y=self.df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#ffaa00', width=2)
                    ),
                    row=2, col=1
                )
                
                # RSI Überkauft/Überverkauft Linien
                self.fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                self.fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
                self.fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
            
            # ================== PANEL 3: MACD HISTOGRAM ==================
            
            if 'macd_histogram' in self.df.columns:
                # Positive und negative Balken unterschiedlich färben
                colors_macd = ['#00ff88' if val >= 0 else '#ff4444' for val in self.df['macd_histogram']]
                
                self.fig.add_trace(
                    go.Bar(
                        x=self.df['date'],
                        y=self.df['macd_histogram'],
                        name='MACD Histogram',
                        marker_color=colors_macd,
                        opacity=0.7
                    ),
                    row=3, col=1
                )
                
                # Zero-Linie
                self.fig.add_hline(y=0, line_color="gray", opacity=0.5, row=3, col=1)
            
            # ================== DIVERGENZ-MARKIERUNGEN ==================
            
            self._add_divergence_markers()
            
            # ================== LAYOUT KONFIGURATION ==================
            
            self.fig.update_layout(
                title=dict(
                    text=f"Technische Chartanalyse - {Path(self.file_selector.selected_file).stem if self.file_selector.selected_file else 'Daten'}",
                    x=0.5,
                    font_size=16
                ),
                template=self.config_manager.config['visualization']['theme'],
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                height=800,
                hovermode='x unified'
            )
            
            # Y-Achsen Titel
            self.fig.update_yaxes(title_text="Preis", row=1, col=1)
            self.fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            self.fig.update_yaxes(title_text="MACD", row=3, col=1)
            
            # X-Achse nur unten beschriften
            self.fig.update_xaxes(title_text="Datum", row=3, col=1)
            
            logger.info("✅ Interaktives Chart erstellt")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei Chart-Erstellung: {e}")
            return False
    
    def _add_divergence_markers(self):
        """
        Fügt Divergenz-Markierungen zum Chart hinzu
        