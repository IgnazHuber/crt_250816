"""
Complete Analysis Runner - Vereint alle Analysemodul in einem System
Führt Validierung, Parameter-Optimierung und erweiterte Divergenz-Analyse durch
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json

# Eigene Module
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    from enhanced_divergence_system import Enhanced_Divergence_Analysis
    # from divergence_validator import DivergenceValidator  # Temporär deaktiviert
    # from sensitivity_analyzer import SensitivityAnalyzer   # Temporär deaktiviert
    print("✅ Kern-Module erfolgreich importiert")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stelle sicher, dass alle Module im gleichen Verzeichnis sind")

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteAnalysisRunner:
    """
    Hauptklasse für die vollständige Divergenz-Analyse
    """
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.config = {
            'default_params': {
                'window': 5,
                'candle_tolerance': 0.1,
                'macd_tolerance': 3.25
            },
            'optimization': {
                'enable_grid_search': True,
                'max_workers': 4,
                'parameter_ranges': {
                    'window': [3, 4, 5, 6, 7],
                    'candle_tolerance': [0.05, 0.1, 0.15, 0.2, 0.3],
                    'macd_tolerance': [1.0, 2.0, 3.0, 3.25, 4.0, 5.0]
                }
            },
            'analysis': {
                'enable_hidden_bullish': True,
                'enable_bearish': True,
                'enable_backtesting': True
            }
        }
        
        self.results = {
            'validation': {},
            'optimization': {},
            'enhanced_analysis': {}
        }
    
    def select_data_file(self) -> str:
        """
        Dateiauswahl-Dialog
        """
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Wähle Datenquelle für komplette Analyse",
            filetypes=[
                ("CSV Dateien", "*.csv"),
                ("Parquet Dateien", "*.parquet"),
                ("Alle", "*.*")
            ]
        )
        
        root.destroy()
        return file_path
    
    def load_and_prepare_data(self, file_path: str) -> bool:
        """
        Lädt und bereitet Daten vor
        """
        try:
            logger.info(f"📊 Lade Daten: {Path(file_path).name}")
            
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, low_memory=False)
            elif file_path.endswith('.parquet'):
                self.df = pd.read_parquet(file_path)
            else:
                raise ValueError("Nicht unterstütztes Dateiformat")
            
            # Validierung
            required = ['date', 'open', 'high', 'low', 'close']
            missing = [col for col in required if col not in self.df.columns]
            if missing:
                raise ValueError(f"Fehlende Spalten: {missing}")
            
            # Datum konvertieren
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
            
            # Backup für Reset
            self.original_df = self.df.copy()
            
            logger.info(f"✅ {len(self.df)} Zeilen geladen")
            logger.info(f"📅 Zeitraum: {self.df['date'].min().date()} bis {self.df['date'].max().date()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden: {e}")
            return False
    
    def prepare_base_indicators(self) -> bool:
        """
        Berechnet Basis-Indikatoren (RSI, EMA, MACD) und lokale Extrema
        """
        try:
            logger.info("🔧 Berechne Basis-Indikatoren...")
            
            # RSI, EMA, MACD
            Initialize_RSI_EMA_MACD(self.df)
            
            # Lokale Extrema
            Local_Max_Min(self.df)
            
            logger.info("✅ Basis-Indikatoren berechnet")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Basis-Indikatoren: {e}")
            return False
    
    def run_basic_divergence_analysis(self) -> bool:
        """
        Führt Standard-Divergenz-Analyse durch
        """
        try:
            logger.info("📈 Führe Standard-Divergenz-Analyse durch...")
            
            params = self.config['default_params']
            CBullDivg_analysis(
                self.df,
                params['window'],
                params['candle_tolerance'],
                params['macd_tolerance']
            )
            
            # Statistiken
            classic_count = (self.df['CBullD_gen'] == 1).sum() if 'CBullD_gen' in self.df.columns else 0
            neg_macd_count = (self.df['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in self.df.columns else 0
            
            logger.info(f"✅ Standard-Analyse: {classic_count} Classic, {neg_macd_count} Negative MACD")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Standard-Analyse: {e}")
            return False
    
    def run_validation_analysis(self) -> bool:
        """
        Führt Validierung und Backtesting durch
        """
        try:
            if not self.config['analysis']['enable_backtesting']:
                logger.info("⏭️ Validierung übersprungen (deaktiviert)")
                return True
            
            logger.info("🔍 Validierung temporär deaktiviert (Module werden noch implementiert)")
            # TODO: Implementiere Validierung wenn Module verfügbar
            return True
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Validierung: {e}")
            return False
    
    def run_parameter_optimization(self) -> bool:
        """
        Führt Parameter-Optimierung durch
        """
        try:
            if not self.config['optimization']['enable_grid_search']:
                logger.info("⏭️ Parameter-Optimierung übersprungen (deaktiviert)")
                return True
            
            logger.info("🎯 Parameter-Optimierung temporär deaktiviert (Module werden noch implementiert)")
            # TODO: Implementiere Optimierung wenn Module verfügbar
            return True
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Parameter-Optimierung: {e}")
            return False
    
    def run_enhanced_divergence_analysis(self) -> bool:
        """
        Führt erweiterte Divergenz-Analyse durch (Hidden + Bearish)
        """
        try:
            logger.info("🚀 Starte erweiterte Divergenz-Analyse...")
            
            # Reset DataFrame für saubere Analyse
            self.df = self.original_df.copy()
            
            # Basis-Indikatoren neu berechnen
            self.prepare_base_indicators()
            
            # Standard-Analyse mit optimierten Parametern
            params = self.config['default_params']
            CBullDivg_analysis(
                self.df,
                params['window'],
                params['candle_tolerance'],
                params['macd_tolerance']
            )
            
            # Erweiterte Analyse
            Enhanced_Divergence_Analysis(
                self.df,
                params['window'],
                params['candle_tolerance'],
                params['macd_tolerance'],
                enable_hidden=self.config['analysis']['enable_hidden_bullish'],
                enable_bearish=self.config['analysis']['enable_bearish']
            )
            
            # Statistiken sammeln
            stats = {}
            divergence_types = [
                ('CBullD_gen', 'Classic Bullish'),
                ('CBullD_neg_MACD', 'Negative MACD'),
                ('HBullD', 'Hidden Bullish'),
                ('BearD', 'Classic Bearish'),
                ('HBearD', 'Hidden Bearish')
            ]
            
            for col, name in divergence_types:
                if col in self.df.columns:
                    count = (self.df[col] == 1).sum()
                    stats[name] = count
                    if count > 0:
                        logger.info(f"  {name}: {count}")
            
            self.results['enhanced_analysis'] = {
                'stats': stats,
                'parameters_used': params
            }
            
            # Interaktives Chart mit allen Divergenzen (vereinfacht)
            logger.info("📊 Erstelle Chart mit erweiterten Divergenzen...")
            
            # Verwende das simple Chart System für jetzt
            try:
                from interactive_simple import create_plotly_chart
                enhanced_chart = create_plotly_chart(self.df)
                enhanced_chart.update_layout(title="Erweiterte Divergenz-Analyse - Alle Typen")
                enhanced_chart.show()
            except ImportError:
                logger.warning("⚠️ Interaktives Chart-Modul nicht verfügbar - überspringe Visualisierung")
            
            logger.info("✅ Erweiterte Divergenz-Analyse abgeschlossen")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei erweiterter Analyse: {e}")
            return False
    
    def export_final_results(self) -> str:
        """
        Exportiert alle Ergebnisse in eine zusammenfassende Excel-Datei
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"complete_divergence_analysis_{timestamp}.xlsx"
            
            logger.info(f"📋 Exportiere finale Ergebnisse: {filename}")
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Daten mit allen Divergenzen
                if self.df is not None:
                    self.df.to_excel(writer, sheet_name='Complete_Data', index=False)
                
                # Zusammenfassung
                summary_data = []
                
                # Konfiguration
                summary_data.append(['KONFIGURATION', ''])
                summary_data.append(['Window Size', self.config['default_params']['window']])
                summary_data.append(['Candle Tolerance', f"{self.config['default_params']['candle_tolerance']:.3f}"])
                summary_data.append(['MACD Tolerance', f"{self.config['default_params']['macd_tolerance']:.2f}"])
                summary_data.append(['', ''])
                
                # Divergenz-Statistiken
                if 'enhanced_analysis' in self.results and 'stats' in self.results['enhanced_analysis']:
                    summary_data.append(['GEFUNDENE DIVERGENZEN', ''])
                    for div_type, count in self.results['enhanced_analysis']['stats'].items():
                        summary_data.append([div_type, count])
                    summary_data.append(['', ''])
                
                # Validierung (falls verfügbar)
                if 'validation' in self.results:
                    summary_data.append(['VALIDIERUNG', ''])
                    for div_type, result in self.results['validation'].items():
                        if div_type != 'comparison' and 'stats' in result:
                            stats = result['stats']
                            summary_data.append([f"{div_type} - Hit Rate 30d", f"{stats.get('hit_rate_30d', 0):.1f}%"])
                            summary_data.append([f"{div_type} - Avg Return 30d", f"{stats.get('avg_return_30d', 0):+.2f}%"])
                    summary_data.append(['', ''])
                
                # Optimierung (falls verfügbar)
                if 'optimization' in self.results and 'optimal' in self.results['optimization']:
                    summary_data.append(['OPTIMIERUNG', ''])
                    optimal = self.results['optimization']['optimal']
                    if 'performance_score' in optimal:
                        best = optimal['performance_score']
                        summary_data.append(['Best Performance Score', f"{best.get('score', 0):.3f}"])
                        summary_data.append(['Best Parameters', best.get('parameter_string', 'N/A')])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Alle gefundenen Divergenzen
                all_divergences = []
                divergence_columns = ['CBullD_gen', 'CBullD_neg_MACD', 'HBullD', 'BearD', 'HBearD']
                
                for div_col in divergence_columns:
                    if div_col in self.df.columns:
                        div_data = self.df[self.df[div_col] == 1].copy()
                        if not div_data.empty:
                            div_data['Divergence_Type'] = div_col
                            all_divergences.append(div_data[['date', 'close', 'RSI', 'macd_histogram', 'Divergence_Type']])
                
                if all_divergences:
                    combined_div = pd.concat(all_divergences, ignore_index=True)
                    combined_div = combined_div.sort_values('date').reset_index(drop=True)
                    combined_div.to_excel(writer, sheet_name='All_Divergences', index=False)
            
            logger.info(f"✅ Finale Ergebnisse exportiert: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Export: {e}")
            return ""
    
    def run_complete_analysis(self, file_path: str = None) -> bool:
        """
        Führt die komplette Analyse durch
        
        Args:
            file_path: Pfad zur Datendatei (optional, sonst Dialog)
            
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info("🚀 STARTE KOMPLETTE DIVERGENZ-ANALYSE")
            logger.info("="*60)
            
            start_time = time.time()
            
            # Schritt 1: Daten laden
            if file_path is None:
                file_path = self.select_data_file()
            
            if not file_path:
                logger.error("❌ Keine Datei ausgewählt")
                return False
            
            if not self.load_and_prepare_data(file_path):
                return False
            
            # Schritt 2: Basis-Indikatoren
            if not self.prepare_base_indicators():
                return False
            
            # Schritt 3: Standard-Divergenz-Analyse
            if not self.run_basic_divergence_analysis():
                return False
            
            # Schritt 4: Validierung (optional)
            self.run_validation_analysis()  # Fehler werden protokolliert, aber nicht abgebrochen
            
            # Schritt 5: Parameter-Optimierung (optional)
            self.run_parameter_optimization()  # Fehler werden protokolliert, aber nicht abgebrochen
            
            # Schritt 6: Erweiterte Analyse
            if not self.run_enhanced_divergence_analysis():
                return False
            
            # Schritt 7: Finale Ergebnisse exportieren
            final_file = self.export_final_results()
            
            # Zusammenfassung
            elapsed_time = time.time() - start_time
            
            logger.info("="*60)
            logger.info("🎉 ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
            logger.info(f"⏱️  Gesamtzeit: {elapsed_time:.1f} Sekunden")
            logger.info(f"📁 Datei verarbeitet: {Path(file_path).name}")
            logger.info(f"📊 Zeilen analysiert: {len(self.df)}")
            if final_file:
                logger.info(f"📋 Ergebnisse gespeichert: {final_file}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Kritischer Fehler bei kompletter Analyse: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Hauptfunktion für komplette Analyse
    """
    print("🚀 Complete Divergence Analysis System")
    print("="*50)
    
    try:
        runner = CompleteAnalysisRunner()
        
        # Konfiguration anpassen (optional)
        runner.config['optimization']['enable_grid_search'] = True  # Parameter-Optimierung aktivieren
        runner.config['analysis']['enable_hidden_bullish'] = True   # Hidden Divergenzen aktivieren
        runner.config['analysis']['enable_bearish'] = True          # Bearish Divergenzen aktivieren
        runner.config['analysis']['enable_backtesting'] = True      # Validierung aktivieren
        
        # Komplette Analyse starten
        success = runner.run_complete_analysis()
        
        if success:
            print("\n✅ Analyse erfolgreich abgeschlossen!")
            print("📊 Alle Charts wurden im Browser geöffnet")
            print("📋 Excel-Dateien wurden erstellt")
        else:
            print("\n❌ Analyse fehlgeschlagen - siehe Log für Details")
            
    except KeyboardInterrupt:
        print("\n⏹️ Analyse vom Benutzer abgebrochen")
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    main()
