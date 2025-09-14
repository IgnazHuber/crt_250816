"""
Parameter Sensitivitätsanalyse für CBullDivg_Analysis
Analysiert die optimalen Werte für Candle_Tol und MACD_tol Parameter
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import time
import tkinter as tk
from tkinter import filedialog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product

# Eigene Module
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min
from CBullDivg_Analysis_vectorized import CBullDivg_analysis

# Output-Ordner
PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "parameter_analysis"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterSensitivityAnalyzer:
    """
    Analysiert die Sensitivität der CBullDivg Parameter
    """
    
    def __init__(self):
        self.df = None
        self.results = []
        
        # Parameter-Bereiche für die Analyse
        self.candle_tol_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        self.macd_tol_range = [1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0]
        self.window = 5  # Fest, da dieser Parameter weniger kritisch ist
        
        logger.info("🔍 Parameter Sensitivitätsanalyzer initialisiert")
    
    def load_and_prepare_data(self, file_path):
        """Lädt und bereitet die Daten vor"""
        try:
            logger.info(f"📊 Lade Daten: {file_path}")
            
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, low_memory=False)
            else:
                self.df = pd.read_parquet(file_path)
            
            # Validierung
            required = ['date', 'open', 'high', 'low', 'close']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Fehlende Spalten: {missing}")
            
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ {len(self.df)} Zeilen geladen")
            
            # Basis-Indikatoren berechnen (nur einmal)
            logger.info("🔧 Berechne Basis-Indikatoren...")
            Initialize_RSI_EMA_MACD(self.df)
            Local_Max_Min(self.df)
            logger.info("✅ Basis-Indikatoren berechnet")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden: {e}")
            return False
    
    def analyze_parameter_combination(self, candle_tol, macd_tol):
        """
        Analysiert eine spezifische Parameter-Kombination
        """
        try:
            # Kopie der Daten für diese Analyse
            df_copy = self.df.copy()
            
            # Divergenz-Analyse mit den gegebenen Parametern
            CBullDivg_analysis(df_copy, self.window, candle_tol, macd_tol)
            
            # Ergebnisse sammeln
            classic_count = (df_copy['CBullD_gen'] == 1).sum() if 'CBullD_gen' in df_copy.columns else 0
            neg_macd_count = (df_copy['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in df_copy.columns else 0
            total_signals = classic_count + neg_macd_count
            
            # Signal-Dichte (Signale pro 100 Tage)
            signal_density = (total_signals / len(df_copy)) * 100
            
            # Durchschnittlicher Gap zwischen Signalen
            avg_gap_classic = 0
            avg_gap_neg = 0
            
            if classic_count > 0:
                classic_gaps = df_copy[df_copy['CBullD_gen'] == 1]['CBullD_Date_Gap_gen']
                avg_gap_classic = classic_gaps.mean() if len(classic_gaps) > 0 else 0
            
            if neg_macd_count > 0:
                neg_gaps = df_copy[df_copy['CBullD_neg_MACD'] == 1]['CBullD_Date_Gap_neg_MACD']
                avg_gap_neg = neg_gaps.mean() if len(neg_gaps) > 0 else 0
            
            return {
                'candle_tol': candle_tol,
                'macd_tol': macd_tol,
                'classic_count': classic_count,
                'neg_macd_count': neg_macd_count,
                'total_signals': total_signals,
                'signal_density': signal_density,
                'avg_gap_classic': avg_gap_classic,
                'avg_gap_neg': avg_gap_neg,
                'classic_pct': (classic_count / len(df_copy)) * 100,
                'neg_macd_pct': (neg_macd_count / len(df_copy)) * 100
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Parameter-Kombination ({candle_tol}, {macd_tol}): {e}")
            return None
    
    def run_full_analysis(self):
        """
        Führt vollständige Parameteranalyse durch
        """
        logger.info("🚀 Starte vollständige Parameteranalyse...")
        
        total_combinations = len(self.candle_tol_range) * len(self.macd_tol_range)
        logger.info(f"📊 Analysiere {total_combinations} Parameter-Kombinationen...")
        
        start_time = time.time()
        
        # Alle Kombinationen durchgehen
        for i, (candle_tol, macd_tol) in enumerate(product(self.candle_tol_range, self.macd_tol_range), 1):
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                progress = (i / total_combinations) * 100
                logger.info(f"⏳ Fortschritt: {progress:.1f}% ({i}/{total_combinations}) - {elapsed:.1f}s")
            
            result = self.analyze_parameter_combination(candle_tol, macd_tol)
            if result:
                self.results.append(result)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Analyse abgeschlossen in {elapsed_time:.1f} Sekunden")
        logger.info(f"📊 {len(self.results)} erfolgreiche Kombinationen analysiert")
        
        return True
    
    def create_sensitivity_analysis_chart(self):
        """
        Erstellt umfassende Visualisierung der Sensitivitätsanalyse
        """
        if not self.results:
            logger.error("❌ Keine Ergebnisse für Visualisierung")
            return None
        
        logger.info("📊 Erstelle Sensitivitäts-Charts...")
        
        results_df = pd.DataFrame(self.results)
        
        # 2x3 Grid für verschiedene Metriken
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Total Signals Heatmap', 
                'Signal Density (%)', 
                'Classic Bullish Count',
                'Negative MACD Count',
                'Average Gap (Classic)',
                'Parameter Optimization Score'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Pivot-Tabellen für Heatmaps erstellen
        def create_pivot(metric):
            return results_df.pivot(index='macd_tol', columns='candle_tol', values=metric)
        
        # 1. Total Signals Heatmap
        total_signals_pivot = create_pivot('total_signals')
        fig.add_trace(
            go.Heatmap(
                z=total_signals_pivot.values,
                x=total_signals_pivot.columns,
                y=total_signals_pivot.index,
                colorscale='Viridis',
                name='Total Signals',
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. Signal Density
        density_pivot = create_pivot('signal_density')
        fig.add_trace(
            go.Heatmap(
                z=density_pivot.values,
                x=density_pivot.columns,
                y=density_pivot.index,
                colorscale='Plasma',
                name='Signal Density'
            ),
            row=1, col=2
        )
        
        # 3. Classic Bullish Count
        classic_pivot = create_pivot('classic_count')
        fig.add_trace(
            go.Heatmap(
                z=classic_pivot.values,
                x=classic_pivot.columns,
                y=classic_pivot.index,
                colorscale='Blues',
                name='Classic Count'
            ),
            row=1, col=3
        )
        
        # 4. Negative MACD Count
        neg_macd_pivot = create_pivot('neg_macd_count')
        fig.add_trace(
            go.Heatmap(
                z=neg_macd_pivot.values,
                x=neg_macd_pivot.columns,
                y=neg_macd_pivot.index,
                colorscale='Reds',
                name='Neg MACD Count'
            ),
            row=2, col=1
        )
        
        # 5. Average Gap (Classic)
        gap_classic_pivot = create_pivot('avg_gap_classic')
        fig.add_trace(
            go.Heatmap(
                z=gap_classic_pivot.values,
                x=gap_classic_pivot.columns,
                y=gap_classic_pivot.index,
                colorscale='Oranges',
                name='Avg Gap Classic'
            ),
            row=2, col=2
        )
        
        # 6. Optimization Score (Balance zwischen Signalanzahl und Qualität)
        # Score = Total Signals * (1 - |avg_gap - target_gap|/target_gap)
        target_gap = 15  # Idealer Gap zwischen Signalen
        results_df['optimization_score'] = results_df.apply(
            lambda row: row['total_signals'] * max(0, 1 - abs(row['avg_gap_classic'] - target_gap) / target_gap)
            if row['avg_gap_classic'] > 0 else row['total_signals'] * 0.5, axis=1
        )
        
        opt_score_pivot = results_df.pivot(index='macd_tol', columns='candle_tol', values='optimization_score')
        fig.add_trace(
            go.Heatmap(
                z=opt_score_pivot.values,
                x=opt_score_pivot.columns,
                y=opt_score_pivot.index,
                colorscale='RdYlGn',
                name='Optimization Score'
            ),
            row=2, col=3
        )
        
        # Layout
        fig.update_layout(
            title=dict(
                text="Parameter Sensitivitätsanalyse - CBullDivg_Analysis",
                x=0.5,
                font_size=16
            ),
            height=800,
            showlegend=False
        )
        
        # Achsen-Labels
        for i in range(1, 4):
            fig.update_xaxes(title_text="Candle Tolerance", row=1, col=i)
            fig.update_xaxes(title_text="Candle Tolerance", row=2, col=i)
            fig.update_yaxes(title_text="MACD Tolerance", row=1, col=i)
            fig.update_yaxes(title_text="MACD Tolerance", row=2, col=i)
        
        return fig, results_df
    
    def find_optimal_parameters(self):
        """
        Findet optimale Parameter basierend auf verschiedenen Kriterien
        """
        if not self.results:
            return None
        
        results_df = pd.DataFrame(self.results)
        
        # Verschiedene Optimierungskriterien
        optimizations = {}
        
        # 1. Maximale Signalanzahl
        max_signals = results_df.loc[results_df['total_signals'].idxmax()]
        optimizations['max_signals'] = {
            'criteria': 'Maximum Total Signals',
            'candle_tol': max_signals['candle_tol'],
            'macd_tol': max_signals['macd_tol'],
            'total_signals': max_signals['total_signals'],
            'signal_density': max_signals['signal_density']
        }
        
        # 2. Optimale Signal-Dichte (nicht zu viele, nicht zu wenige)
        target_density = 5.0  # 5% Signal-Dichte als Ziel
        results_df['density_score'] = 1 / (1 + abs(results_df['signal_density'] - target_density))
        optimal_density = results_df.loc[results_df['density_score'].idxmax()]
        optimizations['optimal_density'] = {
            'criteria': 'Optimal Signal Density (~5%)',
            'candle_tol': optimal_density['candle_tol'],
            'macd_tol': optimal_density['macd_tol'],
            'total_signals': optimal_density['total_signals'],
            'signal_density': optimal_density['signal_density']
        }
        
        # 3. Ausgewogenes Verhältnis Classic/Negative MACD
        results_df['balance_score'] = results_df.apply(
            lambda row: min(row['classic_count'], row['neg_macd_count']) / max(row['classic_count'], row['neg_macd_count'], 1),
            axis=1
        )
        balanced = results_df.loc[results_df['balance_score'].idxmax()]
        optimizations['balanced'] = {
            'criteria': 'Balanced Classic/Negative MACD',
            'candle_tol': balanced['candle_tol'],
            'macd_tol': balanced['macd_tol'],
            'total_signals': balanced['total_signals'],
            'signal_density': balanced['signal_density'],
            'balance_score': balanced['balance_score']
        }
        
        # 4. Aktueller Standard (Baseline)
        current = results_df[(results_df['candle_tol'] == 0.1) & (results_df['macd_tol'] == 3.25)]
        if not current.empty:
            current = current.iloc[0]
            optimizations['current'] = {
                'criteria': 'Current Standard (0.1, 3.25)',
                'candle_tol': current['candle_tol'],
                'macd_tol': current['macd_tol'],
                'total_signals': current['total_signals'],
                'signal_density': current['signal_density']
            }
        
        return optimizations
    
    def export_results(self, results_df, optimizations):
        """
        Exportiert Ergebnisse nach Excel
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_file = OUTPUTS_DIR / f"parameter_sensitivity_analysis_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Alle Ergebnisse
                results_df.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Optimierungen
                opt_df = pd.DataFrame(optimizations).T
                opt_df.to_excel(writer, sheet_name='Optimal_Parameters', index=True)
                
                # Pivot-Tabellen für Heatmaps
                total_signals_pivot = results_df.pivot(index='macd_tol', columns='candle_tol', values='total_signals')
                total_signals_pivot.to_excel(writer, sheet_name='Total_Signals_Matrix')
                
                density_pivot = results_df.pivot(index='macd_tol', columns='candle_tol', values='signal_density')
                density_pivot.to_excel(writer, sheet_name='Signal_Density_Matrix')
            
            logger.info(f"📋 Ergebnisse exportiert: {excel_file}")
            return excel_file
            
        except Exception as e:
            logger.error(f"❌ Export fehlgeschlagen: {e}")
            return None

def main():
    """
    Hauptfunktion für Sensitivitätsanalyse
    """
    print("🔍 Parameter Sensitivitätsanalyse für CBullDivg_Analysis")
    print("="*60)
    
    # Datei auswählen
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Wähle Datendatei für Parameteranalyse",
        filetypes=[("CSV Dateien", "*.csv"), ("Parquet Dateien", "*.parquet")]
    )
    root.destroy()
    
    if not file_path:
        print("❌ Keine Datei ausgewählt")
        return
    
    # Analyzer initialisieren
    analyzer = ParameterSensitivityAnalyzer()
    
    # Daten laden
    if not analyzer.load_and_prepare_data(file_path):
        return
    
    # Vollständige Analyse
    if not analyzer.run_full_analysis():
        return
    
    # Optimale Parameter finden
    optimizations = analyzer.find_optimal_parameters()
    
    if optimizations:
        print("\n" + "="*60)
        print("🎯 OPTIMALE PARAMETER-EMPFEHLUNGEN:")
        print("="*60)
        
        for name, opt in optimizations.items():
            print(f"\n{opt['criteria']}:")
            print(f"  Candle_Tol: {opt['candle_tol']}")
            print(f"  MACD_Tol: {opt['macd_tol']}")
            print(f"  Total Signals: {opt['total_signals']}")
            print(f"  Signal Density: {opt['signal_density']:.2f}%")
    
    # Visualisierung
    fig, results_df = analyzer.create_sensitivity_analysis_chart()
    if fig:
        fig.show()
    
    # Export
    excel_file = analyzer.export_results(results_df, optimizations)
    
    print(f"\n✅ Sensitivitätsanalyse abgeschlossen!")
    print(f"📊 Chart im Browser geöffnet")
    if excel_file:
        print(f"📋 Ergebnisse exportiert: {excel_file}")

if __name__ == "__main__":
    main()