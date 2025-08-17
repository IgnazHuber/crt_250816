"""
Sensitivity Analyzer - Parameter-Optimierung für Divergenz-Erkennung
Findet die optimalen Parameter für maximale Performance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Eigene Module
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from divergence_validator import DivergenceValidator

logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    Analysiert die Sensitivität der Divergenz-Parameter und findet optimale Einstellungen
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialisierung mit vorbereiteten Chart-Daten
        
        Args:
            df: DataFrame mit OHLC-Daten und berechneten Basis-Indikatoren (RSI, MACD, etc.)
        """
        self.df = df.copy()
        self.base_df = df.copy()  # Backup für Reset
        
        # Parameter-Bereiche für Sensitivitätsanalyse
        self.parameter_ranges = {
            'window': [3, 4, 5, 6, 7, 8],
            'candle_tolerance': [0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
            'macd_tolerance': [1.0, 2.0, 3.0, 3.25, 4.0, 5.0, 7.5, 10.0]
        }
        
        self.results = {}
        self.optimization_results = {}
        
        logger.info(f"Sensitivity Analyzer initialisiert mit {len(self.df)} Datenpunkten")
    
    def set_parameter_ranges(self, ranges: Dict[str, List]) -> None:
        """
        Setzt benutzerdefinierte Parameter-Bereiche
        
        Args:
            ranges: Dictionary mit Parameter-Namen und Listen von Werten
        """
        self.parameter_ranges.update(ranges)
        logger.info(f"Parameter-Bereiche aktualisiert: {self.parameter_ranges}")
    
    def run_single_parameter_combination(self, window: int, candle_tol: float, 
                                       macd_tol: float) -> Dict:
        """
        Führt Divergenz-Analyse mit einem spezifischen Parameter-Set durch
        
        Args:
            window: Fenster-Größe für Extrema-Suche
            candle_tol: Toleranz für Candlestick-Preise (%)
            macd_tol: Toleranz für MACD-Werte (%)
            
        Returns:
            Dictionary mit Ergebnissen dieser Parameter-Kombination
        """
        try:
            # Kopiere Basis-DataFrame
            test_df = self.base_df.copy()
            
            # Führe Divergenz-Analyse durch
            CBullDivg_analysis(test_df, window, candle_tol, macd_tol)
            
            # Zähle gefundene Divergenzen
            classic_count = (test_df['CBullD_gen'] == 1).sum() if 'CBullD_gen' in test_df.columns else 0
            neg_macd_count = (test_df['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in test_df.columns else 0
            
            # Schnelle Performance-Bewertung (vereinfacht)
            performance_score = 0
            hit_rate = 0
            avg_return = 0
            
            if classic_count > 0:
                # Vereinfachte Performance-Berechnung
                validator = DivergenceValidator(test_df)
                results = validator.run_full_validation()
                
                if 'classic_bullish' in results:
                    hit_rate = results['classic_bullish']['stats']['hit_rate_30d']
                    avg_return = results['classic_bullish']['stats']['avg_return_30d']
                    performance_score = hit_rate * (1 + avg_return/100)  # Kombinierter Score
            
            return {
                'window': window,
                'candle_tolerance': candle_tol,
                'macd_tolerance': macd_tol,
                'classic_count': classic_count,
                'neg_macd_count': neg_macd_count,
                'total_signals': classic_count + neg_macd_count,
                'hit_rate_30d': hit_rate,
                'avg_return_30d': avg_return,
                'performance_score': performance_score,
                'parameter_string': f"W{window}_C{candle_tol:.2f}_M{macd_tol:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Parameter-Kombination W{window}_C{candle_tol}_M{macd_tol}: {e}")
            return {
                'window': window,
                'candle_tolerance': candle_tol,
                'macd_tolerance': macd_tol,
                'classic_count': 0,
                'neg_macd_count': 0,
                'total_signals': 0,
                'hit_rate_30d': 0,
                'avg_return_30d': 0,
                'performance_score': 0,
                'parameter_string': f"W{window}_C{candle_tol:.2f}_M{macd_tol:.2f}",
                'error': str(e)
            }
    
    def run_grid_search(self, max_workers: int = 4) -> pd.DataFrame:
        """
        Führt Grid Search über alle Parameter-Kombinationen durch
        
        Args:
            max_workers: Anzahl paralleler Prozesse
            
        Returns:
            DataFrame mit allen getesteten Parameter-Kombinationen und Ergebnissen
        """
        logger.info("🔍 Starte Grid Search für Parameter-Optimierung...")
        
        # Generiere alle Parameter-Kombinationen
        combinations = list(product(
            self.parameter_ranges['window'],
            self.parameter_ranges['candle_tolerance'],
            self.parameter_ranges['macd_tolerance']
        ))
        
        total_combinations = len(combinations)
        logger.info(f"Teste {total_combinations} Parameter-Kombinationen...")
        
        results = []
        
        # Parallele Verarbeitung für bessere Performance
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Starte alle Jobs
                future_to_params = {
                    executor.submit(self.run_single_parameter_combination, w, c, m): (w, c, m)
                    for w, c, m in combinations
                }
                
                # Sammle Ergebnisse
                completed = 0
                for future in as_completed(future_to_params):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        logger.info(f"Fortschritt: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
        else:
            # Sequenzielle Verarbeitung
            for i, (window, candle_tol, macd_tol) in enumerate(combinations):
                result = self.run_single_parameter_combination(window, candle_tol, macd_tol)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Fortschritt: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        
        # Erstelle DataFrame mit Ergebnissen
        results_df = pd.DataFrame(results)
        
        # Sortiere nach Performance Score
        results_df = results_df.sort_values('performance_score', ascending=False).reset_index(drop=True)
        
        self.results = results_df
        
        logger.info("✅ Grid Search abgeschlossen")
        logger.info(f"Beste Parameter: {results_df.iloc[0]['parameter_string']}")
        logger.info(f"Beste Performance: {results_df.iloc[0]['performance_score']:.3f}")
        
        return results_df
    
    def analyze_parameter_sensitivity(self) -> Dict:
        """
        Analysiert die Sensitivität einzelner Parameter
        
        Returns:
            Dictionary mit Sensitivitäts-Analysen
        """
        if self.results.empty:
            logger.error("Keine Ergebnisse vorhanden. Führe erst run_grid_search() aus.")
            return {}
        
        logger.info("📊 Analysiere Parameter-Sensitivität...")
        
        sensitivity = {}
        
        # Analyse für jeden Parameter
        for param in ['window', 'candle_tolerance', 'macd_tolerance']:
            param_analysis = self.results.groupby(param).agg({
                'performance_score': ['mean', 'std', 'max', 'min'],
                'total_signals': ['mean', 'std'],
                'hit_rate_30d': ['mean', 'std'],
                'avg_return_30d': ['mean', 'std']
            }).round(3)
            
            # Flatten column names
            param_analysis.columns = ['_'.join(col).strip() for col in param_analysis.columns]
            
            # Berechne Sensitivitäts-Score (Standardabweichung der Performance)
            sensitivity_score = param_analysis['performance_score_std'].mean()
            
            sensitivity[param] = {
                'analysis': param_analysis,
                'sensitivity_score': sensitivity_score,
                'optimal_value': self.results.iloc[0][param]  # Bester