"""
Sensitivity Parameter Optimizer - Systematische Parameter-Optimierung
Findet die optimalen Parameter für jede Divergenz-Art und Asset-Klasse
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Unsere Module
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from simple_enhanced_runner import add_hidden_bullish_divergences, add_bearish_divergences
from performance_analyzer import DivergencePerformanceAnalyzer

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Optimiert Parameter für Divergenz-Erkennung basierend auf Performance-Metriken
    """
    
    def __init__(self, df: pd.DataFrame, asset_name: str = "Unknown"):
        """
        Initialisierung mit Basis-Daten (ohne Divergenzen)
        
        Args:
            df: DataFrame mit OHLC-Daten und Basis-Indikatoren
            asset_name: Name des Assets für Reporting
        """
        self.df = df.copy()
        self.asset_name = asset_name
        
        # Parameter-Bereiche für Grid Search
        self.parameter_ranges = {
            'window': [3, 4, 5, 6, 7],
            'candle_tolerance': [0.05, 0.1, 0.15, 0.2, 0.3],
            'macd_tolerance': [1.0, 2.0, 3.0, 3.25, 4.0, 5.0, 7.5]
        }
        
        # Verschiedene Optimierungsziele
        self.optimization_targets = {
            'hit_rate_30d': {'description': 'Hit Rate 30 Tage', 'higher_better': True},
            'avg_return_30d': {'description': 'Durchschnittlicher Return 30 Tage', 'higher_better': True},
            'avg_risk_reward_30d': {'description': 'Risk-Reward Ratio', 'higher_better': True},
            'total_signals': {'description': 'Anzahl Signale', 'higher_better': True},
            'combined_score': {'description': 'Kombinierter Score', 'higher_better': True}
        }
        
        self.optimization_results = {}
        
        logger.info(f"Parameter Optimizer initialisiert für {asset_name} mit {len(self.df)} Datenpunkten")
    
    def set_parameter_ranges(self, ranges: Dict[str, List]) -> None:
        """
        Setzt benutzerdefinierte Parameter-Bereiche
        
        Args:
            ranges: Dictionary mit Parameter-Namen und Listen von Werten
        """
        self.parameter_ranges.update(ranges)
        logger.info(f"Parameter-Bereiche aktualisiert: {self.parameter_ranges}")
    
    def test_parameter_combination(self, window: int, candle_tol: float, 
                                 macd_tol: float) -> Dict:
        """
        Testet eine spezifische Parameter-Kombination
        
        Args:
            window: Fenster-Größe
            candle_tol: Candlestick-Toleranz
            macd_tol: MACD-Toleranz
            
        Returns:
            Dictionary mit Performance-Metriken
        """
        try:
            # Kopiere Basis-DataFrame
            test_df = self.df.copy()
            
            # Berechne alle Divergenzen mit diesen Parametern
            CBullDivg_analysis(test_df, window, candle_tol, macd_tol)
            add_hidden_bullish_divergences(test_df, window, candle_tol, macd_tol)
            add_bearish_divergences(test_df, window, candle_tol, macd_tol)
            
            # Performance-Analyse
            analyzer = DivergencePerformanceAnalyzer(test_df)
            performance_results = analyzer.run_complete_performance_analysis()
            
            # Sammle Metriken
            result = {
                'window': window,
                'candle_tolerance': candle_tol,
                'macd_tolerance': macd_tol,
                'parameter_string': f"W{window}_C{candle_tol:.2f}_M{macd_tol:.2f}"
            }
            
            # Aggregiere Performance über alle Divergenz-Typen
            total_signals = 0
            weighted_hit_rate = 0
            weighted_avg_return = 0
            weighted_risk_reward = 0
            
            for div_type, div_result in performance_results.items():
                if div_type != 'comparison' and 'stats' in div_result:
                    stats = div_result['stats']
                    signals = stats['total_signals']
                    total_signals += signals
                    
                    if signals > 0:
                        hit_rate = stats.get('hit_rate_30d', 0)
                        avg_return = stats.get('avg_return_30d', 0)
                        risk_reward = stats.get('avg_risk_reward_30d', 0)
                        
                        # Gewichtung nach Anzahl Signale
                        weighted_hit_rate += hit_rate * signals
                        weighted_avg_return += avg_return * signals
                        weighted_risk_reward += risk_reward * signals
            
            # Berechne gewichtete Durchschnitte
            if total_signals > 0:
                result['hit_rate_30d'] = weighted_hit_rate / total_signals
                result['avg_return_30d'] = weighted_avg_return / total_signals
                result['avg_risk_reward_30d'] = weighted_risk_reward / total_signals
            else:
                result['hit_rate_30d'] = 0
                result['avg_return_30d'] = 0
                result['avg_risk_reward_30d'] = 0
            
            result['total_signals'] = total_signals
            
            # Kombinierter Score (anpassbar)
            hit_rate_score = result['hit_rate_30d'] / 100  # 0-1
            return_score = max(0, result['avg_return_30d'] / 10)  # Normalisiert auf ~0-1
            signal_score = min(1, total_signals / 50)  # Normalisiert, max bei 50+ Signalen
            risk_reward_score = min(1, result['avg_risk_reward_30d'] / 3)  # Normalisiert
            
            result['combined_score'] = (
                hit_rate_score * 0.4 +      # 40% Gewichtung Hit Rate
                return_score * 0.3 +        # 30% Gewichtung Returns
                signal_score * 0.2 +        # 20% Gewichtung Signalanzahl
                risk_reward_score * 0.1     # 10% Gewichtung Risk-Reward
            )
            
            # Detaillierte Ergebnisse pro Divergenz-Typ
            for div_type, div_result in performance_results.items():
                if div_type != 'comparison' and 'stats' in div_result:
                    stats = div_result['stats']
                    prefix = div_type.lower()
                    result[f'{prefix}_signals'] = stats['total_signals']
                    result[f'{prefix}_hit_rate'] = stats.get('hit_rate_30d', 0)
                    result[f'{prefix}_return'] = stats.get('avg_return_30d', 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei Parameter-Test W{window}_C{candle_tol}_M{macd_tol}: {e}")
            return {
                'window': window,
                'candle_tolerance': candle_tol,
                'macd_tolerance': macd_tol,
                'parameter_string': f"W{window}_C{candle_tol:.2f}_M{macd_tol:.2f}",
                'hit_rate_30d': 0,
                'avg_return_30d': 0,
                'avg_risk_reward_30d': 0,
                'total_signals': 0,
                'combined_score': 0,
                'error': str(e)
            }
    
    def run_grid_search(self, max_workers: int = 1) -> pd.DataFrame:
        """
        Führt Grid Search über alle Parameter-Kombinationen durch
        
        Args:
            max_workers: Anzahl paralleler Prozesse (1 für sequenziell)
            
        Returns:
            DataFrame mit allen getesteten Kombinationen
        """
        logger.info("🔍 Starte Parameter-Grid-Search...")
        
        # Generiere alle Kombinationen
        combinations = list(product(
            self.parameter_ranges['window'],
            self.parameter_ranges['candle_tolerance'],
            self.parameter_ranges['macd_tolerance']
        ))
        
        total_combinations = len(combinations)
        logger.info(f"Teste {total_combinations} Parameter-Kombinationen...")
        
        results = []
        
        if max_workers > 1:
            # Parallele Verarbeitung (experimentell)
            logger.warning("Parallele Verarbeitung kann bei Plotly-Charts Probleme verursachen")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {
                    executor.submit(self.test_parameter_combination, w, c, m): (w, c, m)
                    for w, c, m in combinations
                }
                
                completed = 0
                for future in as_completed(future_to_params):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 5 == 0:
                        logger.info(f"Fortschritt: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
        else:
            # Sequenzielle Verarbeitung (empfohlen)
            for i, (window, candle_tol, macd_tol) in enumerate(combinations):
                result = self.test_parameter_combination(window, candle_tol, macd_tol)
                results.append(result)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Fortschritt: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        
        # Erstelle DataFrame
        results_df = pd.DataFrame(results)
        
        # Sortiere nach Combined Score
        results_df = results_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        
        logger.info("✅ Grid Search abgeschlossen")
        logger.info(f"Beste Parameter: {results_df.iloc[0]['parameter_string']}")
        logger.info(f"Bester Combined Score: {results_df.iloc[0]['combined_score']:.3f}")
        
        return results_df
    
    def find_optimal_parameters_by_target(self, results_df: pd.DataFrame) -> Dict:
        """
        Findet optimale Parameter für verschiedene Optimierungsziele
        
        Args:
            results_df: DataFrame mit Grid Search Ergebnissen
            
        Returns:
            Dictionary mit optimalen Parametern pro Ziel
        """
        optimal_params = {}
        
        for target, info in self.optimization_targets.items():
            if target in results_df.columns:
                if info['higher_better']:
                    best_idx = results_df[target].idxmax()
                else:
                    best_idx = results_df[target].idxmin()
                
                best_row = results_df.iloc[best_idx]
                
                optimal_params[target] = {
                    'window': int(best_row['window']),
                    'candle_tolerance': float(best_row['candle_tolerance']),
                    'macd_tolerance': float(best_row['macd_tolerance']),
                    'score': float(best_row[target]),
                    'parameter_string': best_row['parameter_string'],
                    'description': info['description']
                }
        
        return optimal_params
    
    def analyze_parameter_sensitivity(self, results_df: pd.DataFrame) -> Dict:
        """
        Analysiert die Sensitivität der Parameter
        
        Args:
            results_df: DataFrame mit Grid Search Ergebnissen
            
        Returns:
            Dictionary mit Sensitivitäts-Analysen
        """
        sensitivity = {}
        
        for param in ['window', 'candle_tolerance', 'macd_tolerance']:
            # Gruppiere nach Parameter und berechne Statistiken
            param_stats = results_df.groupby(param).agg({
                'combined_score': ['mean', 'std', 'max', 'min', 'count'],
                'hit_rate_30d': ['mean', 'std'],
                'avg_return_30d': ['mean', 'std'],
                'total_signals': ['mean', 'std']
            }).round(4)
            
            # Flatten column names
            param_stats.columns = ['_'.join(col).strip() for col in param_stats.columns]
            
            # Sensitivitäts-Score (höhere Standardabweichung = sensibler)
            sensitivity_score = param_stats['combined_score_std'].mean()
            
            # Finde optimalen Wert für diesen Parameter
            best_value_idx = param_stats['combined_score_mean'].idxmax()
            
            sensitivity[param] = {
                'stats': param_stats,
                'sensitivity_score': sensitivity_score,
                'optimal_value': best_value_idx,
                'stability': 1 / (1 + sensitivity_score)  # Stabilität (0-1)
            }
        
        return sensitivity
    
    def create_optimization_dashboard(self, results_df: pd.DataFrame, 
                                    sensitivity: Dict, optimal_params: Dict) -> go.Figure:
        """
        Erstellt umfassendes Optimierungs-Dashboard
        
        Args:
            results_df: Grid Search Ergebnisse
            sensitivity: Sensitivitäts-Analyse
            optimal_params: Optimale Parameter
            
        Returns:
            Plotly Figure mit Optimierungs-Visualisierungen
        """
        # 3x3 Dashboard Layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Combined Score vs Window', 'Combined Score vs Candle Tolerance', 'Combined Score vs MACD Tolerance',
                'Hit Rate Heatmap', 'Return Heatmap', 'Parameter Correlation',
                'Top 10 Kombinationen', 'Sensitivitäts-Analyse', 'Optimization Target Comparison'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Row 1: Parameter vs Combined Score
        for i, param in enumerate(['window', 'candle_tolerance', 'macd_tolerance'], 1):
            param_grouped = results_df.groupby(param)['combined_score'].agg(['mean', 'std']).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=param_grouped[param],
                    y=param_grouped['mean'],
                    error_y=dict(type='data', array=param_grouped['std']),
                    mode='markers+lines',
                    name=f'{param} Sensitivity',
                    marker=dict(size=8, color=f'rgb({50+i*70}, {100+i*50}, {150+i*30})')
                ),
                row=1, col=i
            )
        
        # Row 2: Heatmaps für verschiedene Metriken
        # Hit Rate Heatmap (Window vs Candle Tolerance)
        pivot_hit_rate = results_df.pivot_table(
            values='hit_rate_30d', 
            index='window', 
            columns='candle_tolerance', 
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_hit_rate.values,
                x=pivot_hit_rate.columns,
                y=pivot_hit_rate.index,
                colorscale='RdYlGn',
                name='Hit Rate',
                hovertemplate='Window: %{y}<br>Candle Tol: %{x}<br>Hit Rate: %{z:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Return Heatmap
        pivot_return = results_df.pivot_table(
            values='avg_return_30d',
            index='window',
            columns='candle_tolerance', 
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_return.values,
                x=pivot_return.columns,
                y=pivot_return.index,
                colorscale='RdBu',
                name='Avg Return',
                hovertemplate='Window: %{y}<br>Candle Tol: %{x}<br>Return: %{z:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Parameter Correlation
        correlation_data = results_df[['window', 'candle_tolerance', 'macd_tolerance', 'combined_score']].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_data.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                name='Correlation'
            ),
            row=2, col=3
        )
        
        # Row 3: Rankings und Vergleiche
        # Top 10 Kombinationen
        top_10 = results_df.head(10)
        fig.add_trace(
            go.Bar(
                x=list(range(1, 11)),
                y=top_10['combined_score'],
                text=top_10['parameter_string'],
                textposition='outside',
                name='Top 10 Scores',
                marker=dict(color='lightblue')
            ),
            row=3, col=1
        )
        
        # Sensitivitäts-Vergleich
        sens_params = list(sensitivity.keys())
        sens_scores = [sensitivity[param]['sensitivity_score'] for param in sens_params]
        
        fig.add_trace(
            go.Bar(
                x=sens_params,
                y=sens_scores,
                name='Parameter Sensitivity',
                marker=dict(color=['red', 'orange', 'yellow'])
            ),
            row=3, col=2
        )
        
        # Optimization Target Comparison
        targets = list(optimal_params.keys())
        target_scores = [optimal_params[target]['score'] for target in targets]
        
        fig.add_trace(
            go.Bar(
                x=targets,
                y=target_scores,
                name='Optimization Targets',
                marker=dict(color='green'),
                text=[f"{score:.2f}" for score in target_scores],
                textposition='outside'
            ),
            row=3, col=3
        )
        
        # Layout Updates
        fig.update_layout(
            title_text=f"Parameter Optimization Dashboard - {self.asset_name}",
            template="plotly_white",
            height=1200,
            showlegend=False
        )
        
        # Achsen-Updates
        for i in range(1, 4):
            fig.update_xaxes(title_text="Parameter Value", row=1, col=i)
            fig.update_yaxes(title_text="Combined Score", row=1, col=i)
        
        fig.update_xaxes(title_text="Rank", row=3, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=1)
        fig.update_xaxes(title_text="Parameter", row=3, col=2)
        fig.update_yaxes(title_text="Sensitivity", row=3, col=2)
        
        return fig
    
    def run_complete_optimization(self, max_workers: int = 1) -> Dict:
        """
        Führt komplette Parameter-Optimierung durch
        
        Args:
            max_workers: Anzahl paralleler Prozesse
            
        Returns:
            Dictionary mit allen Optimierungsergebnissen
        """
        logger.info(f"🚀 Starte komplette Parameter-Optimierung für {self.asset_name}...")
        
        start_time = time.time()
        
        # Schritt 1: Grid Search
        results_df = self.run_grid_search(max_workers)
        
        # Schritt 2: Finde optimale Parameter für verschiedene Ziele
        optimal_params = self.find_optimal_parameters_by_target(results_df)
        
        # Schritt 3: Sensitivitäts-Analyse
        sensitivity = self.analyze_parameter_sensitivity(results_df)
        
        # Schritt 4: Dashboard erstellen
        dashboard = self.create_optimization_dashboard(results_df, sensitivity, optimal_params)
        
        elapsed_time = time.time() - start_time
        
        # Sammle alle Ergebnisse
        optimization_results = {
            'asset_name': self.asset_name,
            'grid_search_results': results_df,
            'optimal_parameters': optimal_params,
            'sensitivity_analysis': sensitivity,
            'dashboard': dashboard,
            'execution_time': elapsed_time,
            'total_combinations_tested': len(results_df)
        }
        
        self.optimization_results = optimization_results
        
        logger.info(f"✅ Parameter-Optimierung abgeschlossen in {elapsed_time:.1f} Sekunden")
        
        return optimization_results
    
    def export_optimization_results(self, filename: str = None) -> str:
        """
        Exportiert Optimierungsergebnisse nach Excel
        
        Args:
            filename: Zieldatei (optional)
            
        Returns:
            Pfad zur erstellten Datei
        """
        if not self.optimization_results:
            logger.error("Keine Optimierungsergebnisse vorhanden")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"parameter_optimization_{self.asset_name}_{timestamp}.xlsx"
        
        logger.info(f"📊 Exportiere Optimierungsergebnisse: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Alle Grid Search Ergebnisse
            self.optimization_results['grid_search_results'].to_excel(
                writer, sheet_name='Grid_Search_Results', index=False
            )
            
            # Top 50 Kombinationen
            top_50 = self.optimization_results['grid_search_results'].head(50)
            top_50.to_excel(writer, sheet_name='Top_50_Parameters', index=False)
            
            # Optimale Parameter pro Ziel
            optimal_df = pd.DataFrame(self.optimization_results['optimal_parameters']).T
            optimal_df.to_excel(writer, sheet_name='Optimal_Parameters')
            
            # Sensitivitäts-Analyse
            for param, data in self.optimization_results['sensitivity_analysis'].items():
                sheet_name = f'Sensitivity_{param}'
                data['stats'].to_excel(writer, sheet_name=sheet_name)
            
            # Zusammenfassung
            summary_data = {
                'Metric': [
                    'Asset Name',
                    'Total Combinations Tested',
                    'Best Combined Score',
                    'Best Hit Rate 30d',
                    'Best Avg Return 30d',
                    'Execution Time (seconds)'
                ],
                'Value': [
                    self.optimization_results['asset_name'],
                    self.optimization_results['total_combinations_tested'],
                    f"{self.optimization_results['grid_search_results']['combined_score'].max():.3f}",
                    f"{self.optimization_results['grid_search_results']['hit_rate_30d'].max():.1f}%",
                    f"{self.optimization_results['grid_search_results']['avg_return_30d'].max():.2f}%",
                    f"{self.optimization_results['execution_time']:.1f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"✅ Optimierungsergebnisse exportiert: {filename}")
        return filename
    
    def print_optimization_summary(self):
        """
        Druckt Optimierungs-Zusammenfassung in die Konsole
        """
        if not self.optimization_results:
            logger.error("Keine Optimierungsergebnisse vorhanden")
            return
        
        results = self.optimization_results
        
        print(f"\n" + "="*80)
        print(f"🎯 PARAMETER-OPTIMIERUNG ZUSAMMENFASSUNG - {self.asset_name}")
        print("="*80)
        
        # Allgemeine Statistiken
        print(f"\n📊 ÜBERSICHT:")
        print(f"Getestete Kombinationen:     {results['total_combinations_tested']}")
        print(f"Ausführungszeit:             {results['execution_time']:.1f} Sekunden")
        print(f"Beste Combined Score:        {results['grid_search_results']['combined_score'].max():.3f}")
        
        # Top 5 Parameter-Kombinationen
        print(f"\n🏆 TOP 5 PARAMETER-KOMBINATIONEN:")
        print("-" * 60)
        top_5 = results['grid_search_results'].head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['parameter_string']} - Score: {row['combined_score']:.3f}")
            print(f"   Hit Rate: {row['hit_rate_30d']:.1f}%, Return: {row['avg_return_30d']:+.2f}%, Signale: {row['total_signals']}")
        
        # Optimale Parameter pro Ziel
        print(f"\n🎯 OPTIMALE PARAMETER PRO ZIEL:")
        print("-" * 60)
        for target, params in results['optimal_parameters'].items():
            print(f"{params['description']:<25}: {params['parameter_string']} (Score: {params['score']:.3f})")
        
        # Parameter-Sensitivität
        print(f"\n📈 PARAMETER-SENSITIVITÄT:")
        print("-" * 60)
        for param, data in results['sensitivity_analysis'].items():
            print(f"{param.upper():<15}: Sensitivität: {data['sensitivity_score']:.3f}, "
                  f"Optimal: {data['optimal_value']}, Stabilität: {data['stability']:.3f}")
        
        print("\n" + "="*80)


def main():
    """
    Test-Funktion für Parameter Optimizer
    """
    print("🎯 Parameter Sensitivity Optimizer")
    print("Verwendung:")
    print("optimizer = ParameterOptimizer(df, 'Bitcoin')")
    print("results = optimizer.run_complete_optimization()")
    print("optimizer.print_optimization_summary()")
    print("results['dashboard'].show()")

if __name__ == "__main__":
    main()