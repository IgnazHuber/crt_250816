"""
Divergence Validator - Backtesting und Performance-Analyse
Analysiert die historische Performance von gefundenen Divergenzen
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

class DivergenceValidator:
    """
    Klasse zur Validierung und Bewertung von Divergenz-Signalen
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialisierung mit Chart-Daten
        
        Args:
            df: DataFrame mit OHLC-Daten und berechneten Divergenzen
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        self.validation_results = {}
        self.performance_stats = {}
        
        logger.info(f"Validator initialisiert mit {len(self.df)} Datenpunkten")
    
    def extract_divergence_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Extrahiert alle Divergenz-Signale aus dem DataFrame
        
        Returns:
            Dictionary mit Divergenz-Typ als Key und DataFrame als Value
        """
        signals = {}
        
        # Classic Bullish Divergenzen
        if 'CBullD_gen' in self.df.columns:
            classic_signals = self.df[self.df['CBullD_gen'] == 1].copy()
            if not classic_signals.empty:
                signals['classic_bullish'] = classic_signals
                logger.info(f"Gefunden: {len(classic_signals)} Classic Bullish Divergenzen")
        
        # Negative MACD Divergenzen
        if 'CBullD_neg_MACD' in self.df.columns:
            neg_macd_signals = self.df[self.df['CBullD_neg_MACD'] == 1].copy()
            if not neg_macd_signals.empty:
                signals['negative_macd'] = neg_macd_signals
                logger.info(f"Gefunden: {len(neg_macd_signals)} Negative MACD Divergenzen")
        
        return signals
    
    def calculate_future_returns(self, signal_date: pd.Timestamp, 
                               periods: List[int] = [7, 14, 30, 60]) -> Dict[str, float]:
        """
        Berechnet zukünftige Returns nach einem Divergenz-Signal
        
        Args:
            signal_date: Datum des Signals
            periods: Liste der Tage für Return-Berechnung
            
        Returns:
            Dictionary mit Perioden-Returns
        """
        signal_idx = self.df[self.df['date'] <= signal_date].index[-1]
        signal_price = self.df.loc[signal_idx, 'close']
        
        returns = {}
        
        for period in periods:
            future_idx = signal_idx + period
            
            if future_idx < len(self.df):
                future_price = self.df.loc[future_idx, 'close']
                period_return = ((future_price - signal_price) / signal_price) * 100
                returns[f'{period}d'] = period_return
            else:
                returns[f'{period}d'] = np.nan
        
        # Zusätzlich: Maximum und Minimum in der Period
        for period in periods:
            end_idx = min(signal_idx + period, len(self.df) - 1)
            period_data = self.df.loc[signal_idx:end_idx]
            
            if not period_data.empty:
                max_price = period_data['high'].max()
                min_price = period_data['low'].min()
                
                max_return = ((max_price - signal_price) / signal_price) * 100
                min_return = ((min_price - signal_price) / signal_price) * 100
                
                returns[f'{period}d_max'] = max_return
                returns[f'{period}d_min'] = min_return
        
        return returns
    
    def analyze_divergence_performance(self, divergence_type: str, 
                                     signals_df: pd.DataFrame) -> Dict:
        """
        Analysiert die Performance eines Divergenz-Typs
        
        Args:
            divergence_type: Type der Divergenz (z.B. 'classic_bullish')
            signals_df: DataFrame mit den Signalen
            
        Returns:
            Dictionary mit Performance-Statistiken
        """
        logger.info(f"Analysiere Performance von {divergence_type}")
        
        results = []
        periods = [7, 14, 30, 60]
        
        for idx, signal in signals_df.iterrows():
            signal_data = {
                'date': signal['date'],
                'price': signal['close'],
                'rsi': signal.get('RSI', np.nan),
                'macd': signal.get('macd_histogram', np.nan),
                'volume': signal.get('volume', np.nan)
            }
            
            # Berechne zukünftige Returns
            future_returns = self.calculate_future_returns(signal['date'], periods)
            signal_data.update(future_returns)
            
            # Zusätzliche Metriken
            signal_data['hit_rate_7d'] = 1 if future_returns.get('7d', 0) > 0 else 0
            signal_data['hit_rate_14d'] = 1 if future_returns.get('14d', 0) > 0 else 0
            signal_data['hit_rate_30d'] = 1 if future_returns.get('30d', 0) > 0 else 0
            
            # Risk-Reward Ratio
            max_gain_30d = future_returns.get('30d_max', 0)
            max_loss_30d = abs(future_returns.get('30d_min', 0))
            signal_data['risk_reward_30d'] = max_gain_30d / max_loss_30d if max_loss_30d > 0 else np.inf
            
            results.append(signal_data)
        
        # Erstelle DataFrame mit Ergebnissen
        results_df = pd.DataFrame(results)
        
        # Berechne Gesamtstatistiken
        stats = {
            'total_signals': len(results_df),
            'avg_return_7d': results_df['7d'].mean(),
            'avg_return_14d': results_df['14d'].mean(),
            'avg_return_30d': results_df['30d'].mean(),
            'avg_return_60d': results_df['60d'].mean(),
            'hit_rate_7d': results_df['hit_rate_7d'].mean() * 100,
            'hit_rate_14d': results_df['hit_rate_14d'].mean() * 100,
            'hit_rate_30d': results_df['hit_rate_30d'].mean() * 100,
            'median_return_30d': results_df['30d'].median(),
            'std_return_30d': results_df['30d'].std(),
            'max_return_30d': results_df['30d'].max(),
            'min_return_30d': results_df['30d'].min(),
            'avg_risk_reward': results_df['risk_reward_30d'].replace([np.inf, -np.inf], np.nan).mean(),
            'positive_signals': (results_df['30d'] > 0).sum(),
            'negative_signals': (results_df['30d'] < 0).sum(),
            'neutral_signals': (results_df['30d'] == 0).sum()
        }
        
        return {
            'type': divergence_type,
            'stats': stats,
            'detailed_results': results_df
        }
    
    def run_full_validation(self) -> Dict:
        """
        Führt vollständige Validierung aller Divergenz-Typen durch
        
        Returns:
            Dictionary mit allen Validierungsergebnissen
        """
        logger.info("🔍 Starte vollständige Divergenz-Validierung...")
        
        signals = self.extract_divergence_signals()
        
        if not signals:
            logger.warning("Keine Divergenz-Signale gefunden!")
            return {}
        
        all_results = {}
        
        for divergence_type, signals_df in signals.items():
            result = self.analyze_divergence_performance(divergence_type, signals_df)
            all_results[divergence_type] = result
        
        # Vergleichsstatistiken
        comparison = self._create_comparison_stats(all_results)
        all_results['comparison'] = comparison
        
        self.validation_results = all_results
        
        logger.info("✅ Validierung abgeschlossen")
        return all_results
    
    def _create_comparison_stats(self, results: Dict) -> Dict:
        """
        Erstellt Vergleichsstatistiken zwischen Divergenz-Typen
        """
        comparison = {}
        
        if len(results) > 1:
            # Vergleiche Hit Rates
            hit_rates = {}
            avg_returns = {}
            
            for div_type, result in results.items():
                stats = result['stats']
                hit_rates[div_type] = stats['hit_rate_30d']
                avg_returns[div_type] = stats['avg_return_30d']
            
            comparison['hit_rate_ranking'] = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)
            comparison['return_ranking'] = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
            
            # Beste Performance
            best_hit_rate = max(hit_rates.items(), key=lambda x: x[1])
            best_return = max(avg_returns.items(), key=lambda x: x[1])
            
            comparison['best_hit_rate'] = best_hit_rate
            comparison['best_return'] = best_return
        
        return comparison
    
    def create_performance_report(self) -> go.Figure:
        """
        Erstellt visuellen Performance-Report
        
        Returns:
            Plotly Figure mit Performance-Übersicht
        """
        if not self.validation_results:
            self.run_full_validation()
        
        # 2x2 Subplot Layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hit Rates Vergleich', 'Return Verteilung', 
                          'Performance Timeline', 'Risk-Reward Analyse'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        color_idx = 0
        
        # Plot 1: Hit Rates Vergleich
        div_types = []
        hit_rates_7d = []
        hit_rates_30d = []
        
        for div_type, result in self.validation_results.items():
            if div_type != 'comparison':
                div_types.append(div_type.replace('_', ' ').title())
                hit_rates_7d.append(result['stats']['hit_rate_7d'])
                hit_rates_30d.append(result['stats']['hit_rate_30d'])
        
        fig.add_trace(
            go.Bar(name='7 Tage', x=div_types, y=hit_rates_7d, 
                   marker_color=colors[0], opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='30 Tage', x=div_types, y=hit_rates_30d, 
                   marker_color=colors[1], opacity=0.7),
            row=1, col=1
        )
        
        # Plot 2: Return Verteilung (Box Plot)
        for div_type, result in self.validation_results.items():
            if div_type != 'comparison':
                fig.add_trace(
                    go.Box(y=result['detailed_results']['30d'], 
                           name=div_type.replace('_', ' ').title(),
                           marker_color=colors[color_idx % len(colors)]),
                    row=1, col=2
                )
                color_idx += 1
        
        # Plot 3: Performance Timeline
        color_idx = 0
        for div_type, result in self.validation_results.items():
            if div_type != 'comparison':
                detailed = result['detailed_results']
                fig.add_trace(
                    go.Scatter(x=detailed['date'], y=detailed['30d'],
                             mode='markers', name=div_type.replace('_', ' ').title(),
                             marker=dict(size=8, color=colors[color_idx % len(colors)]),
                             hovertemplate="Datum: %{x}<br>Return: %{y:.2f}%<extra></extra>"),
                    row=2, col=1
                )
                color_idx += 1
        
        # Plot 4: Risk-Reward Analyse
        color_idx = 0
        for div_type, result in self.validation_results.items():
            if div_type != 'comparison':
                detailed = result['detailed_results']
                fig.add_trace(
                    go.Scatter(x=detailed['30d_max'], y=detailed['30d_min'],
                             mode='markers', name=div_type.replace('_', ' ').title(),
                             marker=dict(size=8, color=colors[color_idx % len(colors)]),
                             hovertemplate="Max Gain: %{x:.2f}%<br>Max Loss: %{y:.2f}%<extra></extra>"),
                    row=2, col=2
                )
                color_idx += 1
        
        # Layout Updates
        fig.update_layout(
            title_text="Divergenz Performance Report",
            template="plotly_white",
            height=700,
            showlegend=True
        )
        
        # Y-Achsen Labels
        fig.update_yaxes(title_text="Hit Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="30d Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Max Loss (%)", row=2, col=2)
        fig.update_xaxes(title_text="Max Gain (%)", row=2, col=2)
        
        return fig
    
    def export_results_to_excel(self, filename: str = None) -> str:
        """
        Exportiert Validierungsergebnisse nach Excel
        
        Args:
            filename: Zieldatei (optional)
            
        Returns:
            Pfad zur erstellten Datei
        """
        if not self.validation_results:
            self.run_full_validation()
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"divergence_validation_{timestamp}.xlsx"
        
        logger.info(f"Exportiere Ergebnisse nach {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary Sheet
            summary_data = []
            for div_type, result in self.validation_results.items():
                if div_type != 'comparison':
                    stats = result['stats']
                    summary_data.append({
                        'Divergenz_Typ': div_type,
                        'Anzahl_Signale': stats['total_signals'],
                        'Hit_Rate_7d': f"{stats['hit_rate_7d']:.1f}%",
                        'Hit_Rate_30d': f"{stats['hit_rate_30d']:.1f}%",
                        'Avg_Return_30d': f"{stats['avg_return_30d']:.2f}%",
                        'Max_Return_30d': f"{stats['max_return_30d']:.2f}%",
                        'Min_Return_30d': f"{stats['min_return_30d']:.2f}%",
                        'Risk_Reward': f"{stats['avg_risk_reward']:.2f}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detaillierte Ergebnisse pro Typ
            for div_type, result in self.validation_results.items():
                if div_type != 'comparison':
                    sheet_name = div_type.replace('_', ' ').title()[:31]  # Excel Limit
                    result['detailed_results'].to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"✅ Excel-Export erstellt: {filename}")
        return filename
    
    def print_summary_report(self):
        """
        Druckt zusammenfassenden Bericht in die Konsole
        """
        if not self.validation_results:
            self.run_full_validation()
        
        print("\n" + "="*60)
        print("📊 DIVERGENZ VALIDIERUNG - ZUSAMMENFASSUNG")
        print("="*60)
        
        for div_type, result in self.validation_results.items():
            if div_type != 'comparison':
                stats = result['stats']
                
                print(f"\n🔍 {div_type.replace('_', ' ').upper()}")
                print("-" * 40)
                print(f"Anzahl Signale:      {stats['total_signals']}")
                print(f"Hit Rate (7d):       {stats['hit_rate_7d']:.1f}%")
                print(f"Hit Rate (30d):      {stats['hit_rate_30d']:.1f}%")
                print(f"Ø Return (30d):      {stats['avg_return_30d']:+.2f}%")
                print(f"Median Return (30d): {stats['median_return_30d']:+.2f}%")
                print(f"Max Return (30d):    {stats['max_return_30d']:+.2f}%")
                print(f"Min Return (30d):    {stats['min_return_30d']:+.2f}%")
                print(f"Risk/Reward:         {stats['avg_risk_reward']:.2f}")
                print(f"Positive Signale:    {stats['positive_signals']}")
                print(f"Negative Signale:    {stats['negative_signals']}")
        
        # Vergleich
        if 'comparison' in self.validation_results:
            comp = self.validation_results['comparison']
            if 'best_hit_rate' in comp:
                print(f"\n🏆 BESTE PERFORMANCE")
                print("-" * 40)
                print(f"Beste Hit Rate:      {comp['best_hit_rate'][0]} ({comp['best_hit_rate'][1]:.1f}%)")
                print(f"Beste Returns:       {comp['best_return'][0]} ({comp['best_return'][1]:+.2f}%)")
        
        print("\n" + "="*60)


def main():
    """
    Test-Funktion für Standalone-Verwendung
    """
    # Beispiel-Usage
    print("🔍 Divergenz Validator - Standalone Test")
    print("Bitte mit vorbereiteten Daten verwenden:")
    print("validator = DivergenceValidator(df)")
    print("results = validator.run_full_validation()")
    print("validator.print_summary_report()")
    print("fig = validator.create_performance_report()")
    print("fig.show()")

if __name__ == "__main__":
    main()
