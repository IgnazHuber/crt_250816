"""
Performance Analyzer - Systematische Validierung aller Divergenz-Typen
Analysiert ob Bullish-Signale zu Anstiegen und Bearish-Signale zu Rückgängen führen
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

class DivergencePerformanceAnalyzer:
    """
    Analysiert die Performance aller Divergenz-Typen systematisch
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialisierung mit Daten die alle Divergenz-Typen enthalten
        
        Args:
            df: DataFrame mit OHLC-Daten und allen berechneten Divergenzen
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Definiere alle Divergenz-Typen
        self.divergence_types = {
            'Classic_Bullish': {
                'column': 'CBullD_gen',
                'expected_direction': 'up',
                'description': 'Classic Bullish Divergence',
                'color': '#00cc44'
            },
            'Negative_MACD': {
                'column': 'CBullD_neg_MACD', 
                'expected_direction': 'up',
                'description': 'Negative MACD Bullish Divergence',
                'color': '#0088cc'
            },
            'Hidden_Bullish': {
                'column': 'HBullD',
                'expected_direction': 'up', 
                'description': 'Hidden Bullish Divergence',
                'color': '#ff8800'
            },
            'Classic_Bearish': {
                'column': 'BearD',
                'expected_direction': 'down',
                'description': 'Classic Bearish Divergence', 
                'color': '#cc0000'
            }
        }
        
        self.analysis_periods = [3, 7, 14, 30, 60]  # Tage nach Signal
        self.performance_results = {}
        
        logger.info(f"Performance Analyzer initialisiert mit {len(self.df)} Datenpunkten")
    
    def calculate_signal_performance(self, signal_date: pd.Timestamp, 
                                   expected_direction: str,
                                   periods: List[int] = None) -> Dict[str, float]:
        """
        Berechnet Performance-Metriken für ein einzelnes Signal
        
        Args:
            signal_date: Datum des Signals
            expected_direction: 'up' für Bullish, 'down' für Bearish
            periods: Liste der Analysezeiträume in Tagen
            
        Returns:
            Dictionary mit Performance-Metriken
        """
        if periods is None:
            periods = self.analysis_periods
        
        signal_idx = self.df[self.df['date'] <= signal_date].index[-1]
        signal_price = self.df.loc[signal_idx, 'close']
        
        results = {
            'signal_date': signal_date,
            'signal_price': signal_price,
            'expected_direction': expected_direction
        }
        
        for period in periods:
            future_idx = signal_idx + period
            
            if future_idx < len(self.df):
                future_price = self.df.loc[future_idx, 'close']
                
                # Berechne Returns
                period_return = ((future_price - signal_price) / signal_price) * 100
                results[f'return_{period}d'] = period_return
                
                # Erfolg basierend auf erwarteter Richtung
                if expected_direction == 'up':
                    results[f'success_{period}d'] = 1 if period_return > 0 else 0
                    results[f'strong_success_{period}d'] = 1 if period_return > 2 else 0  # >2% Anstieg
                else:  # 'down'
                    results[f'success_{period}d'] = 1 if period_return < 0 else 0
                    results[f'strong_success_{period}d'] = 1 if period_return < -2 else 0  # >2% Rückgang
                
                # Max/Min in der Periode
                end_idx = min(signal_idx + period, len(self.df) - 1)
                period_data = self.df.loc[signal_idx:end_idx]
                
                max_price = period_data['high'].max()
                min_price = period_data['low'].min()
                
                max_return = ((max_price - signal_price) / signal_price) * 100
                min_return = ((min_price - signal_price) / signal_price) * 100
                
                results[f'max_return_{period}d'] = max_return
                results[f'min_return_{period}d'] = min_return
                
                # Risk-Reward
                if expected_direction == 'up':
                    potential_gain = max_return
                    potential_loss = abs(min_return) if min_return < 0 else 0.1
                else:
                    potential_gain = abs(min_return) if min_return < 0 else 0.1
                    potential_loss = max_return if max_return > 0 else 0.1
                
                results[f'risk_reward_{period}d'] = potential_gain / potential_loss if potential_loss > 0 else 0
            else:
                # Nicht genügend Zukunftsdaten
                for metric in ['return', 'success', 'strong_success', 'max_return', 'min_return', 'risk_reward']:
                    results[f'{metric}_{period}d'] = np.nan
        
        return results
    
    def analyze_divergence_type_performance(self, divergence_type: str) -> Dict:
        """
        Analysiert die Performance eines spezifischen Divergenz-Typs
        
        Args:
            divergence_type: Key aus self.divergence_types
            
        Returns:
            Dictionary mit aggregierten Performance-Statistiken
        """
        if divergence_type not in self.divergence_types:
            raise ValueError(f"Unbekannter Divergenz-Typ: {divergence_type}")
        
        div_info = self.divergence_types[divergence_type]
        column = div_info['column']
        expected_direction = div_info['expected_direction']
        
        if column not in self.df.columns:
            logger.warning(f"Spalte {column} nicht in Daten gefunden")
            return {}
        
        # Finde alle Signale dieses Typs
        signals = self.df[self.df[column] == 1].copy()
        
        if signals.empty:
            logger.warning(f"Keine Signale für {divergence_type} gefunden")
            return {}
        
        logger.info(f"Analysiere {len(signals)} {divergence_type} Signale...")
        
        # Analysiere jedes Signal
        signal_results = []
        for idx, signal in signals.iterrows():
            signal_performance = self.calculate_signal_performance(
                signal['date'], 
                expected_direction
            )
            signal_results.append(signal_performance)
        
        # Erstelle DataFrame für bessere Aggregation
        results_df = pd.DataFrame(signal_results)
        
        # Berechne aggregierte Statistiken
        stats = {
            'type': divergence_type,
            'description': div_info['description'],
            'expected_direction': expected_direction,
            'total_signals': len(signals),
            'color': div_info['color']
        }
        
        # Für jede Periode
        for period in self.analysis_periods:
            # Hit Rates
            stats[f'hit_rate_{period}d'] = results_df[f'success_{period}d'].mean() * 100
            stats[f'strong_hit_rate_{period}d'] = results_df[f'strong_success_{period}d'].mean() * 100
            
            # Returns
            stats[f'avg_return_{period}d'] = results_df[f'return_{period}d'].mean()
            stats[f'median_return_{period}d'] = results_df[f'return_{period}d'].median()
            stats[f'std_return_{period}d'] = results_df[f'return_{period}d'].std()
            
            # Extrema
            stats[f'max_return_{period}d'] = results_df[f'return_{period}d'].max()
            stats[f'min_return_{period}d'] = results_df[f'return_{period}d'].min()
            
            # Risk-Reward
            stats[f'avg_risk_reward_{period}d'] = results_df[f'risk_reward_{period}d'].mean()
            
            # Positive/Negative Signale
            stats[f'positive_signals_{period}d'] = (results_df[f'return_{period}d'] > 0).sum()
            stats[f'negative_signals_{period}d'] = (results_df[f'return_{period}d'] < 0).sum()
        
        return {
            'stats': stats,
            'detailed_results': results_df,
            'signals_data': signals
        }
    
    def run_complete_performance_analysis(self) -> Dict:
        """
        Führt komplette Performance-Analyse für alle Divergenz-Typen durch
        
        Returns:
            Dictionary mit allen Analyseergebnissen
        """
        logger.info("🔍 Starte komplette Performance-Analyse...")
        
        all_results = {}
        
        for div_type in self.divergence_types.keys():
            try:
                result = self.analyze_divergence_type_performance(div_type)
                if result:
                    all_results[div_type] = result
                    
                    # Log wichtige Metriken
                    stats = result['stats']
                    logger.info(f"✅ {div_type}: {stats['total_signals']} Signale, "
                              f"Hit Rate 30d: {stats.get('hit_rate_30d', 0):.1f}%, "
                              f"Avg Return 30d: {stats.get('avg_return_30d', 0):+.2f}%")
            except Exception as e:
                logger.error(f"❌ Fehler bei {div_type}: {e}")
                continue
        
        # Vergleichsanalyse
        comparison = self._create_comparison_analysis(all_results)
        all_results['comparison'] = comparison
        
        self.performance_results = all_results
        
        logger.info("✅ Performance-Analyse abgeschlossen")
        return all_results
    
    def _create_comparison_analysis(self, results: Dict) -> Dict:
        """
        Erstellt Vergleichsanalyse zwischen allen Divergenz-Typen
        """
        comparison = {
            'rankings': {},
            'effectiveness': {},
            'summary': {}
        }
        
        # Sammle Metriken für Vergleich
        bullish_types = []
        bearish_types = []
        
        for div_type, result in results.items():
            if 'stats' in result:
                stats = result['stats']
                if stats['expected_direction'] == 'up':
                    bullish_types.append((div_type, stats))
                else:
                    bearish_types.append((div_type, stats))
        
        # Ranking für 30-Tage Performance
        def rank_by_metric(types_list, metric, reverse=True):
            return sorted(types_list, 
                         key=lambda x: x[1].get(metric, 0), 
                         reverse=reverse)
        
        # Rankings für verschiedene Metriken
        if bullish_types:
            comparison['rankings']['bullish_hit_rate'] = rank_by_metric(bullish_types, 'hit_rate_30d')
            comparison['rankings']['bullish_returns'] = rank_by_metric(bullish_types, 'avg_return_30d')
            comparison['rankings']['bullish_risk_reward'] = rank_by_metric(bullish_types, 'avg_risk_reward_30d')
        
        if bearish_types:
            comparison['rankings']['bearish_hit_rate'] = rank_by_metric(bearish_types, 'hit_rate_30d')
            comparison['rankings']['bearish_returns'] = rank_by_metric(bearish_types, 'avg_return_30d', reverse=False)  # Für Bearish: niedrigere Returns sind besser
        
        # Effectiveness Summary
        total_signals = sum(r['stats']['total_signals'] for r in results.values() if 'stats' in r)
        
        comparison['summary'] = {
            'total_signals_analyzed': total_signals,
            'bullish_types_count': len(bullish_types),
            'bearish_types_count': len(bearish_types),
            'analysis_periods': self.analysis_periods
        }
        
        return comparison
    
    def create_performance_dashboard(self) -> go.Figure:
        """
        Erstellt umfassendes Performance-Dashboard
        
        Returns:
            Plotly Figure mit mehreren Performance-Plots
        """
        if not self.performance_results:
            self.run_complete_performance_analysis()
        
        # 3x2 Dashboard Layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Hit Rates Vergleich (30 Tage)', 'Return Verteilung (30 Tage)',
                'Hit Rate über Zeit', 'Risk-Reward Analyse', 
                'Signal Häufigkeit', 'Erfolg vs. Misserfolg'
            ),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Daten für Plots sammeln
        types = []
        hit_rates = []
        avg_returns = []
        colors = []
        signal_counts = []
        
        for div_type, result in self.performance_results.items():
            if div_type != 'comparison' and 'stats' in result:
                stats = result['stats']
                types.append(stats['description'])
                hit_rates.append(stats.get('hit_rate_30d', 0))
                avg_returns.append(stats.get('avg_return_30d', 0))
                colors.append(stats['color'])
                signal_counts.append(stats['total_signals'])
        
        # Plot 1: Hit Rates Vergleich
        fig.add_trace(
            go.Bar(
                x=types,
                y=hit_rates,
                name='Hit Rate 30d',
                marker_color=colors,
                text=[f"{hr:.1f}%" for hr in hit_rates],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Plot 2: Return Verteilung (Box Plots)
        for i, (div_type, result) in enumerate(self.performance_results.items()):
            if div_type != 'comparison' and 'detailed_results' in result:
                returns_30d = result['detailed_results']['return_30d'].dropna()
                fig.add_trace(
                    go.Box(
                        y=returns_30d,
                        name=result['stats']['description'],
                        marker_color=result['stats']['color']
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Hit Rate über verschiedene Zeitperioden
        for div_type, result in self.performance_results.items():
            if div_type != 'comparison' and 'stats' in result:
                stats = result['stats']
                periods = [3, 7, 14, 30, 60]
                period_hit_rates = [stats.get(f'hit_rate_{p}d', 0) for p in periods]
                
                fig.add_trace(
                    go.Scatter(
                        x=periods,
                        y=period_hit_rates,
                        mode='lines+markers',
                        name=stats['description'],
                        line=dict(color=stats['color'], width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Risk-Reward Analyse
        for div_type, result in self.performance_results.items():
            if div_type != 'comparison' and 'detailed_results' in result:
                detailed = result['detailed_results']
                fig.add_trace(
                    go.Scatter(
                        x=detailed['max_return_30d'],
                        y=detailed['min_return_30d'].abs(),
                        mode='markers',
                        name=result['stats']['description'],
                        marker=dict(
                            size=8,
                            color=result['stats']['color'],
                            opacity=0.7
                        ),
                        hovertemplate="Max Gain: %{x:.2f}%<br>Max Loss: %{y:.2f}%<extra></extra>"
                    ),
                    row=2, col=2
                )
        
        # Plot 5: Signal Häufigkeit
        fig.add_trace(
            go.Bar(
                x=types,
                y=signal_counts,
                name='Anzahl Signale',
                marker_color=colors,
                text=signal_counts,
                textposition='outside'
            ),
            row=3, col=1
        )
        
        # Plot 6: Erfolg vs. Misserfolg (Pie Chart)
        total_success = sum(stats.get('positive_signals_30d', 0) 
                           for result in self.performance_results.values() 
                           if 'stats' in result for stats in [result['stats']])
        
        total_failure = sum(stats.get('negative_signals_30d', 0) 
                           for result in self.performance_results.values() 
                           if 'stats' in result for stats in [result['stats']])
        
        fig.add_trace(
            go.Pie(
                labels=['Erfolgreiche Signale', 'Fehlgeschlagene Signale'],
                values=[total_success, total_failure],
                marker_colors=['#00cc44', '#cc0000'],
                hole=0.3
            ),
            row=3, col=2
        )
        
        # Layout Updates
        fig.update_layout(
            title_text="Divergenz Performance Dashboard",
            template="plotly_white",
            height=1000,
            showlegend=True
        )
        
        # Achsen-Updates
        fig.update_yaxes(title_text="Hit Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Hit Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Tage nach Signal", row=2, col=1)
        fig.update_yaxes(title_text="Max Loss (%)", row=2, col=2)
        fig.update_xaxes(title_text="Max Gain (%)", row=2, col=2)
        fig.update_yaxes(title_text="Anzahl Signale", row=3, col=1)
        
        return fig
    
    def export_performance_analysis(self, filename: str = None) -> str:
        """
        Exportiert komplette Performance-Analyse nach Excel
        
        Args:
            filename: Zieldatei (optional)
            
        Returns:
            Pfad zur erstellten Datei
        """
        if not self.performance_results:
            self.run_complete_performance_analysis()
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"divergence_performance_analysis_{timestamp}.xlsx"
        
        logger.info(f"📊 Exportiere Performance-Analyse: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary Sheet
            summary_data = []
            
            for div_type, result in self.performance_results.items():
                if div_type != 'comparison' and 'stats' in result:
                    stats = result['stats']
                    summary_data.append({
                        'Divergence_Type': stats['description'],
                        'Expected_Direction': stats['expected_direction'],
                        'Total_Signals': stats['total_signals'],
                        'Hit_Rate_3d': f"{stats.get('hit_rate_3d', 0):.1f}%",
                        'Hit_Rate_7d': f"{stats.get('hit_rate_7d', 0):.1f}%", 
                        'Hit_Rate_14d': f"{stats.get('hit_rate_14d', 0):.1f}%",
                        'Hit_Rate_30d': f"{stats.get('hit_rate_30d', 0):.1f}%",
                        'Hit_Rate_60d': f"{stats.get('hit_rate_60d', 0):.1f}%",
                        'Avg_Return_30d': f"{stats.get('avg_return_30d', 0):+.2f}%",
                        'Risk_Reward_30d': f"{stats.get('avg_risk_reward_30d', 0):.2f}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            # Detaillierte Ergebnisse pro Typ
            for div_type, result in self.performance_results.items():
                if div_type != 'comparison' and 'detailed_results' in result:
                    sheet_name = div_type.replace('_', ' ')[:31]  # Excel Sheet Name Limit
                    detailed_df = result['detailed_results'].copy()
                    
                    # Timezone entfernen für Excel
                    if 'signal_date' in detailed_df.columns:
                        detailed_df['signal_date'] = pd.to_datetime(detailed_df['signal_date']).dt.tz_localize(None)
                    
                    detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"✅ Performance-Analyse exportiert: {filename}")
        return filename
    
    def print_performance_summary(self):
        """
        Druckt Performance-Zusammenfassung in die Konsole
        """
        if not self.performance_results:
            self.run_complete_performance_analysis()
        
        print("\n" + "="*80)
        print("📊 DIVERGENZ PERFORMANCE ANALYSE - ZUSAMMENFASSUNG")
        print("="*80)
        
        # Bullish Divergenzen
        print(f"\n🔵 BULLISH DIVERGENZEN (Erwartung: Preisanstieg)")
        print("-" * 60)
        
        for div_type, result in self.performance_results.items():
            if div_type != 'comparison' and 'stats' in result:
                stats = result['stats']
                if stats['expected_direction'] == 'up':
                    print(f"\n📈 {stats['description']}")
                    print(f"   Signale:           {stats['total_signals']}")
                    print(f"   Hit Rate (30d):    {stats.get('hit_rate_30d', 0):.1f}%")
                    print(f"   Ø Return (30d):    {stats.get('avg_return_30d', 0):+.2f}%")
                    print(f"   Risk/Reward:       {stats.get('avg_risk_reward_30d', 0):.2f}")
                    print(f"   Erfolgreiche:      {stats.get('positive_signals_30d', 0)}")
                    print(f"   Fehlgeschlagene:   {stats.get('negative_signals_30d', 0)}")
        
        # Bearish Divergenzen
        print(f"\n🔴 BEARISH DIVERGENZEN (Erwartung: Preisrückgang)")
        print("-" * 60)
        
        for div_type, result in self.performance_results.items():
            if div_type != 'comparison' and 'stats' in result:
                stats = result['stats']
                if stats['expected_direction'] == 'down':
                    print(f"\n📉 {stats['description']}")
                    print(f"   Signale:           {stats['total_signals']}")
                    print(f"   Hit Rate (30d):    {stats.get('hit_rate_30d', 0):.1f}%")
                    print(f"   Ø Return (30d):    {stats.get('avg_return_30d', 0):+.2f}%")
                    print(f"   Risk/Reward:       {stats.get('avg_risk_reward_30d', 0):.2f}")
                    print(f"   Erfolgreiche:      {stats.get('positive_signals_30d', 0)}")
                    print(f"   Fehlgeschlagene:   {stats.get('negative_signals_30d', 0)}")
        
        # Vergleich
        if 'comparison' in self.performance_results:
            comp = self.performance_results['comparison']
            print(f"\n🏆 RANKINGS")
            print("-" * 60)
            
            if 'rankings' in comp:
                rankings = comp['rankings']
                
                if 'bullish_hit_rate' in rankings:
                    print(f"Beste Bullish Hit Rate:")
                    for i, (div_type, stats) in enumerate(rankings['bullish_hit_rate'][:3], 1):
                        print(f"  {i}. {stats['description']}: {stats.get('hit_rate_30d', 0):.1f}%")
                
                if 'bearish_hit_rate' in rankings:
                    print(f"\nBeste Bearish Hit Rate:")
                    for i, (div_type, stats) in enumerate(rankings['bearish_hit_rate'][:3], 1):
                        print(f"  {i}. {stats['description']}: {stats.get('hit_rate_30d', 0):.1f}%")
        
        print("\n" + "="*80)


def main():
    """
    Test-Funktion für Performance Analyzer
    """
    print("📊 Divergence Performance Analyzer")
    print("Verwendung:")
    print("analyzer = DivergencePerformanceAnalyzer(df)")
    print("results = analyzer.run_complete_performance_analysis()") 
    print("analyzer.print_performance_summary()")
    print("dashboard = analyzer.create_performance_dashboard()")
    print("dashboard.show()")

if __name__ == "__main__":
    main()
