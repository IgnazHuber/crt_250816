            logger.info(f"✅ {asset_name}: {len(df)} Zeilen geladen, Zeitraum: {df['date'].min().date()} bis {df['date'].max().date()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von {asset_name if asset_name else file_path}: {e}")
            return False
    
    def load_multiple_assets_interactive(self) -> int:
        """
        Interaktive Auswahl und Laden mehrerer Asset-Dateien
        
        Returns:
            Anzahl erfolgreich geladener Assets
        """
        logger.info("📁 Starte interaktive Asset-Auswahl...")
        
        root = tk.Tk()
        root.withdraw()
        
        file_paths = filedialog.askopenfilenames(
            title="Wähle mehrere Asset-Dateien für Vergleichsanalyse",
            filetypes=[
                ("CSV Dateien", "*.csv"),
                ("Parquet Dateien", "*.parquet"),
                ("Alle unterstützten", "*.csv;*.parquet"),
                ("Alle Dateien", "*.*")
            ]
        )
        
        root.destroy()
        
        if not file_paths:
            logger.warning("Keine Dateien ausgewählt")
            return 0
        
        successful_loads = 0
        
        for file_path in file_paths:
            if self.load_asset_data(file_path):
                successful_loads += 1
        
        logger.info(f"✅ {successful_loads}/{len(file_paths)} Assets erfolgreich geladen")
        return successful_loads
    
    def prepare_asset_indicators(self, asset_name: str) -> bool:
        """
        Bereitet Basis-Indikatoren für ein Asset vor
        
        Args:
            asset_name: Name des Assets
            
        Returns:
            True wenn erfolgreich
        """
        try:
            if asset_name not in self.assets:
                logger.error(f"Asset {asset_name} nicht gefunden")
                return False
            
            df = self.assets[asset_name]
            
            logger.info(f"🔧 Berechne Indikatoren für {asset_name}...")
            
            # Basis-Indikatoren berechnen
            Initialize_RSI_EMA_MACD(df)
            Local_Max_Min(df)
            
            logger.info(f"✅ Indikatoren für {asset_name} berechnet")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Indikator-Berechnung für {asset_name}: {e}")
            return False
    
    def analyze_asset_divergences(self, asset_name: str, 
                                window: int = 5, candle_tol: float = 0.1, 
                                macd_tol: float = 3.25) -> Dict:
        """
        Führt komplette Divergenz-Analyse für ein Asset durch
        
        Args:
            asset_name: Name des Assets
            window: Fenster-Größe
            candle_tol: Candlestick-Toleranz
            macd_tol: MACD-Toleranz
            
        Returns:
            Dictionary mit Analyseergebnissen
        """
        try:
            if asset_name not in self.assets:
                logger.error(f"Asset {asset_name} nicht gefunden")
                return {}
            
            df = self.assets[asset_name].copy()
            
            logger.info(f"📈 Analysiere Divergenzen für {asset_name}...")
            
            # Alle Divergenz-Typen berechnen
            CBullDivg_analysis(df, window, candle_tol, macd_tol)
            add_hidden_bullish_divergences(df, window, candle_tol, macd_tol)
            add_bearish_divergences(df, window, candle_tol, macd_tol)
            
            # Performance-Analyse
            analyzer = DivergencePerformanceAnalyzer(df)
            performance_results = analyzer.run_complete_performance_analysis()
            
            # Sammle Statistiken
            stats = {}
            divergence_types = [
                ('CBullD_gen', 'Classic Bullish'),
                ('CBullD_neg_MACD', 'Negative MACD'),
                ('HBullD', 'Hidden Bullish'),
                ('BearD', 'Classic Bearish')
            ]
            
            total_signals = 0
            for col, name in divergence_types:
                if col in df.columns:
                    count = (df[col] == 1).sum()
                    stats[name] = count
                    total_signals += count
            
            stats['Total Signals'] = total_signals
            stats['Signal Density'] = (total_signals / len(df)) * 100  # Signale pro 100 Datenpunkte
            
            result = {
                'asset_name': asset_name,
                'data': df,
                'performance_analysis': performance_results,
                'divergence_statistics': stats,
                'parameters_used': {
                    'window': window,
                    'candle_tolerance': candle_tol,
                    'macd_tolerance': macd_tol
                }
            }
            
            self.analysis_results[asset_name] = result
            
            logger.info(f"✅ {asset_name}: {total_signals} Divergenzen gefunden ({stats['Signal Density']:.1f} pro 100 Bars)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Divergenz-Analyse für {asset_name}: {e}")
            return {}
    
    def optimize_asset_parameters(self, asset_name: str, 
                                max_workers: int = 1) -> Dict:
        """
        Optimiert Parameter für ein spezifisches Asset
        
        Args:
            asset_name: Name des Assets
            max_workers: Anzahl paralleler Prozesse
            
        Returns:
            Dictionary mit Optimierungsergebnissen
        """
        try:
            if asset_name not in self.assets:
                logger.error(f"Asset {asset_name} nicht gefunden")
                return {}
            
            df = self.assets[asset_name].copy()
            
            logger.info(f"🎯 Optimiere Parameter für {asset_name}...")
            
            # Parameter-Optimierung
            optimizer = ParameterOptimizer(df, asset_name)
            
            # Asset-spezifische Parameter-Bereiche (optional anpassbar)
            if asset_name in ['Bitcoin', 'Ethereum']:
                # Crypto: Höhere Volatilität, andere Parameter
                optimizer.set_parameter_ranges({
                    'window': [3, 4, 5, 6, 7],
                    'candle_tolerance': [0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
                    'macd_tolerance': [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
                })
            elif asset_name in ['SP500', 'EUR/USD']:
                # Traditional Assets: Konservativere Parameter
                optimizer.set_parameter_ranges({
                    'window': [4, 5, 6, 7, 8],
                    'candle_tolerance': [0.05, 0.1, 0.15, 0.2],
                    'macd_tolerance': [2.0, 3.0, 3.25, 4.0, 5.0]
                })
            
            # Vollständige Optimierung
            optimization_results = optimizer.run_complete_optimization(max_workers)
            
            self.optimization_results[asset_name] = optimization_results
            
            logger.info(f"✅ Parameter-Optimierung für {asset_name} abgeschlossen")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Parameter-Optimierung für {asset_name}: {e}")
            return {}
    
    def run_asset_workflow(self, asset_name: str, 
                          optimize_parameters: bool = True,
                          max_workers: int = 1) -> bool:
        """
        Führt kompletten Workflow für ein Asset durch
        
        Args:
            asset_name: Name des Assets
            optimize_parameters: Ob Parameter optimiert werden sollen
            max_workers: Anzahl paralleler Prozesse
            
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info(f"🚀 Starte Workflow für {asset_name}")
            
            # Schritt 1: Indikatoren vorbereiten
            if not self.prepare_asset_indicators(asset_name):
                return False
            
            # Schritt 2: Parameter optimieren (optional)
            if optimize_parameters:
                optimization_result = self.optimize_asset_parameters(asset_name, max_workers)
                
                # Verwende optimierte Parameter wenn verfügbar
                if optimization_result and 'optimal_parameters' in optimization_result:
                    optimal = optimization_result['optimal_parameters'].get('combined_score', {})
                    if optimal:
                        window = optimal.get('window', 5)
                        candle_tol = optimal.get('candle_tolerance', 0.1)
                        macd_tol = optimal.get('macd_tolerance', 3.25)
                        
                        logger.info(f"🎯 Verwende optimierte Parameter für {asset_name}: W{window}_C{candle_tol:.2f}_M{macd_tol:.2f}")
                    else:
                        window, candle_tol, macd_tol = 5, 0.1, 3.25
                else:
                    window, candle_tol, macd_tol = 5, 0.1, 3.25
            else:
                window, candle_tol, macd_tol = 5, 0.1, 3.25
            
            # Schritt 3: Divergenz-Analyse mit optimalen/Standard-Parametern
            analysis_result = self.analyze_asset_divergences(asset_name, window, candle_tol, macd_tol)
            
            if not analysis_result:
                return False
            
            logger.info(f"✅ Workflow für {asset_name} erfolgreich abgeschlossen")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler im Workflow für {asset_name}: {e}")
            return False
    
    def create_cross_asset_comparison(self) -> Dict:
        """
        Erstellt Asset-übergreifende Vergleichsanalyse
        
        Returns:
            Dictionary mit Vergleichsergebnissen
        """
        logger.info("📊 Erstelle Asset-übergreifende Vergleichsanalyse...")
        
        if not self.analysis_results:
            logger.error("Keine Analyseergebnisse vorhanden")
            return {}
        
        comparison = {
            'asset_summary': {},
            'performance_ranking': {},
            'divergence_effectiveness': {},
            'parameter_analysis': {}
        }
        
        # Asset Summary
        for asset_name, result in self.analysis_results.items():
            stats = result['divergence_statistics']
            
            comparison['asset_summary'][asset_name] = {
                'total_signals': stats['Total Signals'],
                'signal_density': stats['Signal Density'],
                'classic_bullish': stats.get('Classic Bullish', 0),
                'hidden_bullish': stats.get('Hidden Bullish', 0),
                'classic_bearish': stats.get('Classic Bearish', 0),
                'data_points': len(result['data'])
            }
        
        # Performance Ranking
        for metric in ['hit_rate_30d', 'avg_return_30d', 'avg_risk_reward_30d']:
            ranking = []
            
            for asset_name, result in self.analysis_results.items():
                perf_results = result['performance_analysis']
                
                # Berechne gewichteten Durchschnitt über alle Divergenz-Typen
                total_weight = 0
                weighted_metric = 0
                
                for div_type, div_result in perf_results.items():
                    if div_type != 'comparison' and 'stats' in div_result:
                        stats = div_result['stats']
                        weight = stats['total_signals']
                        value = stats.get(metric, 0)
                        
                        weighted_metric += value * weight
                        total_weight += weight
                
                if total_weight > 0:
                    avg_metric = weighted_metric / total_weight
                    ranking.append((asset_name, avg_metric, total_weight))
            
            # Sortiere nach Metrik-Wert
            ranking.sort(key=lambda x: x[1], reverse=True)
            comparison['performance_ranking'][metric] = ranking
        
        # Divergence Effectiveness (welcher Typ funktioniert wo am besten)
        divergence_effectiveness = {}
        
        for div_type_name in ['Classic Bullish', 'Hidden Bullish', 'Classic Bearish']:
            effectiveness = []
            
            for asset_name, result in self.analysis_results.items():
                perf_results = result['performance_analysis']
                
                # Finde entsprechenden Divergenz-Typ in Performance-Ergebnissen
                for div_type, div_result in perf_results.items():
                    if div_type != 'comparison' and 'stats' in div_result:
                        stats = div_result['stats']
                        if div_type_name.lower().replace(' ', '_') in div_type.lower():
                            hit_rate = stats.get('hit_rate_30d', 0)
                            signals = stats['total_signals']
                            
                            if signals > 0:
                                effectiveness.append((asset_name, hit_rate, signals))
                            break
            
            effectiveness.sort(key=lambda x: x[1], reverse=True)
            divergence_effectiveness[div_type_name] = effectiveness
        
        comparison['divergence_effectiveness'] = divergence_effectiveness
        
        # Parameter Analysis (falls Optimierungen durchgeführt wurden)
        if self.optimization_results:
            parameter_summary = {}
            
            for asset_name, opt_result in self.optimization_results.items():
                if 'optimal_parameters' in opt_result:
                    optimal = opt_result['optimal_parameters'].get('combined_score', {})
                    if optimal:
                        parameter_summary[asset_name] = {
                            'window': optimal.get('window', 5),
                            'candle_tolerance': optimal.get('candle_tolerance', 0.1),
                            'macd_tolerance': optimal.get('macd_tolerance', 3.25),
                            'score': optimal.get('score', 0)
                        }
            
            comparison['parameter_analysis'] = parameter_summary
        
        self.comparison_results = comparison
        
        logger.info("✅ Asset-Vergleichsanalyse abgeschlossen")
        return comparison
    
    def create_multi_asset_dashboard(self) -> go.Figure:
        """
        Erstellt umfassendes Multi-Asset Dashboard
        
        Returns:
            Plotly Figure mit Asset-Vergleichen
        """
        if not self.comparison_results:
            self.create_cross_asset_comparison()
        
        # 2x3 Dashboard Layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Signal Density Vergleich', 'Hit Rate Ranking', 'Divergenz-Typ Effectiveness',
                'Signal Verteilung', 'Performance Scatter', 'Parameter Heatmap'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Daten für Plots vorbereiten
        assets = list(self.comparison_results['asset_summary'].keys())
        colors = px.colors.qualitative.Set3[:len(assets)]
        
        # Plot 1: Signal Density Vergleich
        signal_densities = [self.comparison_results['asset_summary'][asset]['signal_density'] 
                           for asset in assets]
        
        fig.add_trace(
            go.Bar(
                x=assets,
                y=signal_densities,
                name='Signal Density',
                marker_color=colors,
                text=[f"{sd:.1f}" for sd in signal_densities],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Plot 2: Hit Rate Ranking
        if 'hit_rate_30d' in self.comparison_results['performance_ranking']:
            ranking = self.comparison_results['performance_ranking']['hit_rate_30d']
            ranked_assets = [item[0] for item in ranking]
            ranked_hit_rates = [item[1] for item in ranking]
            
            fig.add_trace(
                go.Bar(
                    x=ranked_assets,
                    y=ranked_hit_rates,
                    name='Hit Rate 30d',
                    marker_color='lightcoral',
                    text=[f"{hr:.1f}%" for hr in ranked_hit_rates],
                    textposition='outside'
                ),
                row=1, col=2
            )
        
        # Plot 3: Divergenz-Typ Effectiveness
        if 'Classic Bullish' in self.comparison_results['divergence_effectiveness']:
            classic_effectiveness = self.comparison_results['divergence_effectiveness']['Classic Bullish']
            classic_assets = [item[0] for item in classic_effectiveness]
            classic_rates = [item[1] for item in classic_effectiveness]
            
            fig.add_trace(
                go.Bar(
                    x=classic_assets,
                    y=classic_rates,
                    name='Classic Bullish Hit Rate',
                    marker_color='lightgreen'
                ),
                row=1, col=3
            )
        
        # Plot 4: Signal Verteilung (Stacked Bar)
        signal_types = ['Classic Bullish', 'Hidden Bullish', 'Classic Bearish']
        
        for i, signal_type in enumerate(signal_types):
            signal_counts = []
            for asset in assets:
                count = self.comparison_results['asset_summary'][asset].get(signal_type.lower().replace(' ', '_'), 0)
                signal_counts.append(count)
            
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=signal_counts,
                    name=signal_type,
                    marker_color=colors[i % len(colors)]
                ),
                row=2, col=1
            )
        
        # Plot 5: Performance Scatter (Hit Rate vs Return)
        if 'hit_rate_30d' in self.comparison_results['performance_ranking']:
            hit_rates = []
            returns = []
            asset_labels = []
            
            for asset in assets:
                if asset in [item[0] for item in self.comparison_results['performance_ranking']['hit_rate_30d']]:
                    hit_rate = next(item[1] for item in self.comparison_results['performance_ranking']['hit_rate_30d'] if item[0] == asset)
                    return_val = next(item[1] for item in self.comparison_results['performance_ranking'].get('avg_return_30d', []) if item[0] == asset) if 'avg_return_30d' in self.comparison_results['performance_ranking'] else 0
                    
                    hit_rates.append(hit_rate)
                    returns.append(return_val)
                    asset_labels.append(asset)
            
            fig.add_trace(
                go.Scatter(
                    x=hit_rates,
                    y=returns,
                    mode='markers+text',
                    text=asset_labels,
                    textposition='top center',
                    marker=dict(size=12, color=colors[:len(asset_labels)]),
                    name='Asset Performance'
                ),
                row=2, col=2
            )
        
        # Plot 6: Parameter Heatmap (falls verfügbar)
        if 'parameter_analysis' in self.comparison_results and self.comparison_results['parameter_analysis']:
            param_assets = list(self.comparison_results['parameter_analysis'].keys())
            param_data = []
            param_labels = ['Window', 'Candle Tol', 'MACD Tol']
            
            for asset in param_assets:
                params = self.comparison_results['parameter_analysis'][asset]
                param_data.append([
                    params.get('window', 5),
                    params.get('candle_tolerance', 0.1) * 100,  # Prozent
                    params.get('macd_tolerance', 3.25)
                ])
            
            if param_data:
                fig.add_trace(
                    go.Heatmap(
                        z=np.array(param_data).T,
                        x=param_assets,
                        y=param_labels,
                        colorscale='Viridis',
                        text=np.array(param_data).T,
                        texttemplate="%{text:.2f}",
                        textfont={"size": 10}
                    ),
                    row=2, col=3
                )
        
        # Layout Updates
        fig.update_layout(
            title_text="Multi-Asset Divergenz Analyse Dashboard",
            template="plotly_white",
            height=800,
            showlegend=True
        )
        
        # Achsen-Updates
        fig.update_yaxes(title_text="Signale pro 100 Bars", row=1, col=1)
        fig.update_yaxes(title_text="Hit Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="Hit Rate (%)", row=1, col=3)
        fig.update_yaxes(title_text="Anzahl Signale", row=2, col=1)
        fig.update_xaxes(title_text="Hit Rate (%)", row=2, col=2)
        fig.update_yaxes(title_text="Avg Return (%)", row=2, col=2)
        
        return fig
    
    def run_complete_multi_asset_analysis(self, optimize_parameters: bool = True,
                                        max_workers: int = 1) -> Dict:
        """
        Führt komplette Multi-Asset Analyse durch
        
        Args:
            optimize_parameters: Ob Parameter für jedes Asset optimiert werden sollen
            max_workers: Anzahl paralleler Prozesse
            
        Returns:
            Dictionary mit allen Ergebnissen
        """
        logger.info("🚀 STARTE KOMPLETTE MULTI-ASSET ANALYSE")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Lade Assets interaktiv falls noch keine geladen
        if not self.assets:
            loaded_count = self.load_multiple_assets_interactive()
            if loaded_count == 0:
                logger.error("❌ Keine Assets geladen")
                return {}
        
        # Führe Workflow für jedes Asset durch
        successful_analyses = 0
        
        for asset_name in self.assets.keys():
            logger.info(f"\n📊 Verarbeite {asset_name}...")
            
            if self.run_asset_workflow(asset_name, optimize_parameters, max_workers):
                successful_analyses += 1
            else:
                logger.warning(f"⚠️ Workflow für {asset_name} fehlgeschlagen")
        
        if successful_analyses == 0:
            logger.error("❌ Keine erfolgreichen Asset-Analysen")
            return {}
        
        # Cross-Asset Vergleichsanalyse
        comparison_results = self.create_cross_asset_comparison()
        
        # Dashboard erstellen
        dashboard = self.create_multi_asset_dashboard()
        
        elapsed_time = time.time() - start_time
        
        # Sammle alle Ergebnisse
        complete_results = {
            'assets_analyzed': list(self.assets.keys()),
            'successful_analyses': successful_analyses,
            'analysis_results': self.analysis_results,
            'optimization_results': self.optimization_results,
            'comparison_results': comparison_results,
            'dashboard': dashboard,
            'execution_time': elapsed_time,
            'parameters': {
                'optimize_parameters': optimize_parameters,
                'max_workers': max_workers
            }
        }
        
        # Zusammenfassung ausgeben
        logger.info("="*80)
        logger.info("🎉 MULTI-ASSET ANALYSE ABGESCHLOSSEN!")
        logger.info(f"⏱️  Gesamtzeit: {elapsed_time:.1f} Sekunden")
        logger.info(f"📊 Assets analysiert: {successful_analyses}/{len(self.assets)}")
        
        for asset_name in self.assets.keys():
            if asset_name in self.analysis_results:
                stats = self.analysis_results[asset_name]['divergence_statistics']
                logger.info(f"   {asset_name}: {stats['Total Signals']} Signale ({stats['Signal Density']:.1f}/100)")
        
        logger.info("📈 Dashboard im Browser geöffnet")
        logger.info("="*80)
        
        return complete_results
    
    def export_complete_analysis(self, filename: str = None) -> str:
        """
        Exportiert komplette Multi-Asset Analyse nach Excel
        
        Args:
            filename: Zieldatei (optional)
            
        Returns:
            Pfad zur erstellten Datei
        """
        if not self.analysis_results:
            logger.error("Keine Analyseergebnisse zum Exportieren vorhanden")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"multi_asset_divergence_analysis_{timestamp}.xlsx"
        
        logger.info(f"📋 Exportiere Multi-Asset Analyse: {filename}")
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary Sheet
                summary_data = []
                summary_data.append(['MULTI-ASSET DIVERGENZ ANALYSE', ''])
                summary_data.append(['Analysiert am', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                summary_data.append(['', ''])
                
                for asset_name, result in self.analysis_results.items():
                    stats = result['divergence_statistics']
                    summary_data.append([f'{asset_name} - Total Signale', stats['Total Signals']])
                    summary_data.append([f'{asset_name} - Signal Density', f"{stats['Signal Density']:.1f}/100"])
                    summary_data.append([f'{asset_name} - Classic Bullish', stats.get('Classic Bullish', 0)])
                    summary_data.append([f'{asset_name} - Hidden Bullish', stats.get('Hidden Bullish', 0)])
                    summary_data.append([f'{asset_name} - Classic Bearish', stats.get('Classic Bearish', 0)])
                    summary_data.append(['', ''])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Asset-spezifische Sheets
                for asset_name, result in self.analysis_results.items():
                    # Performance-Daten pro Asset
                    perf_data = []
                    for div_type, div_result in result['performance_analysis'].items():
                        if div_type != 'comparison' and 'stats' in div_result:
                            stats = div_result['stats']
                            perf_data.append({
                                'Divergence_Type': stats['description'],
                                'Total_Signals': stats['total_signals'],
                                'Hit_Rate_30d': f"{stats.get('hit_rate_30d', 0):.1f}%",
                                'Avg_Return_30d': f"{stats.get('avg_return_30d', 0):+.2f}%",
                                'Risk_Reward_30d': f"{stats.get('avg_risk_reward_30d', 0):.2f}"
                            })
                    
                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        sheet_name = asset_name.replace('/', '_')[:31]  # Excel limit
                        perf_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Vergleichsanalyse
                if self.comparison_results:
                    # Asset Summary
                    asset_summary = []
                    for asset_name, summary in self.comparison_results['asset_summary'].items():
                        asset_summary.append({
                            'Asset': asset_name,
                            'Total_Signals': summary['total_signals'],
                            'Signal_Density': f"{summary['signal_density']:.1f}",
                            'Classic_Bullish': summary['classic_bullish'],
                            'Hidden_Bullish': summary['hidden_bullish'],
                            'Classic_Bearish': summary['classic_bearish'],
                            'Data_Points': summary['data_points']
                        })
                    
                    asset_summary_df = pd.DataFrame(asset_summary)
                    asset_summary_df.to_excel(writer, sheet_name='Asset_Comparison', index=False)
                    
                    # Performance Rankings
                    for metric, ranking in self.comparison_results['performance_ranking'].items():
                        ranking_data = []
                        for i, (asset, value, weight) in enumerate(ranking, 1):
                            ranking_data.append({
                                'Rank': i,
                                'Asset': asset,
                                'Value': f"{value:.2f}",
                                'Weight_Signals': weight
                            })
                        
                        if ranking_data:
                            ranking_df = pd.DataFrame(ranking_data)
                            sheet_name = f'Ranking_{metric}'[:31]
                            ranking_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"✅ Multi-Asset Analyse exportiert: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Excel-Export: {e}")
            return ""
    
    def print_multi_asset_summary(self):
        """
        Druckt Multi-Asset Zusammenfassung in die Konsole
        """
        if not self.analysis_results:
            logger.error("Keine Analyseergebnisse vorhanden")
            return
        
        print(f"\n" + "="*100"""
Multi-Asset Analyzer - Systematischer Workflow für Asset-übergreifende Analyse
Analysiert verschiedene Assets und vergleicht Divergenz-Performance
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog, messagebox

# Unsere Module
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from simple_enhanced_runner import add_hidden_bullish_divergences, add_bearish_divergences
from performance_analyzer import DivergencePerformanceAnalyzer
from sensitivity_parameter_optimizer import ParameterOptimizer

logger = logging.getLogger(__name__)

class MultiAssetDivergenceAnalyzer:
    """
    Führt systematische Divergenz-Analyse über mehrere Assets durch
    """
    
    def __init__(self):
        """
        Initialisierung des Multi-Asset Analyzers
        """
        self.assets = {}  # asset_name -> DataFrame
        self.analysis_results = {}  # asset_name -> results
        self.optimization_results = {}  # asset_name -> optimization
        self.comparison_results = {}
        
        # Standard-Assets und ihre typischen Dateinamen
        self.common_assets = {
            'Bitcoin': ['btc', 'bitcoin', 'BTC'],
            'Ethereum': ['eth', 'ethereum', 'ETH'],
            'SP500': ['sp500', 'spx', 'spy'],
            'Gold': ['gold', 'xau', 'GOLD'],
            'EUR/USD': ['eurusd', 'EUR', 'forex'],
            'Oil': ['oil', 'crude', 'wti']
        }
        
        logger.info("Multi-Asset Analyzer initialisiert")
    
    def detect_asset_type(self, filename: str) -> str:
        """
        Erkennt Asset-Typ basierend auf Dateinamen
        
        Args:
            filename: Name der Datei
            
        Returns:
            Erkannter Asset-Name oder 'Unknown'
        """
        filename_lower = filename.lower()
        
        for asset_name, keywords in self.common_assets.items():
            for keyword in keywords:
                if keyword.lower() in filename_lower:
                    return asset_name
        
        # Fallback: Verwende Dateinamen ohne Erweiterung
        return Path(filename).stem.upper()
    
    def load_asset_data(self, file_path: str, asset_name: str = None) -> bool:
        """
        Lädt Asset-Daten und bereitet sie vor
        
        Args:
            file_path: Pfad zur Datendatei
            asset_name: Name des Assets (optional, wird automatisch erkannt)
            
        Returns:
            True wenn erfolgreich geladen
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Datei nicht gefunden: {file_path}")
                return False
            
            # Asset-Name erkennen falls nicht angegeben
            if asset_name is None:
                asset_name = self.detect_asset_type(file_path.name)
            
            logger.info(f"📊 Lade {asset_name} Daten: {file_path.name}")
            
            # Daten laden basierend auf Dateierweiterung
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"Nicht unterstütztes Dateiformat: {file_path.suffix}")
                return False
            
            # Validierung
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Fehlende Spalten in {asset_name}: {missing_columns}")
                return False
            
            # Datum konvertieren und sortieren
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Asset speichern
            self.assets[asset_name] = df
            
            logger.info(f"✅ {