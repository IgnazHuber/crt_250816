"""
Workflow Orchestrator - Hauptworkflow für komplette Divergenz-Analyse
Koordiniert alle Module und bietet verschiedene Analyse-Modi
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import messagebox, simpledialog
import time

# Unsere Module
from performance_analyzer import DivergencePerformanceAnalyzer
from sensitivity_parameter_optimizer import ParameterOptimizer
from multi_asset_analyzer import MultiAssetDivergenceAnalyzer
from quick_enhanced_test import main as run_quick_test

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'divergence_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DivergenceAnalysisWorkflow:
    """
    Hauptklasse für orchestrierte Divergenz-Analyse
    """
    
    def __init__(self):
        """
        Initialisierung des Workflow Orchestrators
        """
        self.analysis_modes = {
            '1': {
                'name': 'Quick Test',
                'description': 'Schneller Test mit einem Asset (funktioniert garantiert)',
                'function': self.run_quick_analysis
            },
            '2': {
                'name': 'Single Asset Analysis',
                'description': 'Detaillierte Analyse mit Performance-Validierung',
                'function': self.run_single_asset_analysis
            },
            '3': {
                'name': 'Parameter Optimization',
                'description': 'Parameter-Optimierung für ein Asset',
                'function': self.run_parameter_optimization
            },
            '4': {
                'name': 'Multi-Asset Comparison',
                'description': 'Vergleichsanalyse mehrerer Assets',
                'function': self.run_multi_asset_analysis
            },
            '5': {
                'name': 'Complete Analysis',
                'description': 'Komplette Analyse: Optimierung + Validierung + Vergleich',
                'function': self.run_complete_analysis
            }
        }
        
        logger.info("Divergence Analysis Workflow initialisiert")
    
    def show_analysis_menu(self) -> str:
        """
        Zeigt Analyse-Menü und gibt Auswahl zurück
        
        Returns:
            Gewählter Modus oder 'quit'
        """
        print(f"\n" + "="*80)
        print("🚀 DIVERGENZ ANALYSE WORKFLOW")
        print("="*80)
        print("\nVerfügbare Analyse-Modi:")
        print("-" * 40)
        
        for mode_id, mode_info in self.analysis_modes.items():
            print(f"{mode_id}. {mode_info['name']}")
            print(f"   {mode_info['description']}")
            print()
        
        print("q. Beenden")
        print("-" * 40)
        
        while True:
            choice = input("\nWähle einen Modus (1-5 oder q): ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                return 'quit'
            elif choice in self.analysis_modes:
                return choice
            else:
                print("❌ Ungültige Auswahl. Bitte wähle 1-5 oder q.")
    
    def run_quick_analysis(self) -> bool:
        """
        Führt schnelle Analyse durch (quick_enhanced_test.py)
        
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info("🚀 Starte Quick Analysis...")
            
            print("\n" + "="*60)
            print("📊 QUICK ANALYSIS")
            print("="*60)
            print("Diese Analyse ist schnell und funktioniert garantiert.")
            print("Du bekommst alle Divergenz-Typen in einem interaktiven Chart.")
            print("-" * 60)
            
            # Führe Quick Test aus
            run_quick_test()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Quick Analysis: {e}")
            return False
    
    def run_single_asset_analysis(self) -> bool:
        """
        Führt detaillierte Single-Asset Analyse durch
        
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info("🚀 Starte Single Asset Analysis...")
            
            print("\n" + "="*60)
            print("📈 SINGLE ASSET ANALYSIS")
            print("="*60)
            print("Detaillierte Analyse mit Performance-Validierung.")
            print("Zeigt wie gut die Divergenzen wirklich funktionieren.")
            print("-" * 60)
            
            # Multi-Asset Analyzer für einzelnes Asset verwenden
            analyzer = MultiAssetDivergenceAnalyzer()
            
            # Einzelnes Asset laden
            if analyzer.load_multiple_assets_interactive() == 0:
                logger.error("❌ Kein Asset geladen")
                return False
            
            # Erstes (und einziges) Asset analysieren
            asset_name = list(analyzer.assets.keys())[0]
            
            # Workflow für einzelnes Asset
            success = analyzer.run_asset_workflow(
                asset_name, 
                optimize_parameters=False,  # Erstmal ohne Optimierung
                max_workers=1
            )
            
            if success:
                # Performance-Analyse anzeigen
                if asset_name in analyzer.analysis_results:
                    result = analyzer.analysis_results[asset_name]
                    
                    # Performance Dashboard erstellen und anzeigen
                    if 'performance_analysis' in result:
                        perf_analyzer = DivergencePerformanceAnalyzer(result['data'])
                        perf_analyzer.performance_results = result['performance_analysis']
                        
                        # Dashboard anzeigen
                        dashboard = perf_analyzer.create_performance_dashboard()
                        dashboard.show()
                        
                        # Zusammenfassung ausgeben
                        perf_analyzer.print_performance_summary()
                        
                        # Excel Export
                        excel_file = perf_analyzer.export_performance_analysis()
                        logger.info(f"📊 Performance-Analyse exportiert: {excel_file}")
                
                logger.info("✅ Single Asset Analysis erfolgreich")
                return True
            else:
                logger.error("❌ Single Asset Analysis fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Single Asset Analysis: {e}")
            return False
    
    def run_parameter_optimization(self) -> bool:
        """
        Führt Parameter-Optimierung durch
        
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info("🚀 Starte Parameter Optimization...")
            
            print("\n" + "="*60)
            print("🎯 PARAMETER OPTIMIZATION")
            print("="*60)
            print("Findet die optimalen Parameter für maximale Performance.")
            print("⚠️  WARNUNG: Kann 5-15 Minuten dauern!")
            print("-" * 60)
            
            # Bestätigung vom Benutzer
            confirm = input("Möchtest du die Parameter-Optimierung starten? (j/n): ").strip().lower()
            if confirm not in ['j', 'ja', 'y', 'yes']:
                print("❌ Parameter-Optimierung abgebrochen")
                return False
            
            # Asset laden
            analyzer = MultiAssetDivergenceAnalyzer()
            if analyzer.load_multiple_assets_interactive() == 0:
                logger.error("❌ Kein Asset geladen")
                return False
            
            asset_name = list(analyzer.assets.keys())[0]
            
            # Indikatoren vorbereiten
            if not analyzer.prepare_asset_indicators(asset_name):
                return False
            
            # Parameter-Optimierung durchführen
            optimization_result = analyzer.optimize_asset_parameters(asset_name, max_workers=1)
            
            if optimization_result:
                # Ergebnisse anzeigen
                if 'dashboard' in optimization_result:
                    optimization_result['dashboard'].show()
                
                # Zusammenfassung ausgeben
                optimizer = ParameterOptimizer(analyzer.assets[asset_name], asset_name)
                optimizer.optimization_results = optimization_result
                optimizer.print_optimization_summary()
                
                # Excel Export
                excel_file = optimizer.export_optimization_results()
                logger.info(f"📊 Parameter-Optimierung exportiert: {excel_file}")
                
                logger.info("✅ Parameter Optimization erfolgreich")
                return True
            else:
                logger.error("❌ Parameter Optimization fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Parameter Optimization: {e}")
            return False
    
    def run_multi_asset_analysis(self) -> bool:
        """
        Führt Multi-Asset Vergleichsanalyse durch
        
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info("🚀 Starte Multi-Asset Analysis...")
            
            print("\n" + "="*60)
            print("🌍 MULTI-ASSET ANALYSIS")
            print("="*60)
            print("Vergleicht Divergenz-Performance zwischen verschiedenen Assets.")
            print("Empfohlene Assets: BTC, ETH, SP500, Gold, EUR/USD")
            print("-" * 60)
            
            # Multi-Asset Analyzer
            analyzer = MultiAssetDivergenceAnalyzer()
            
            # Mehrere Assets laden
            loaded_count = analyzer.load_multiple_assets_interactive()
            if loaded_count < 2:
                logger.error("❌ Mindestens 2 Assets für Vergleich erforderlich")
                return False
            
            # Parameter-Optimierung optional
            optimize = input("Sollen Parameter für jedes Asset optimiert werden? (j/n) [Dauert länger]: ").strip().lower()
            optimize_parameters = optimize in ['j', 'ja', 'y', 'yes']
            
            # Komplette Multi-Asset Analyse
            results = analyzer.run_complete_multi_asset_analysis(
                optimize_parameters=optimize_parameters,
                max_workers=1
            )
            
            if results:
                # Dashboard anzeigen
                if 'dashboard' in results:
                    results['dashboard'].show()
                
                # Zusammenfassung ausgeben
                analyzer.print_multi_asset_summary()
                
                # Excel Export
                excel_file = analyzer.export_complete_analysis()
                if excel_file:
                    logger.info(f"📊 Multi-Asset Analyse exportiert: {excel_file}")
                
                logger.info("✅ Multi-Asset Analysis erfolgreich")
                return True
            else:
                logger.error("❌ Multi-Asset Analysis fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Multi-Asset Analysis: {e}")
            return False
    
    def run_complete_analysis(self) -> bool:
        """
        Führt komplette Analyse durch (alles zusammen)
        
        Returns:
            True wenn erfolgreich
        """
        try:
            logger.info("🚀 Starte Complete Analysis...")
            
            print("\n" + "="*60)
            print("🎯 COMPLETE ANALYSIS")
            print("="*60)
            print("Führt ALLES durch:")
            print("• Parameter-Optimierung für jedes Asset")
            print("• Performance-Validierung aller Divergenz-Typen")
            print("• Asset-übergreifende Vergleichsanalyse")
            print("• Umfassende Excel-Reports")
            print()
            print("⚠️  WARNUNG: Kann 15-30 Minuten dauern bei mehreren Assets!")
            print("-" * 60)
            
            # Bestätigung
            confirm = input("Möchtest du die komplette Analyse starten? (j/n): ").strip().lower()
            if confirm not in ['j', 'ja', 'y', 'yes']:
                print("❌ Komplette Analyse abgebrochen")
                return False
            
            start_time = time.time()
            
            # Multi-Asset Analyzer mit allen Features
            analyzer = MultiAssetDivergenceAnalyzer()
            
            # Assets laden
            loaded_count = analyzer.load_multiple_assets_interactive()
            if loaded_count == 0:
                logger.error("❌ Keine Assets geladen")
                return False
            
            print(f"\n🎯 Starte komplette Analyse für {loaded_count} Assets...")
            
            # Komplette Analyse mit Parameter-Optimierung
            results = analyzer.run_complete_multi_asset_analysis(
                optimize_parameters=True,   # Optimiere Parameter für alle Assets
                max_workers=1              # Sequenziell für Stabilität
            )
            
            if results:
                elapsed_time = time.time() - start_time
                
                # Alle Dashboards anzeigen
                if 'dashboard' in results:
                    results['dashboard'].show()
                
                # Zusätzlich: Optimierungs-Dashboards für jedes Asset
                for asset_name, opt_result in analyzer.optimization_results.items():
                    if 'dashboard' in opt_result:
                        print(f"📊 Zeige Parameter-Optimierung für {asset_name}...")
                        opt_result['dashboard'].show()
                
                # Zusammenfassungen ausgeben
                analyzer.print_multi_asset_summary()
                
                # Excel-Exports
                excel_file = analyzer.export_complete_analysis()
                if excel_file:
                    logger.info(f"📊 Multi-Asset Analyse exportiert: {excel_file}")
                
                # Individuelle Parameter-Optimierung Exports
                for asset_name in analyzer.optimization_results.keys():
                    if asset_name in analyzer.optimization_results:
                        optimizer = ParameterOptimizer(analyzer.assets[asset_name], asset_name)
                        optimizer.optimization_results = analyzer.optimization_results[asset_name]
                        opt_excel = optimizer.export_optimization_results()
                        logger.info(f"📊 Parameter-Optimierung für {asset_name} exportiert: {opt_excel}")
                
                print(f"\n" + "="*80)
                print("🎉 KOMPLETTE ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
                print(f"⏱️  Gesamtzeit: {elapsed_time:.1f} Sekunden ({elapsed_time/60:.1f} Minuten)")
                print(f"📊 Assets analysiert: {loaded_count}")
                print("📈 Alle Dashboards wurden im Browser geöffnet")
                print("📋 Alle Excel-Dateien wurden erstellt")
                print("="*80)
                
                logger.info("✅ Complete Analysis erfolgreich")
                return True
            else:
                logger.error("❌ Complete Analysis fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Complete Analysis: {e}")
            return False
    
    def run_workflow(self):
        """
        Hauptworkflow - zeigt Menü und führt gewählte Analyse durch
        """
        print("🚀 Willkommen beim Divergenz Analyse Workflow!")
        print("Dieser Workflow führt dich durch verschiedene Analyse-Modi.")
        
        while True:
            try:
                # Zeige Menü und hole Auswahl
                choice = self.show_analysis_menu()
                
                if choice == 'quit':
                    print("\n👋 Auf Wiedersehen!")
                    break
                
                # Führe gewählte Analyse durch
                mode_info = self.analysis_modes[choice]
                
                print(f"\n🚀 Starte {mode_info['name']}...")
                print(f"📄 {mode_info['description']}")
                
                start_time = time.time()
                success = mode_info['function']()
                elapsed_time = time.time() - start_time
                
                if success:
                    print(f"\n✅ {mode_info['name']} erfolgreich abgeschlossen!")
                    print(f"⏱️  Zeit: {elapsed_time:.1f} Sekunden")
                else:
                    print(f"\n❌ {mode_info['name']} fehlgeschlagen!")
                
                # Frage ob weiter
                continue_choice = input("\nMöchtest du eine weitere Analyse durchführen? (j/n): ").strip().lower()
                if continue_choice not in ['j', 'ja', 'y', 'yes']:
                    print("\n👋 Auf Wiedersehen!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n⏹️ Workflow vom Benutzer abgebrochen")
                break
            except Exception as e:
                logger.error(f"❌ Unerwarteter Fehler im Workflow: {e}")
                print(f"\n💥 Unerwarteter Fehler: {e}")
                
                continue_choice = input("Möchtest du trotzdem fortfahren? (j/n): ").strip().lower()
                if continue_choice not in ['j', 'ja', 'y', 'yes']:
                    break


def main():
    """
    Hauptfunktion für Workflow Orchestrator
    """
    try:
        # Workflow erstellen und starten
        workflow = DivergenceAnalysisWorkflow()
        workflow.run_workflow()
        
    except Exception as e:
        logger.error(f"❌ Kritischer Fehler: {e}")
        print(f"\n💥 Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
