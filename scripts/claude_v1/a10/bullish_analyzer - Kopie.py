#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 BULLISH DIVERGENCE ANALYZER - Modular Edition
Hauptstartdatei für den modularen Bullish Divergence Analyzer
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Prüft alle erforderlichen Module"""
    required_files = [
        'server.py',
        'analysis_engine.py', 
        'config.py',
        'Initialize_RSI_EMA_MACD.py',
        'Local_Maximas_Minimas.py',
        'CBullDivg_Analysis_vectorized.py'
    ]
    
    print("🔍 Prüfe Abhängigkeiten...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"💥 Fehlende Dateien: {', '.join(missing_files)}")
        return False
    
    return True

def show_features():
    """Zeigt verfügbare Features"""
    print("✨ Features:")
    print("   • Modulare Architektur")
    print("   • Dynamische Marker-Verwaltung") 
    print("   • Export (CSV/JSON)")
    print("   • Varianten speichern/laden")
    print("   • Performance-Metriken")
    print("   • EMA 20, 50, 100, 200")
    print("   • Y-Achsen Zoom")
    print("   • Vergleich zur Basis-Variante")

def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("🚀 BULLISH DIVERGENCE ANALYZER - Modular Edition")
    print("=" * 60)
    
    # Abhängigkeiten prüfen
    if not check_dependencies():
        sys.exit(1)
    
    show_features()
    print("=" * 60)
    
    # Server starten
    try:
        from server import start_server
        start_server()
    except KeyboardInterrupt:
        print("\n👋 Server beendet")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()