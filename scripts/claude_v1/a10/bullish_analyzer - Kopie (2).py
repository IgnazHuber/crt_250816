#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 BULLISH DIVERGENCE ANALYZER - Modular Edition
Hauptstartdatei für den modularen Bullish Divergence Analyzer
"""

import os
import sys
import logging
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Prüft alle erforderlichen Module"""
    required_files = [
        'server.py',
        'analysis_engine.py', 
        'config.py',
        'Initialize_RSI_EMA_MACD.py',
        'Local_Maximas_Minimas.py',
        'CBullDivg_Analysis_vectorized.py',
        'DivergenceArrows.py'  # Optional hinzugefügt
    ]
    
    logger.info("🔍 Prüfe Abhängigkeiten...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file}")
        else:
            if file == 'DivergenceArrows.py':
                logger.warning(f"⚠️ {file} fehlt (optional, verwende Fallback für Annotationen)")
            else:
                logger.error(f"❌ {file}")
                missing_files.append(file)
    
    if missing_files:
        logger.error(f"💥 Fehlende erforderliche Dateien: {', '.join(missing_files)}")
        return False
    
    return True

def show_features():
    """Zeigt verfügbare Features"""
    logger.info("✨ Features:")
    logger.info("   • Modulare Architektur")
    logger.info("   • Dynamische Marker-Verwaltung") 
    logger.info("   • Export (CSV/JSON)")
    logger.info("   • Varianten speichern/laden")
    logger.info("   • Performance-Metriken")
    logger.info("   • EMA 20, 50, 100, 200")
    logger.info("   • Y-Achsen Zoom")
    logger.info("   • Vergleich zur Basis-Variante")

def main():
    """Hauptfunktion"""
    logger.info("=" * 60)
    logger.info("🚀 BULLISH DIVERGENCE ANALYZER - Modular Edition")
    logger.info("=" * 60)
    
    # Abhängigkeiten prüfen
    if not check_dependencies():
        logger.error("Nicht alle erforderlichen Abhängigkeiten vorhanden, Server wird nicht gestartet")
        sys.exit(1)
    
    show_features()
    logger.info("=" * 60)
    
    # Server starten
    try:
        from server import start_server
        start_server()
    except KeyboardInterrupt:
        logger.info("\n👋 Server beendet")
    except Exception as e:
        logger.error(f"❌ Fehler: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()