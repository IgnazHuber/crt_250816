import logging
import os
import sys
from server import app

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Prüft, ob alle erforderlichen Dateien vorhanden sind."""
    required_files = [
        "server.py",
        "analysis_engine.py",
        "config.py",
        "Initialize_RSI_EMA_MACD.py",
        "Local_Maximas_Minimas.py",
        "CBullDivg_Analysis_vectorized.py",
        "DivergenceArrows.py"
    ]
    
    logger.info("============================================================")
    logger.info("🚀 BULLISH DIVERGENCE ANALYZER - Modular Edition")
    logger.info("============================================================")
    logger.info("🔍 Prüfe Abhängigkeiten...")
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file}")
        else:
            if file == "DivergenceArrows.py":
                logger.warning(f"⚠️ {file} fehlt (optional, verwende Fallback für Annotationen)")
            else:
                logger.error(f"❌ {file} fehlt")
                sys.exit(1)
    
    logger.info("✨ Features:")
    logger.info("    • Modulare Architektur")
    logger.info("    • Dynamische Marker-Verwaltung")
    logger.info("    • Export (CSV/JSON)")
    logger.info("    • Varianten speichern/laden")
    logger.info("    • Performance-Metriken")
    logger.info("    • EMA 20, 50, 100, 200")
    logger.info("    • Y-Achsen Zoom")
    logger.info("    • Vergleich zur Basis-Variante")
    logger.info("============================================================")

def main():
    check_dependencies()
    try:
        # Starte Flask-App direkt
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()