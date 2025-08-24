import logging
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
import uuid

# Flask-Anwendung initialisieren
app = Flask(__name__)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Überprüfe Abhängigkeiten
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

# Route für die Root-Seite (index.html aus static/ ausliefern)
@app.route('/')
def index():
    """Route für die Root-Seite, gibt index.html aus dem static/ Verzeichnis zurück"""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'index.html')

# API-Endpunkt für den Health-Check
@app.route('/api/health', methods=['GET'])
def health_check():
    """API-Endpunkt zur Überprüfung der Servergesundheit"""
    try:
        return jsonify({'success': True, 'message': 'Server läuft'}), 200
    except Exception as e:
        logger.error(f"Fehler bei der Gesundheitsprüfung: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# API-Endpunkt für die Analyse
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API-Endpunkt für die Analyse"""
    try:
        data = request.json
        session_id = data.get('session_id')
        variants = data.get('variants')
        
        if not session_id or not variants:
            logger.error(f"Ungültige Daten empfangen: {data}")
            return jsonify({'success': False, 'error': 'Ungültige Daten'}), 400
        
        logger.info(f"Starte Analyse für Session {session_id} mit {len(variants)} Varianten")
        
        # Initialisiere das DataFrame für diese Session
        df = load_dataframe_for_session(session_id)
        if df is None:
            logger.error(f"Session {session_id} nicht gefunden oder DataFrame konnte nicht geladen werden")
            return jsonify({'success': False, 'error': 'Session nicht gefunden oder Fehler beim Laden der Daten'}), 400
        
        # Initialisiere technische Indikatoren
        df = initialize_technical_indicators(df)
        if df is None:
            logger.error(f"Fehler bei der Initialisierung der technischen Indikatoren")
            return jsonify({'success': False, 'error': 'Fehler bei der Initialisierung der technischen Indikatoren'}), 500
        
        results = {}

        for variant in variants:
            variant_id = str(variant.get("id", uuid.uuid4()))
            window = variant.get("window", 5)
            candle_tol = variant.get("candleTol", 0.1)
            macd_tol = variant.get("macdTol", 3.25)
            calculate_performance = variant.get("calculate_performance", True)
            
            logger.info(f"Analysiere: {variant.get('name', 'Unbenannt')}")
            
            df_variant = df.copy()
            df_variant = CBullDivg_analysis(df_variant, window, candle_tol, macd_tol)
            
            divergences = extract_divergences(df_variant, variant)
            
            results[variant_id] = {
                "classic": divergences.get("classic", []),
                "hidden": divergences.get("hidden", []),
                "total": len(divergences.get("classic", [])) + len(divergences.get("hidden", [])),
                "annotations": generate_annotations(df_variant, divergences, window, variant.get("name", "Unbenannt"))
            }
        
        chart_data = create_chart_data(df)
        
        logger.info("Analyse abgeschlossen")
        return jsonify({'success': True, 'chartData': chart_data, 'results': results}), 200

    except Exception as e:
        logger.error(f"Fehler bei der Analyse: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Platzhalter für das Laden eines DataFrames
def load_dataframe_for_session(session_id):
    """Simuliert das Laden eines DataFrames für eine gegebene Session-ID"""
    logger.info(f"Lade DataFrame für Session {session_id}")
    return pd.DataFrame()  # Hier sollte der tatsächliche DataFrame zurückgegeben werden

# Platzhalter für die Initialisierung technischer Indikatoren
def initialize_technical_indicators(df):
    """Simuliert die Initialisierung technischer Indikatoren"""
    logger.info("Initialisiere technische Indikatoren")
    return df

# Platzhalter für die CBullDivg-Analyse
def CBullDivg_analysis(df, window, candle_tol, macd_tol):
    """Platzhalter für die Analyse der Bullish Divergence"""
    logger.info(f"Führe CBullDivg-Analyse mit Fenster={window}, Kerzen-Toleranz={candle_tol}, MACD-Toleranz={macd_tol} durch")
    return df

# Main-Function zum Starten des Servers
def main():
    check_dependencies()
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
