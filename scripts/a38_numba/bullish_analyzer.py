import os
import sys
import logging
from pathlib import Path
from typing import List

# Wichtig: Wir verwenden die bestehende Flask-App aus server.py
try:
    from server import app  # server.py must provide Flask instance `app`
except ImportError as e:
    print(f"❌ Could not import server.app: {e}", file=sys.stderr)
    print("Make sure server.py exists and exports 'app'", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error importing server.app: {e}", file=sys.stderr)
    raise

# Optionale Flask-Imports nur für Fallbacks (werden unten ggf. genutzt)
from flask import jsonify, send_from_directory

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Abhängigkeiten prüfen (gleicher Ordner wie dieses Skript)
# --------------------------------------------------------------------
def check_dependencies() -> None:
    base = Path(__file__).resolve().parent

    required_files: List[str] = [
        "server.py",
        "data_processor_v2.py",  # Fast loading module
        "Initialize_RSI_EMA_MACD.py",
        "Local_Maximas_Minimas.py",
        "CBullDivg_Analysis_vectorized.py",
    ]
    
    optional_files: List[str] = [
        "analysis_engine.py",
        "config.py", 
        "DivergenceArrows.py",
    ]

    logger.info("============================================================")
    logger.info("🚀 BULLISH DIVERGENCE ANALYZER - Modular Edition")
    logger.info("============================================================")
    logger.info("🔍 Checking dependencies...")

    missing = False
    for fname in required_files:
        file_path = base / fname
        if file_path.exists():
            logger.info(f"✅ {fname}")
        else:
            # Case-insensitive check for Windows compatibility
            found = False
            for existing_file in base.glob(f"{fname.lower()}"):
                if existing_file.name.lower() == fname.lower():
                    logger.info(f"✅ {existing_file.name} (case variant)")
                    found = True
                    break
            if not found:
                logger.error(f"❌ {fname} missing")
                missing = True
    
    # Check optional files (warning only)
    for fname in optional_files:
        file_path = base / fname
        if file_path.exists():
            logger.info(f"✅ {fname} (optional)")
        else:
            logger.warning(f"⚠️ {fname} (optional, not found)")

    # Check static files (non-critical, but log)
    static_dir = base / "static"
    if static_dir.exists():
        for fname in ("index.html", "app.js", "style.css"):
            p = static_dir / fname
            if p.exists():
                logger.info(f"✅ static/{fname}")
            else:
                logger.warning(f"⚠️ static/{fname} not found")
    else:
        logger.warning("⚠️ static/ directory not found")

    logger.info("✨ Features:")
    logger.info("    • Modular architecture")
    logger.info("    • Fast Polars-based data loading (2-10x faster)")
    logger.info("    • File explorer upload workflow")
    logger.info("    • Dynamic marker management")
    logger.info("    • Export (CSV/JSON)")
    logger.info("    • Save/load variants")
    logger.info("    • Performance metrics")
    logger.info("    • EMA 20, 50, 100, 200")
    logger.info("    • Y-axis zoom")
    logger.info("    • Compare to base variant")
    logger.info("============================================================")

    if missing:
        logger.error("❌ Missing required dependencies. Exiting.")
        sys.exit(1)

# --------------------------------------------------------------------
# App-Konfiguration (STATIC) & Fallback-Routen nur wenn nötig
# --------------------------------------------------------------------
def configure_app_and_fallbacks() -> None:
    """
    - Sets the static folder correctly.
    - Adds fallbacks for "/" and "/api/health" only if not present.
      This preserves existing routes from server.py.
    """
    base = Path(__file__).resolve().parent
    static_dir = base / "static"

    # Ensure Flask knows the correct static path
    if static_dir.exists():
        app.static_folder = str(static_dir)
    else:
        logger.warning("Static directory not found, using default")

    # Get existing routes to avoid conflicts
    existing_routes = {rule.rule for rule in app.url_map.iter_rules()}
    
    # Add root route fallback if not present
    if "/" not in existing_routes:
        logger.info("ℹ️ Root route ('/') not found - enabling static fallback to static/index.html")

        @app.route("/", methods=["GET"])
        def _index_fallback():
            try:
                index_path = static_dir / "index.html"
                if index_path.exists():
                    return send_from_directory(app.static_folder, "index.html")
                return jsonify({"success": False, "error": "index.html not found"}), 404
            except Exception as e:
                logger.error(f"Error serving index: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

    # Add health endpoint fallback if not present  
    if "/api/health" not in existing_routes:
        logger.info("ℹ️ /api/health not found - registering health check fallback")

        @app.route("/api/health", methods=["GET"])
        def _health_fallback():
            try:
                return jsonify({"success": True, "message": "Server running"}), 200
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

    # Note: No additional endpoints defined here to preserve existing API from server.py

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    try:
        check_dependencies()
        configure_app_and_fallbacks()
        logger.info("🚀 Starting Flask application...")
        # Important: we start the app defined in server.py
        app.run(debug=True, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
