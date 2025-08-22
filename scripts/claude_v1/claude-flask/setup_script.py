#!/usr/bin/env python3
"""
Setup und Start Script für Bullish Divergence Analyzer
Prüft Abhängigkeiten und startet den Server
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Prüft ob alle notwendigen Pakete installiert sind"""
    required_packages = {
        'flask': 'Flask Web Framework',
        'flask_cors': 'Flask CORS Support',
        'pandas': 'Datenanalyse',
        'numpy': 'Numerische Berechnungen',
        'pyarrow': 'Parquet Support (optional)'
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package:15} - {description}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package:15} - {description} FEHLT")
    
    return missing

def check_modules():
    """Prüft ob alle Python-Module vorhanden sind"""
    required_modules = [
        'Initialize_RSI_EMA_MACD.py',
        'Local_Maximas_Minimas.py',
        'CBullDivg_Analysis_vectorized.py'
    ]
    
    missing = []
    for module in required_modules:
        if Path(module).exists():
            print(f"✅ {module}")
        else:
            print(f"❌ {module} FEHLT")
            missing.append(module)
    
    return missing

def install_requirements(packages):
    """Installiert fehlende Pakete"""
    print("\n📦 Installiere fehlende Pakete...")
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("✅ Alle Pakete installiert!")

def setup_directories():
    """Erstellt notwendige Verzeichnisse"""
    dirs = ['static', 'data', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Verzeichnis '{dir_name}' bereit")

def save_html():
    """Speichert HTML in static Ordner"""
    # HTML sollte als index.html in static/ gespeichert werden
    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True)
    
    html_path = static_dir / 'index.html'
    if not html_path.exists():
        print("⚠️  Bitte speichere index.html im 'static' Ordner!")
        return False
    else:
        print(f"✅ HTML gefunden: {html_path}")
        return True

def main():
    """Hauptfunktion"""
    print("="*60)
    print("🚀 BULLISH DIVERGENCE ANALYZER - SETUP & START")
    print("="*60)
    
    # 1. Python-Version prüfen
    print(f"\n🐍 Python Version: {sys.version}")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ wird benötigt!")
        sys.exit(1)
    
    # 2. Pakete prüfen
    print("\n📋 Prüfe Pakete...")
    missing_packages = check_requirements()
    
    if missing_packages:
        response = input(f"\n❓ {len(missing_packages)} Pakete fehlen. Installieren? (j/n): ")
        if response.lower() == 'j':
            install_requirements(missing_packages)
        else:
            print("❌ Ohne die Pakete kann der Server nicht starten!")
            sys.exit(1)
    
    # 3. Module prüfen
    print("\n📋 Prüfe Analyse-Module...")
    missing_modules = check_modules()
    
    if missing_modules:
        print("\n❌ WICHTIG: Folgende Python-Module fehlen:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nBitte kopiere diese Module in das aktuelle Verzeichnis!")
        print("Diese Module sind Teil deines originalen Analyse-Codes.")
        response = input("\nTrotzdem fortfahren? (j/n): ")
        if response.lower() != 'j':
            sys.exit(1)
    
    # 4. Verzeichnisse erstellen
    print("\n📁 Erstelle Verzeichnisse...")
    setup_directories()
    
    # 5. HTML prüfen
    print("\n📄 Prüfe HTML...")
    if not save_html():
        print("\n⚠️  Speichere die index.html Datei im 'static' Ordner")
        print("   und starte dann dieses Script erneut.")
        sys.exit(1)
    
    # 6. Server starten
    print("\n" + "="*60)
    print("✅ SETUP ABGESCHLOSSEN - STARTE SERVER")
    print("="*60)
    
    # Prüfe ob app.py existiert
    if not Path('app.py').exists():
        print("❌ app.py nicht gefunden!")
        print("   Speichere den Flask-Server Code als 'app.py'")
        sys.exit(1)
    
    print("\n🌐 Server wird gestartet auf: http://localhost:5000")
    print("   Der Browser öffnet sich automatisch in 3 Sekunden...")
    print("\n   Zum Beenden: Strg+C drücken")
    print("="*60 + "\n")
    
    # Browser nach kurzer Verzögerung öffnen
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Flask Server starten
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n✅ Server gestoppt.")
    except Exception as e:
        print(f"\n❌ Fehler beim Start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()