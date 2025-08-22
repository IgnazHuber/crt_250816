#!/usr/bin/env python3
"""
Schneller Test ob alles bereit ist
"""

import sys
from pathlib import Path

print("="*50)
print("🔍 QUICK CHECK - Bullish Divergence Analyzer")
print("="*50)

# 1. Python Version
print(f"\n✅ Python {sys.version.split()[0]}")

# 2. Benötigte Dateien prüfen
files_needed = {
    'app.py': 'Flask Server',
    'static/index.html': 'Web Interface',
    'Initialize_RSI_EMA_MACD.py': 'RSI/MACD Modul',
    'Local_Maximas_Minimas.py': 'Extrema Modul',
    'CBullDivg_Analysis_vectorized.py': 'Divergenz Modul'
}

print("\n📁 Dateien:")
all_ok = True
for file, desc in files_needed.items():
    if Path(file).exists():
        print(f"  ✅ {file:40} - {desc}")
    else:
        print(f"  ❌ {file:40} - FEHLT! ({desc})")
        all_ok = False

# 3. Pakete prüfen
print("\n📦 Pakete:")
packages = ['flask', 'flask_cors', 'pandas', 'numpy']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} - Installiere mit: pip install {pkg}")
        all_ok = False

# 4. Ergebnis
print("\n" + "="*50)
if all_ok:
    print("✅ ALLES BEREIT!")
    print("\nStarte den Server mit:")
    print("  python app.py")
    print("\nÖffne dann im Browser:")
    print("  http://localhost:5000")
else:
    print("⚠️  EINIGE KOMPONENTEN FEHLEN")
    print("\nBitte prüfe die fehlenden Dateien/Pakete oben!")

print("="*50)