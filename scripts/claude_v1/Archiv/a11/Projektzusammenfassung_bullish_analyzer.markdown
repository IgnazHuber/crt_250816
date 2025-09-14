# Projektzusammenfassung: Crypto-Trading und technische Chartanalyse (Workflow mit bullish_analyzer.py)

## Kontext
Entwicklung eines Tools für die Analyse von Preisdaten (Parquet-Dateien) mit Fokus auf technische Indikatoren (EMA, RSI, MACD) und Divergenz-Detektion (klassisch und versteckt). Ziel ist eine skalierbare, performante Lösung für große Datenmengen (>1M Zeilen), die in einer interaktiven Web-UI visualisiert wird. Der Workflow mit `bullish_analyzer.py` ist eine alternative Implementierung, die schneller als der `app.py`-Workflow ist.

## Status
### Backend (bullish_analyzer.py)
- **Funktion**: Verarbeitet Parquet-Dateien (`date`, `open`, `high`, `low`, `close`), berechnet EMA, RSI, MACD und Divergenzen (bullish/klassisch, bearish/versteckt).
- **Technologien**: Vermutlich Polars (v1.32.3) oder Pandas, mit PyArrow. Details zur Implementierung fehlen.
- **Performance**: Schneller als `app.py`, genaue Zeiten fehlen (z. B. „0.5s für 1M Zeilen“ benötigt).
- **Unterschiede zu `app.py`**: Wahrscheinlich kein FastAPI-Server, möglicherweise lokales Skript oder direkte Datenübergabe an `app.js`.
- **Ausgabe**: Vermutlich JSON oder ähnliches für UI-Visualisierung.

### Frontend (index.html, app.js)
- **UI**: Läuft auf `http://localhost:8080` (via `python -m http.server 8080` in `C:\Projekte\crt_250816\scripts\claude_v1\static`).
- **Funktionen**:
  - Datei-Upload (erfolgreich via `/api/upload` in `app.py`).
  - Varianten hinzufügen (`basis`, `v1`, `v2`, etc.), aktuell nicht funktionsfähig.
  - Visualisierung: Candlestick-Charts, EMA-Linien (20, 50, 100, 200), RSI, MACD, Divergenz-Pfeile.
- **Farben**:
  - Basis: Hellgrün (#90EE90)
  - V1: Hellblau (#ADD8E6)
  - V2: Orange (#FFA500)
  - V3: Pink (#FF69B4)
  - V4: Gelb (#FFFF00)
  - Fallback: z. B. #FF6B6B
- **Markerpfeile**:
  - Additional: `arrow-up`, Position `low * 0.92`, Farbe `variant.color`, gelbe Umrandung (#FFFF00), durchgezogen (`width: 1`).
  - Missing: `arrow-down`, Position `low * 0.85`, Farbe `variant.color`, gelbe Umrandung, gepunktet (`dash: 'dot'`, `width: 1`).
  - Hover: „Neue Divergenz“ bzw. „Fehlende Divergenz“ mit `div_id`, `type`, `strength`, `low`, `y`.
- **Features**: Zoom, Checkboxen (Classic/Hidden), Hoverinfo, Legende (`itemspacing: 0.5`, `tracewidth: 1`).

### Aktuelle Probleme
- **Varianten hinzufügen**: Funktioniert nicht (UI reagiert nicht, keine Logs wie `✅ Variant added`).
- **Performance**: `bullish_analyzer.py` schneller als `app.py`, aber Details fehlen.
- **Integration**: Unklar, wie `bullish_analyzer.py` mit der UI interagiert.

### Vergleich bullish_analyzer.py vs. app.py
- **bullish_analyzer.py**: Schneller, möglicherweise durch weniger Overhead oder optimierte Algorithmen.
- **app.py**: Skalierbare REST-API, aber langsamer bei großen Dateien.

### Nächste Schritte
- **Debugging Varianten**:
  - Prüfe Browser-Konsole (F12) auf Fehler beim Hinzufügen.
  - Teile `app.js`-Code (Varianten-Logik) und `index.html`-Code (Eingabefeld/Button).
  - Prüfe Netzwerk-Tab: Antworten von `/api/upload` und `/api/analyze` (für `app.py`).
- **Performance-Vergleich**:
  - Miss Analysezeit für `bullish_analyzer.py` und `app.py` mit großer Parquet-Datei (>1M Zeilen).
  - Teile Dateigröße, Anzahl Varianten, Analysezeit.
- **Integration von bullish_analyzer.py**:
  - Kläre, wie `bullish_analyzer.py` ausgeführt wird und mit der UI interagiert.
  - Teile Code oder Beschreibung von `bullish_analyzer.py`.
- **Optimierungen**:
  - Verbessere `app.py` basierend auf `bullish_analyzer.py`.
  - Optional: Zeitraum-Filter, neue Indikatoren.

### Benötigte Informationen
- Konsolenausgabe (F12) beim Hinzufügen von Varianten.
- Relevanter Code: `app.js` (Varianten-Logik), `index.html` (Eingabefeld/Button), `bullish_analyzer.py` (Analyse-Logik).
- Performance: Analysezeit für `bullish_analyzer.py` und `app.py`, Dateigröße, Anzahl Varianten.
- Visuelle Beschreibung: z. B. „Button reagiert nicht“, „Farben fehlerhaft“.
- Workflow von `bullish_analyzer.py`: Ausführung, Interaktion mit UI.

## Anmerkungen
- **Erhaltene Features**: Streaming, Caching, Arrow, Divergenz-Detektion, Zoom, Checkboxen.
- **Umgebung**: Python 3.11.6, Polars 1.32.3, `POLARS_MAX_THREADS=8` (optional).
- **Ziel**: Schnelle, skalierbare Analyse mit interaktiver UI, Integration der Vorteile von `bullish_analyzer.py`.