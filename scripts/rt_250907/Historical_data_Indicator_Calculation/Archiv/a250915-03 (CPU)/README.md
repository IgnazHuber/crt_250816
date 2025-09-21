# Projektzusammenfassung

## Überblick
Dieses Projekt bereitet historische Kursdaten auf und berechnet diverse Indikatoren in drei weitgehend ähnlichen Skripten:
- `Mainframe_vectorized_daily_timeframe_inp.py`
- `Mainframe_wo_TL_vectorized_1hour_inp.py`
- `Mainframe_wo_TL_vectorized_4hour_inp.py`

Die Skripte wählen interaktiv die Eingabedatei (CSV), schreiben die Ausgaben in denselben Ordner wie die CSV, zeigen einen Live‑Fortschrittsbalken an, können Berechnungen sanft pausieren und später ohne doppelte Arbeit fortsetzen. I/O und Parallelisierung wurden optimiert.

## Architektur
- Einmaliges CSV‑Einlesen im Hauptprozess → ein gemeinsamer Parquet‑Snapshot (`temp_df.parquet`).
- Worker‑Prozesse laden den Snapshot einmalig und erzeugen pro Slice `head(i)` als In‑Memory‑DataFrame.
- Indikatoren/Analysen laufen pro Slice; pro Slice wird ein Parquet geschrieben.
- Fortschritt wird periodisch und beim Stop in `.state_<CSVBasisname>.json` gesichert.
- Sidecar‑JSONs (`output_*.json`) dienen als Fallback für Resume ohne State und werden nach sauberem Stopp/Ende gelöscht.

### Datenfluss (Mermaid)
```mermaid
flowchart LR
    A[CSV wählen (GUI)] --> B[Hauptprozess: CSV lesen]
    B --> C[temp_df.parquet schreiben]
    C --> D[Pool: Worker initialisieren]
    D --> E[Worker: temp_df.parquet laden]
    E --> F[Slice head(i) bilden]
    F --> G[Indikatoren/Analysen]
    G --> H[output_*.parquet + Sidecar JSON]
    H --> I[State speichern (periodisch)]
    I -->|Stop/Resume| J[.state_<CSV>.json]
```

### Datenfluss (ASCII, Fallback)
```
CSV (GUI)
  -> Main: read_csv
    -> temp_df.parquet
      -> Pool init
        -> Worker: read_parquet once
          -> for i in slices: df=head(i)
            -> indicators/analysis
              -> output_*.parquet + output_*.json
                -> periodic save .state_<CSV>.json
```

## Features
- Interaktive CSV‑Auswahl (GUI) mit „zuletzt verwendetem“ Ordner (persistiert in `.io_paths.json`).
- Ausgabe im selben Ordner wie die ausgewählte CSV.
- Live‑Fortschrittsbalken im Terminal (verbleibend und gesamt).
- Sanftes Pausieren/Beenden (Ctrl+C oder Kontroll‑Datei) und nahtloses Fortsetzen.
- Robuster Resume ohne State über Sidecar‑JSONs je Output.
- Einmaliges CSV‑Laden; Worker nutzen einen gemeinsamen Parquet‑Snapshot, kein per‑Slice CSV/Parquet‑I/O.
- BLAS‑Threads in Workern gedrosselt (Oversubscription vermeiden).
- Prozesse/Chunksize zur Laufzeit per CLI oder Umgebungsvariablen steuerbar.

## Abhängigkeiten
- Python: `pandas`, `numpy`, `pyarrow`, `scikit-learn` (für `KernelDensity`) sowie die projektspezifischen Module im Ordner.
- Windows GUI: Tkinter (falls vorhanden). Fallbacks auf native Windows‑Dialoge (IFileDialog/SHBrowseForFolder) bzw. PowerShell sind integriert.

## Nutzung
1) Skript starten (Beispiel):
```
python Mainframe_vectorized_daily_timeframe_inp.py --procs auto --chunksize 32
```
- GUI fragt die CSV ab; Ausgaben werden in denselben Ordner geschrieben.
- Bestätigungsdialog zeigt CSV‑ und Output‑Pfad.

2) Prozesse und Chunksize steuern (optional):
- CLI: `--procs N`, `--chunksize N` (oder `--procs auto`)
- ENV: `PROCESSES`/`PROC_COUNT`, `CHUNKSIZE`

Beispiele:
```
# Automatisch: CPU-1 Prozesse, Chunksize 16
python Mainframe_wo_TL_vectorized_1hour_inp.py --procs auto

# Manuell: 6 Prozesse, Chunksize 8
set PROCESSES=6
set CHUNKSIZE=8
python Mainframe_wo_TL_vectorized_4hour_inp.py
```

## Pause und Fortsetzen
- Pausieren (sanft):
  - Ctrl+C im Terminal, oder
  - Datei `.control_<CSVBasisname>.txt` in den CSV‑Ordner legen (Inhalt z. B. „pause“, „stop“, „abort“, „quit“).
- Fortsetzen:
  - Gleiches Skript erneut starten. Es liest `.state_<CSVBasisname>.json` (oder Sidecar‑JSONs) und setzt mit den fehlenden Slices fort.

### Beispiele
```
# Sanft pausieren (im Terminal):
# Ctrl+C drücken → laufende Tasks werden beendet, State gespeichert

# Pausieren via Datei (ohne Terminal):
echo pause > path\zur\csv\mappe\.control_MeinCSV.txt

# Fortsetzen:
python Mainframe_vectorized_daily_timeframe_inp.py --procs auto --chunksize 16
```

## Dateien und Artefakte
- Präferenzen (neben den Skripten):
  - `.io_paths.json` – merkt „zuletzt verwendete“ Ordner.
- Laufsteuerung/Zustand (im CSV‑Ordner):
  - `.state_<CSVBasisname>.json` – persistierter Fortschritt (bevorzugte Quelle beim Resume).
  - `.control_<CSVBasisname>.txt` – Steuersignal zum Pausieren/Beenden.
- Outputs (im CSV‑Ordner):
  - `output_<LastDate>.parquet` – Ergebnis pro Slice.
  - `output_<LastDate>.json` – Sidecar mit `slice_index` (wird nach sauberem Stopp/Ende automatisch aufgeräumt).
- Temporär (im CSV‑Ordner):
  - `temp_df.parquet` – gemeinsamer Snapshot für Worker (wird am Ende entfernt, wenn möglich).

## Fortschrittsanzeige
- Beispielausgabe:
```
[Remaining ##########------------------------------] 120/480 (25.0%) | Overall 320/1600 (20.0%)
```
- „Remaining“ zeigt den aktuellen Lauf (noch zu verarbeitende Slices), „Overall“ den Gesamtfortschritt relativ zum gesamten Bereich.

## Performance‑Hinweise
- Prozesse: `auto` nutzt `max(1, CPU-1)`. Manuell kann vorteilhaft sein (z. B. I/O‑Grenzen beachten).
- Chunksize: Größer = höherer Durchsatz, kleiner = schnellere Reaktion auf Stop.
- BLAS‑Threads: Werden pro Worker auf 1 gesetzt (`OMP/MKL/OPENBLAS/NUMEXPR`), um CPU‑Überbelegung zu vermeiden.
- Weitergehende Optimierungen (optional):
  - Inkrementelle Indikator‑Berechnung (Rolling‑State statt `head(i)`).
  - Arrow Dataset/Memory‑Mapping für zielgerichtete Teillesevorgänge.

## Bekannte Einschränkungen
- Die Heuristik „Outputs zählen“ greift nur, wenn keine State‑Datei existiert; ansonsten hat der State Vorrang.
- Falls Gruppenrichtlinien GUI‑Dialoge blockieren, kann die CSV‑Auswahl auf native Dialoge/PowerShell‑Fallback zurückfallen; in streng gehärteten Umgebungen ist manuelle Pfadsetzung erforderlich.

## Module intern

### Erwartetes Input‑Schema (DataFrame)
- Spalten: `date` (Datetime), `open`, `high`, `low`, `close`, idealerweise `volume`.
- `date` wird beim CSV‑Einlesen als Datum geparst (`parse_dates=['date']`).

### Indikator‑ und Analyse‑Bausteine
- `Initialize_RSI_EMA_MACD_vectorized.Initialize_RSI_EMA_MACD(df)`
  - Input: DataFrame (OHLC[+volume])
  - Wirkung: Fügt RSI-, EMA‑ und MACD‑Spalten hinzu (in‑place).
- `CS_Type.Candlestick_Type(df)`
  - Wirkung: Klassifiziert Kerzen (z. B. doji/hammer/engulfing) und schreibt Typ‑Spalten (in‑place).
- `Level_1_Maximas_Minimas.Level_1_Max_Min(df)`
  - Wirkung: Markiert lokale Extrempunkte (Minima/Maxima) der ersten Ebene (in‑place).
- Divergenzen (Close/High vs. Indikator):
  - `HBullDivg_analysis_vectorized.HBullDivg_analysis(df, sens, ratio)`
  - `HBearDivg_analysis_vectorized.HBearDivg_analysis(df, sens, ratio)`
  - `CBullDivg_analysis_vectorized.CBullDivg_analysis(df, sens, ratio)`
  - `CBearDivg_analysis_vectorized.CBearDivg_analysis(df, sens, ratio)`
  - `CBullDivg_x2_analysis_vectorized.CBullDivg_x2_analysis(df, sens, ratio)`
  - Wirkung: Markiert bullische/bärische Divergenzen; Schwellen/Empfindlichkeit über `sens`/`ratio` (in‑place).
- `Support_Resistance_vectorized.calculate_support_levels(df, lookback_years=..., pivot_threshold=...)`
  - Wirkung: Berechnet/annotiert Unterstützungs‑/Widerstandszonen; nutzt Pivots/Kernel‑Methoden; gibt `df` zurück.
- Trendlinien:
  - `Trendline_Up_Support_vectorized.calc_TL_Up_Support(df, ...)`
  - `Trendline_Up_Resistance_vectorized.calc_TL_Up_Resistance(df, ...)`
  - `Trendline_Down_Resistance_vectorized.calc_TL_Down_Resistance(df, ...)`
  - Wirkung: Zeichnet/annotiert aufwärts/abwärts gerichtete Unterstützungs‑/Widerstandslinien; gibt `df` zurück.
- `Goldenratio_vectorized.calculate_golden_ratios(df)`
  - Wirkung: Fügt Golden‑Ratio/Fibonacci‑Levels hinzu; gibt `df` zurück.

Hinweise:
- Viele Funktionen modifizieren `df` in‑place; einige geben das angereicherte `df` zusätzlich zurück. In der Pipeline wird beides unterstützt.
- Divergenz‑Parameter (`sens`, `ratio`) sind je Zeiteinheit unterschiedlich gesetzt (Daily vs. 1h/4h) und steuern die Empfindlichkeit/Bestätigung.

### Pipeline‑Ablauf (vereinfacht)
```python
Initialize_RSI_EMA_MACD(df)
Level_1_Max_Min(df)
Candlestick_Type(df)
CBullDivg_analysis(df, sens, ratio)
CBullDivg_x2_analysis(df, sens, ratio)
HBullDivg_analysis(df, sens, ratio)
CBearDivg_analysis(df, sens, ratio)
HBearDivg_analysis(df, sens, ratio)

df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
# (daily) Trendlinien
df = calc_TL_Up_Support(df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
df = calc_TL_Up_Resistance(df, min_gap=20, adjacent_candles=10, exclude_end_points=7)
df = calc_TL_Down_Resistance(df, min_gap=20, adjacent_candles=10, exclude_end_points=7)

df = calculate_golden_ratios(df)
```

### Output
- Pro Slice (Index i) wird ein Parquet `output_<LastDate>.parquet` erzeugt (Daily: vollständiger Tail; Intraday: oft `tail(400)`).
- Zusätzlich entsteht ein Sidecar `output_<LastDate>.json` mit `slice_index`, `last_date`, Pfad zum Parquet (wird nach sauberem Stopp/Ende aufgeräumt).

### Beispiel‑Outputspalten (Auszug)
Hinweis: Konkrete Spaltennamen können je nach Implementierungsdetails variieren. Die folgende Liste zeigt typische Felder, die von den Modulen erzeugt oder erweitert werden.

- Basis/Pre‑Processing (gemeinsam)
  - `date`, `open`, `high`, `low`, `close`, `volume`
  - `rsi`, `ema_fast`, `ema_slow`, `macd`, `macd_signal`, `macd_hist`
  - `candle_type`, `body_size`, `upper_shadow`, `lower_shadow`
  - `is_local_max`, `is_local_min`, `pivot_id`

- Divergenzen (Close/High vs. Indikator)
  - `bull_divg_close`, `bear_divg_close` (bool/int‑Flags)
  - `bull_divg_high`, `bear_divg_high`
  - `bull_divg_x2_close` (verstärkte/mehrfache Divergenz)
  - `divg_strength`, `divg_confirmed_at` (sofern implementiert)

- Support/Resistance
  - `sr_zone_id`, `sr_level_price`, `sr_strength`
  - Optional aggregierte Felder: `sr_levels_window`, `sr_touch_count`

- Trendlinien (nur Daily‑Modul mit TL)
  - `tl_up_support_id`, `tl_up_support_slope`, `tl_up_support_intercept`
  - `tl_up_res_id`, `tl_up_res_slope`, `tl_up_res_intercept`
  - `tl_down_res_id`, `tl_down_res_slope`, `tl_down_res_intercept`
  - Referenzen: `tl_x1_date`, `tl_x2_date`, `tl_last_touch`

- Golden Ratio / Fibonacci
  - `fib_0_236`, `fib_0_382`, `fib_0_5`, `fib_0_618`, `fib_0_786`
  - `golden_range_high`, `golden_range_low`

- Meta (nicht im DataFrame, aber im Sidecar/Dateinamen)
  - Sidecar JSON: `slice_index`, `last_date`, `parquet`
  - Dateiname enthält `last_date` der Slice‑Endezeile

## Troubleshooting
- Kein GUI‑Dialog sichtbar: Sicherstellen, dass ein Desktop‑Benutzersitzung aktiv ist. Fallbacks (IFileDialog/PowerShell) sind integriert; Gruppenrichtlinien können diese ggf. blockieren.
- `pyarrow`/`scikit-learn` Fehler: Pakete in das aktive venv installieren.
- Fortsetzen klappt nicht: Prüfen, ob `.state_<CSV>.json` vorhanden ist. Ohne State werden Sidecars ausgewertet; nach sauberem Stopp/Ende werden Sidecars gelöscht, da der State Vorrang hat.

---
Wenn gewünscht, kann dieses README um Screenshots, Beispiel‑Kommandos oder eine kurze Architektur‑Skizze ergänzt werden.
