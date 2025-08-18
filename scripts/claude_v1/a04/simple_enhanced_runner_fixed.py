"""
Vereinfachte Enhanced Runner - Funktioniert mit bestehenden Modulen
Erweitert das bestehende System um Hidden und Bearish Divergenzen
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# Basis-Module (die funktionieren)
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    from interactive_simple import create_plotly_chart, main as show_interactive_chart
    print("✅ Alle Basis-Module erfolgreich importiert")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_all_datetime_columns_for_excel(df):
    """
    Aggressiver Fix für alle datetime-Spalten um Excel-Kompatibilität zu gewährleisten
    """
    df_fixed = df.copy()
    
    # Methode 1: Gehe durch ALLE Spalten und repariere sie
    for col in df_fixed.columns:
        try:
            # Prüfe ob es sich um eine datetime-Spalte handelt
            if (str(df_fixed[col].dtype).startswith('datetime64') or 
                'date' in col.lower() or 
                'time' in col.lower()):
                
                # Konvertiere zu timezone-naive datetime
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce').dt.tz_localize(None)
                logger.debug(f"✅ Fixed column: {col}")
                
        except Exception as e:
            logger.debug(f"⚠️ Could not fix column {col}: {e}")
            # Falls immer noch problematisch, konvertiere zu String
            try:
                if 'date' in col.lower():
                    df_fixed[col] = df_fixed[col].astype(str)
            except:
                pass
    
    # Methode 2: Spezifische bekannte Problemspalten
    problem_columns = [
        'date', 'Date', 'DATE',
        'CBullD_Lower_Low_date_gen', 'CBullD_Higher_Low_date_gen',
        'CBullD_Lower_Low_date_neg_MACD', 'CBullD_Higher_Low_date_neg_MACD',
        'HBullD_Lower_Low_date', 'HBullD_Higher_Low_date',
        'BearD_Higher_High_date', 'BearD_Lower_High_date'
    ]
    
    for col in problem_columns:
        if col in df_fixed.columns:
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce').dt.tz_localize(None)
            except:
                # Als letzter Ausweg: als String speichern
                df_fixed[col] = df_fixed[col].astype(str)
    
    # Methode 3: Alle object-Spalten prüfen
    for col in df_fixed.select_dtypes(include=['object']).columns:
        if len(df_fixed) > 0:
            try:
                sample = df_fixed[col].dropna().iloc[0] if not df_fixed[col].dropna().empty else None
                if sample and hasattr(sample, 'tzinfo'):
                    df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce').dt.tz_localize(None)
            except:
                pass
    
    return df_fixed

def add_hidden_bullish_divergences(df: pd.DataFrame, window: int = 5, 
                                  candle_tol: float = 0.1, macd_tol: float = 3.25) -> pd.DataFrame:
    """
    Fügt Hidden Bullish Divergenzen hinzu
    Hidden: Höhere Lows im Preis, niedrigere Lows in Indikatoren
    """
    logger.info("🔍 Analysiere Hidden Bullish Divergenzen...")
    
    n = len(df)
    dates = pd.to_datetime(df['date']).to_numpy()
    
    # Convert columns if needed
    for col in ['LM_Low_window_2_CS', 'LM_Low_window_1_CS', 'RSI', 'macd_histogram']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Initialize arrays
    HBullD = np.zeros(n)
    HBullD_Lower_Low = np.zeros(n)
    HBullD_Higher_Low = np.zeros(n)
    HBullD_Lower_Low_RSI = np.zeros(n)
    HBullD_Higher_Low_RSI = np.zeros(n)
    HBullD_Lower_Low_MACD = np.zeros(n)
    HBullD_Higher_Low_MACD = np.zeros(n)
    HBullD_Lower_Low_date = np.zeros(n, dtype=object)
    HBullD_Higher_Low_date = np.zeros(n, dtype=object)
    HBullD_Date_Gap = np.zeros(n)
    
    # Extract data
    LM_Low_window_2_CS = df['LM_Low_window_2_CS'].to_numpy()
    LM_Low_window_1_CS = df['LM_Low_window_1_CS'].to_numpy()
    RSI = df['RSI'].to_numpy()
    macd_histogram = df['macd_histogram'].to_numpy()
    
    # Find Hidden Bullish Divergences
    valid_indices = np.where(LM_Low_window_2_CS > 0)[0]
    valid_indices = valid_indices[(valid_indices >= window) & (valid_indices < n - 1)]
    
    hidden_count = 0
    
    for i in valid_indices:
        # Find last non-zero LM_Low_window_1_CS before index i
        prev_nonzero = np.where(LM_Low_window_1_CS[:i] != 0)[0]
        if len(prev_nonzero) == 0:
            continue
        
        j = prev_nonzero[-1]
        
        # Hidden Bullish conditions
        current_low = LM_Low_window_2_CS[i]
        previous_low = LM_Low_window_1_CS[j]
        
        current_rsi = RSI[i]
        previous_rsi = RSI[j]
        
        current_macd = macd_histogram[i]
        previous_macd = macd_histogram[j]
        
        # Hidden Bullish: Price makes higher lows, but indicators make lower lows
        price_condition = current_low > previous_low  # Higher lows in price
        rsi_condition = current_rsi < previous_rsi    # Lower lows in RSI
        macd_condition = current_macd < previous_macd # Lower lows in MACD
        
        # Tolerance checks
        price_diff_percent = abs(100 * (current_low - previous_low) / previous_low)
        rsi_diff = abs(current_rsi - previous_rsi)
        
        if (price_condition and rsi_condition and macd_condition and
            price_diff_percent > candle_tol and rsi_diff > 2):
            
            HBullD[i] = 1
            HBullD_Lower_Low[i] = previous_low
            HBullD_Higher_Low[i] = current_low
            HBullD_Lower_Low_RSI[i] = previous_rsi
            HBullD_Higher_Low_RSI[i] = current_rsi
            HBullD_Lower_Low_MACD[i] = previous_macd
            HBullD_Higher_Low_MACD[i] = current_macd
            HBullD_Lower_Low_date[i] = dates[j]
            HBullD_Higher_Low_date[i] = dates[i]
            HBullD_Date_Gap[i] = i - j
            hidden_count += 1
    
    # Add to dataframe
    df['HBullD'] = HBullD
    df['HBullD_Lower_Low'] = HBullD_Lower_Low
    df['HBullD_Higher_Low'] = HBullD_Higher_Low
    df['HBullD_Lower_Low_RSI'] = HBullD_Lower_Low_RSI
    df['HBullD_Higher_Low_RSI'] = HBullD_Higher_Low_RSI
    df['HBullD_Lower_Low_MACD'] = HBullD_Lower_Low_MACD
    df['HBullD_Higher_Low_MACD'] = HBullD_Higher_Low_MACD
    df['HBullD_Lower_Low_date'] = HBullD_Lower_Low_date
    df['HBullD_Higher_Low_date'] = HBullD_Higher_Low_date
    df['HBullD_Date_Gap'] = HBullD_Date_Gap
    
    logger.info(f"✅ Hidden Bullish Divergenzen gefunden: {hidden_count}")
    return df

def add_bearish_divergences(df: pd.DataFrame, window: int = 5, 
                           candle_tol: float = 0.1, macd_tol: float = 3.25) -> pd.DataFrame:
    """
    Fügt Bearish Divergenzen hinzu
    Bearish: Höhere Highs im Preis, niedrigere Highs in Indikatoren
    """
    logger.info("🔍 Analysiere Bearish Divergenzen...")
    
    n = len(df)
    dates = pd.to_datetime(df['date']).to_numpy()
    
    # Convert columns if needed
    for col in ['LM_High_window_2_CS', 'LM_High_window_1_CS', 'RSI', 'macd_histogram']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Initialize arrays
    BearD = np.zeros(n)
    BearD_Higher_High = np.zeros(n)
    BearD_Lower_High = np.zeros(n)
    BearD_Higher_High_RSI = np.zeros(n)
    BearD_Lower_High_RSI = np.zeros(n)
    BearD_Higher_High_MACD = np.zeros(n)
    BearD_Lower_High_MACD = np.zeros(n)
    BearD_Higher_High_date = np.zeros(n, dtype=object)
    BearD_Lower_High_date = np.zeros(n, dtype=object)
    BearD_Date_Gap = np.zeros(n)
    
    # Extract data
    LM_High_window_2_CS = df['LM_High_window_2_CS'].to_numpy()
    LM_High_window_1_CS = df['LM_High_window_1_CS'].to_numpy()
    RSI = df['RSI'].to_numpy()
    macd_histogram = df['macd_histogram'].to_numpy()
    
    # Find Bearish Divergences
    valid_indices = np.where(LM_High_window_2_CS > 0)[0]
    valid_indices = valid_indices[(valid_indices >= window) & (valid_indices < n - 1)]
    
    bearish_count = 0
    
    for i in valid_indices:
        # Find last non-zero LM_High_window_1_CS before index i
        prev_nonzero = np.where(LM_High_window_1_CS[:i] != 0)[0]
        if len(prev_nonzero) == 0:
            continue
        
        j = prev_nonzero[-1]
        
        # Bearish conditions
        current_high = LM_High_window_2_CS[i]
        previous_high = LM_High_window_1_CS[j]
        
        current_rsi = RSI[i]
        previous_rsi = RSI[j]
        
        current_macd = macd_histogram[i]
        previous_macd = macd_histogram[j]
        
        # Bearish: Price makes higher highs, but indicators make lower highs
        price_condition = current_high > previous_high  # Higher highs in price
        rsi_condition = current_rsi < previous_rsi      # Lower highs in RSI
        macd_condition = current_macd < previous_macd   # Lower highs in MACD
        
        # Tolerance checks
        price_diff_percent = abs(100 * (current_high - previous_high) / previous_high)
        rsi_diff = abs(current_rsi - previous_rsi)
        
        # Additional condition for bearish (typically at high RSI levels)
        rsi_condition_extended = (
            rsi_condition or 
            (current_rsi > 60 and abs(rsi_diff) < 8)  # At high RSI levels
        )
        
        if (price_condition and rsi_condition_extended and macd_condition and
            price_diff_percent > candle_tol):
            
            BearD[i] = 1
            BearD_Higher_High[i] = current_high
            BearD_Lower_High[i] = previous_high
            BearD_Higher_High_RSI[i] = current_rsi
            BearD_Lower_High_RSI[i] = previous_rsi
            BearD_Higher_High_MACD[i] = current_macd
            BearD_Lower_High_MACD[i] = previous_macd
            BearD_Higher_High_date[i] = dates[i]
            BearD_Lower_High_date[i] = dates[j]
            BearD_Date_Gap[i] = i - j
            bearish_count += 1
    
    # Add to dataframe
    df['BearD'] = BearD
    df['BearD_Higher_High'] = BearD_Higher_High
    df['BearD_Lower_High'] = BearD_Lower_High
    df['BearD_Higher_High_RSI'] = BearD_Higher_High_RSI
    df['BearD_Lower_High_RSI'] = BearD_Lower_High_RSI
    df['BearD_Higher_High_MACD'] = BearD_Higher_High_MACD
    df['BearD_Lower_High_MACD'] = BearD_Lower_High_MACD
    df['BearD_Higher_High_date'] = BearD_Higher_High_date
    df['BearD_Lower_High_date'] = BearD_Lower_High_date
    df['BearD_Date_Gap'] = BearD_Date_Gap
    
    logger.info(f"✅ Bearish Divergenzen gefunden: {bearish_count}")
    return df

def select_data_file():
    """Dateiauswahl-Dialog"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Wähle Datenquelle für erweiterte Analyse",
        filetypes=[
            ("CSV Dateien", "*.csv"),
            ("Parquet Dateien", "*.parquet"),
            ("Alle", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def run_enhanced_analysis():
    """
    Führt erweiterte Divergenz-Analyse durch
    """
    try:
        logger.info("🚀 STARTE ERWEITERTE DIVERGENZ-ANALYSE")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Schritt 1: Datei auswählen
        file_path = select_data_file()
        if not file_path:
            logger.error("❌ Keine Datei ausgewählt")
            return False
        
        # Schritt 2: Daten laden
        logger.info(f"📊 Lade Daten: {Path(file_path).name}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Nicht unterstütztes Dateiformat")
        
        # Validierung
        required = ['date', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Fehlende Spalten: {missing}")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ {len(df)} Zeilen geladen")
        logger.info(f"📅 Zeitraum: {df['date'].min().date()} bis {df['date'].max().date()}")
        
        # Schritt 3: Basis-Indikatoren berechnen
        logger.info("🔧 Berechne Basis-Indikatoren...")
        Initialize_RSI_EMA_MACD(df)
        Local_Max_Min(df)
        logger.info("✅ Basis-Indikatoren berechnet")
        
        # Schritt 4: Standard-Divergenzen
        logger.info("📈 Standard-Divergenz-Analyse...")
        CBullDivg_analysis(df, 5, 0.1, 3.25)
        
        classic_count = (df['CBullD_gen'] == 1).sum() if 'CBullD_gen' in df.columns else 0
        neg_macd_count = (df['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in df.columns else 0
        logger.info(f"✅ Standard-Divergenzen: {classic_count} Classic, {neg_macd_count} Negative MACD")
        
        # Schritt 5: Erweiterte Divergenzen
        logger.info("🚀 Erweiterte Divergenz-Analyse...")
        add_hidden_bullish_divergences(df, 5, 0.1, 3.25)
        add_bearish_divergences(df, 5, 0.1, 3.25)
        
        # Schritt 6: Gesamtstatistik
        logger.info("\n📊 GESAMTERGEBNISSE:")
        logger.info("-" * 40)
        
        divergence_stats = {
            'Classic Bullish': (df['CBullD_gen'] == 1).sum() if 'CBullD_gen' in df.columns else 0,
            'Negative MACD': (df['CBullD_neg_MACD'] == 1).sum() if 'CBullD_neg_MACD' in df.columns else 0,
            'Hidden Bullish': (df['HBullD'] == 1).sum() if 'HBullD' in df.columns else 0,
            'Classic Bearish': (df['BearD'] == 1).sum() if 'BearD' in df.columns else 0
        }
        
        total_divergences = sum(divergence_stats.values())
        
        for div_type, count in divergence_stats.items():
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            logger.info(f"  {div_type:<15}: {count:>3} ({percentage:.1f}%)")
        
        logger.info(f"  {'TOTAL':<15}: {total_divergences:>3}")
        
        # Schritt 7: Interaktives Chart
        logger.info("\n📊 Erstelle interaktives Chart...")
        chart = create_plotly_chart(df)
        chart.update_layout(title="Erweiterte Divergenz-Analyse - Alle Typen")
        chart.show()
        
        # Schritt 8: Excel Export mit kugelsicherem Fix
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"enhanced_divergence_analysis_{timestamp}.xlsx"
        
        logger.info(f"📋 Exportiere Ergebnisse: {excel_filename}")
        
        try:
            # KUGELSICHERER Excel Export
            df_export = fix_all_datetime_columns_for_excel(df)
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # Komplette Daten
                df_export.to_excel(writer, sheet_name='Complete_Data', index=False)
                
                # Zusammenfassung
                summary_data = []
                summary_data.append(['DIVERGENZ-STATISTIKEN', ''])
                for div_type, count in divergence_stats.items():
                    summary_data.append([div_type, count])
                summary_data.append(['TOTAL', total_divergences])
                summary_data.append(['', ''])
                summary_data.append(['PARAMETER', ''])
                summary_data.append(['Window Size', 5])
                summary_data.append(['Candle Tolerance', 0.1])
                summary_data.append(['MACD Tolerance', 3.25])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Alle Divergenzen
                all_divergences = []
                for div_col in ['CBullD_gen', 'CBullD_neg_MACD', 'HBullD', 'BearD']:
                    if div_col in df.columns:
                        div_data = df[df[div_col] == 1].copy()
                        if not div_data.empty:
                            div_data['Divergence_Type'] = div_col
                            # Timezone auch hier entfernen!
                            if 'date' in div_data.columns:
                                div_data['date'] = pd.to_datetime(div_data['date']).dt.tz_localize(None)
                            all_divergences.append(div_data[['date', 'close', 'RSI', 'macd_histogram', 'Divergence_Type']])
                
                if all_divergences:
                    combined_div = pd.concat(all_divergences, ignore_index=True)
                    combined_div = combined_div.sort_values('date').reset_index(drop=True)
                    combined_div.to_excel(writer, sheet_name='All_Divergences', index=False)
            
            logger.info(f"✅ Excel erfolgreich exportiert: {excel_filename}")
            
        except Exception as e:
            logger.error(f"❌ Excel Export fehlgeschlagen: {e}")
            logger.info("📊 Aber Chart und Analyse waren erfolgreich!")
            # Excel-Fehler sollte nicht die ganze Analyse zum Scheitern bringen
        
        # Zusammenfassung
        elapsed_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info("🎉 ERWEITERTE ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
        logger.info(f"⏱️  Gesamtzeit: {elapsed_time:.1f} Sekunden")
        logger.info(f"📁 Datei verarbeitet: {Path(file_path).name}")
        logger.info(f"📊 Zeilen analysiert: {len(df)}")
        logger.info(f"🎯 Divergenzen gefunden: {total_divergences}")
        logger.info(f"📋 Excel-Export: {excel_filename}")
        logger.info("📊 Interaktives Chart im Browser geöffnet")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Hauptfunktion
    """
    print("🚀 Enhanced Divergence Analysis System")
    print("="*50)
    
    try:
        success = run_enhanced_analysis()
        
        if success:
            print("\n✅ Erweiterte Analyse erfolgreich abgeschlossen!")
            print("📊 Chart wurde im Browser geöffnet")
            print("📋 Excel-Datei wurde erstellt")
        else:
            print("\n❌ Analyse fehlgeschlagen - siehe Log für Details")
            
    except KeyboardInterrupt:
        print("\n⏹️ Analyse vom Benutzer abgebrochen")
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    main()