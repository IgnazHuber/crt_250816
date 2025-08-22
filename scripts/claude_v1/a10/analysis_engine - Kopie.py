"""
Analysis Engine für Bullish Divergence Analyzer
Kapselt die gesamte Analyse-Logik
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Hauptklasse für technische Analyse"""
    
    def __init__(self):
        self.modules_loaded = self._load_modules()
        
    def _load_modules(self):
        """Lädt die Python Analyse-Module"""
        modules = {}
        try:
            from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
            modules['Initialize_RSI_EMA_MACD'] = Initialize_RSI_EMA_MACD
            
            from Local_Maximas_Minimas import Local_Max_Min
            modules['Local_Max_Min'] = Local_Max_Min
            
            from CBullDivg_Analysis_vectorized import CBullDivg_analysis
            modules['CBullDivg_analysis'] = CBullDivg_analysis
            
            logger.info("✅ Alle Analyse-Module geladen")
            return modules
            
        except ImportError as e:
            logger.error(f"❌ Module fehlen: {e}")
            return None
    
    def check_modules(self):
        """Prüft verfügbare Module"""
        if not self.modules_loaded:
            return {'status': 'error', 'modules': []}
        
        return {
            'status': 'ok',
            'modules': list(self.modules_loaded.keys())
        }
    
    def load_data(self, filepath):
        """Lädt und validiert Daten"""
        try:
            path = Path(filepath)
            
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
            elif path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path)
            else:
                logger.error(f"Unbekanntes Format: {path.suffix}")
                return None
            
            # Validierung
            required = ['date', 'open', 'high', 'low', 'close']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                logger.error(f"Fehlende Spalten: {missing}")
                return None
            
            logger.info(f"✅ Daten geladen: {len(df)} Zeilen")
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Laden: {e}")
            return None
    
    def analyze(self, filepath, variants, date_range=None):
        """Führt komplette Analyse durch"""
        try:
            if not self.modules_loaded:
                return None
            
            # Daten laden
            df = self.load_data(filepath)
            if df is None:
                return None
            
            # Optional: Datum-Filter
            if date_range:
                df = self._filter_date_range(df, date_range)
            
            # Technische Indikatoren berechnen
            logger.info("Berechne Indikatoren...")
            df = self.modules_loaded['Initialize_RSI_EMA_MACD'](df)
            
            # Lokale Extrema
            logger.info("Finde lokale Extrema...")
            self.modules_loaded['Local_Max_Min'](df)
            
            # Ergebnisse sammeln
            results = {}
            performance = {}
            
            # Für jede Variante analysieren
            for variant in variants:
                logger.info(f"Analysiere: {variant['name']}")
                
                df_var = df.copy()
                
                # Divergenz-Analyse
                self.modules_loaded['CBullDivg_analysis'](
                    df_var,
                    variant['window'],
                    variant['candleTol'],
                    variant['macdTol']
                )
                
                # Ergebnisse extrahieren
                divergences = self._extract_divergences(df_var)
                results[variant['id']] = divergences
                
                # Performance berechnen (optional)
                if 'calculate_performance' in variant:
                    performance[variant['id']] = self._calculate_performance(
                        df_var, divergences
                    )
            
            # Chart-Daten vorbereiten
            chart_data = self._prepare_chart_data(df)
            
            return {
                'chartData': chart_data,
                'results': results,
                'performance': performance
            }
            
        except Exception as e:
            logger.error(f"Analyse-Fehler: {e}")
            return None
    
    def _filter_date_range(self, df, date_range):
        """Filtert DataFrame nach Datum"""
        try:
            df['date'] = pd.to_datetime(df['date'])
            
            if date_range.get('start'):
                start = pd.to_datetime(date_range['start'])
                df = df[df['date'] >= start]
            
            if date_range.get('end'):
                end = pd.to_datetime(date_range['end'])
                df = df[df['date'] <= end]
            
            return df
        except Exception as e:
            logger.warning(f"Datum-Filter fehlgeschlagen: {e}")
            return df
    
    def _extract_divergences(self, df):
        """Extrahiert Divergenzen aus DataFrame"""
        classic = []
        hidden = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if row.get('CBullD_gen', 0) == 1:
                classic.append({
                    'index': i,
                    'date': str(row['date']),
                    'low': float(row['low']),
                    'high': float(row['high']),
                    'close': float(row['close']),
                    'rsi': float(row.get('RSI', 0)),
                    'macd': float(row.get('macd_histogram', 0))
                })
            
            if row.get('CBullD_neg_MACD', 0) == 1:
                hidden.append({
                    'index': i,
                    'date': str(row['date']),
                    'low': float(row['low']),
                    'high': float(row['high']),
                    'close': float(row['close']),
                    'rsi': float(row.get('RSI', 0)),
                    'macd': float(row.get('macd_histogram', 0))
                })
        
        return {
            'classic': classic,
            'hidden': hidden,
            'total': len(classic) + len(hidden)
        }
    
    def _prepare_chart_data(self, df):
        """Bereitet Daten für Chart vor"""
        try:
            data = {
                'dates': df['date'].astype(str).tolist(),
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'close': df['close'].tolist(),
                'volume': df.get('volume', pd.Series(np.random.randint(
                    100000, 1000000, len(df)))).tolist()
            }
            
            # Technische Indikatoren
            if 'RSI' in df:
                data['rsi'] = df['RSI'].fillna(0).tolist()
            
            if 'macd_histogram' in df:
                data['macd_histogram'] = df['macd_histogram'].fillna(0).tolist()
            
            # EMAs
            for period in [20, 50, 100, 200]:
                col = f'EMA_{period}'
                if col in df:
                    data[f'ema{period}'] = df[col].fillna(0).tolist()
            
            return data
            
        except Exception as e:
            logger.error(f"Chart-Daten Fehler: {e}")
            return {}
    
    def _calculate_performance(self, df, divergences):
        """Berechnet Performance-Metriken für Divergenzen"""
        try:
            metrics = {
                'total_signals': divergences['total'],
                'classic_signals': len(divergences['classic']),
                'hidden_signals': len(divergences['hidden'])
            }
            
            # Weitere Metriken könnten hier berechnet werden
            # z.B. Erfolgsrate, durchschnittliche Bewegung nach Signal, etc.
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Performance-Berechnung fehlgeschlagen: {e}")
            return {}