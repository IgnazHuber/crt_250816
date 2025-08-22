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
        """Extrahiert Divergenzen aus DataFrame mit Validierungsdetails"""
        classic = []
        hidden = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if row.get('CBullD_gen', 0) == 1:
                # Validierungsdetails für Classic Divergence sammeln
                validation_data = self._get_validation_details(df, i, 'classic')
                
                classic.append({
                    'index': i,
                    'date': str(row['date']),
                    'low': float(row['low']),
                    'high': float(row['high']),
                    'close': float(row['close']),
                    'rsi': float(row.get('RSI', 0)),
                    'macd': float(row.get('macd_histogram', 0)),
                    'validation': validation_data
                })
            
            if row.get('CBullD_neg_MACD', 0) == 1:
                # Validierungsdetails für Hidden Divergence sammeln
                validation_data = self._get_validation_details(df, i, 'hidden')
                
                hidden.append({
                    'index': i,
                    'date': str(row['date']),
                    'low': float(row['low']),
                    'high': float(row['high']),
                    'close': float(row['close']),
                    'rsi': float(row.get('RSI', 0)),
                    'macd': float(row.get('macd_histogram', 0)),
                    'validation': validation_data
                })
        
        return {
            'classic': classic,
            'hidden': hidden,
            'total': len(classic) + len(hidden)
        }
    
    def _get_validation_details(self, df, current_index, div_type):
        """Sammelt detaillierte Validierungsdaten für eine Divergenz"""
        try:
            # Kontext-Fenster um die Divergenz
            window_size = 10
            start_idx = max(0, current_index - window_size)
            end_idx = min(len(df), current_index + window_size + 1)
            
            context_data = df.iloc[start_idx:end_idx].copy()
            
            # Relevante Spalten für Validierung
            validation_columns = ['date', 'low', 'high', 'close', 'RSI', 'macd_histogram', 
                                'local_min_price', 'local_min_RSI', 'local_min_MACD']
            
            validation_details = {
                'signal_type': div_type,
                'signal_index': current_index,
                'window_start': start_idx,
                'window_end': end_idx - 1,
                'context_data': [],
                'local_minima': [],
                'divergence_explanation': '',
                'strength_score': 0
            }
            
            # Kontext-Daten sammeln
            for idx, (_, row) in enumerate(context_data.iterrows()):
                data_point = {
                    'relative_index': start_idx + idx,
                    'is_signal': (start_idx + idx) == current_index,
                    'date': str(row['date']),
                    'low': float(row.get('low', 0)),
                    'high': float(row.get('high', 0)),
                    'close': float(row.get('close', 0)),
                    'rsi': float(row.get('RSI', 0)),
                    'macd': float(row.get('macd_histogram', 0)),
                    'is_price_min': bool(row.get('local_min_price', 0)),
                    'is_rsi_min': bool(row.get('local_min_RSI', 0)),
                    'is_macd_min': bool(row.get('local_min_MACD', 0))
                }
                validation_details['context_data'].append(data_point)
                
                # Lokale Minima sammeln
                if data_point['is_price_min'] or data_point['is_rsi_min'] or data_point['is_macd_min']:
                    validation_details['local_minima'].append(data_point)
            
            # Divergenz-Erklärung generieren
            signal_row = df.iloc[current_index]
            if div_type == 'classic':
                validation_details['divergence_explanation'] = self._explain_classic_divergence(validation_details, signal_row)
            else:
                validation_details['divergence_explanation'] = self._explain_hidden_divergence(validation_details, signal_row)
            
            # Stärke-Score berechnen (0-100)
            validation_details['strength_score'] = self._calculate_divergence_strength(validation_details)
            
            return validation_details
            
        except Exception as e:
            logger.warning(f"Validierungsdetails konnten nicht erstellt werden: {e}")
            return {
                'signal_type': div_type,
                'error': str(e),
                'strength_score': 0
            }
    
    def _explain_classic_divergence(self, validation_data, signal_row):
        """Erklärt eine Classic Bullish Divergence (Preis vs. RSI)"""
        try:
            # Finde die letzten zwei Preis-Minima und RSI-Minima
            price_minima = [p for p in validation_data['local_minima'] if p['is_price_min']]
            rsi_minima = [p for p in validation_data['local_minima'] if p['is_rsi_min']]
            
            if len(price_minima) >= 2 and len(rsi_minima) >= 2:
                # Letztes und vorletztes Minimum
                last_price_min = price_minima[-1]
                prev_price_min = price_minima[-2]
                last_rsi_min = rsi_minima[-1]
                prev_rsi_min = rsi_minima[-2]
                
                price_direction = "tieferes Tief" if last_price_min['low'] < prev_price_min['low'] else "höheres Tief"
                rsi_direction = "höheres Tief" if last_rsi_min['rsi'] > prev_rsi_min['rsi'] else "tieferes Tief"
                
                explanation = f"""CLASSIC BULLISH DIVERGENCE (Preis vs. RSI):

📉 PREIS-BEWEGUNG: {price_direction}
  • Vorheriges Tief: {prev_price_min['low']:.4f} am {prev_price_min['date'][:10]}
  • Aktuelles Tief: {last_price_min['low']:.4f} am {last_price_min['date'][:10]}
  • Preis-Differenz: {last_price_min['low'] - prev_price_min['low']:.4f}

📊 RSI-BEWEGUNG: {rsi_direction}  
  • Vorheriges RSI-Tief: {prev_rsi_min['rsi']:.2f}
  • Aktuelles RSI-Tief: {last_rsi_min['rsi']:.2f}
  • RSI-Differenz: {last_rsi_min['rsi'] - prev_rsi_min['rsi']:.2f}

✅ DIVERGENZ: Preis macht {price_direction}, RSI macht {rsi_direction}
🎯 BULLISH SIGNAL: RSI zeigt nachlassenden Verkaufsdruck"""
                
                return explanation.strip()
            
            return "Classic Divergence (Preis vs. RSI) erkannt, aber unvollständige Daten"
            
        except Exception as e:
            return f"Fehler bei Preis-vs-RSI Erklärung: {e}"
    
    def _explain_hidden_divergence(self, validation_data, signal_row):
        """Erklärt eine Hidden Bullish Divergence (Preis vs. MACD)"""
        try:
            # Analysiere MACD-Minima und Preis-Bewegung
            macd_minima = [p for p in validation_data['local_minima'] if p['is_macd_min']]
            price_minima = [p for p in validation_data['local_minima'] if p['is_price_min']]
            
            if len(macd_minima) >= 2:
                last_macd_min = macd_minima[-1]
                prev_macd_min = macd_minima[-2]
                
                macd_direction = "höheres Tief" if last_macd_min['macd'] > prev_macd_min['macd'] else "tieferes Tief"
                
                explanation = f"""HIDDEN BULLISH DIVERGENCE (Preis vs. MACD):

📈 MACD HISTOGRAM: {macd_direction}
  • Vorheriges MACD-Tief: {prev_macd_min['macd']:.4f}
  • Aktuelles MACD-Tief: {last_macd_min['macd']:.4f}
  • MACD-Differenz: {last_macd_min['macd'] - prev_macd_min['macd']:.4f}

🔋 MOMENTUM-ANALYSE: MACD zeigt Stärke-Zunahme
🎯 HIDDEN BULLISH: Unterliegende Stärke trotz Preiskonsolidierung"""
                
                return explanation.strip()
            
            return "Hidden Divergence (Preis vs. MACD) erkannt, aber unvollständige MACD-Daten"
            
        except Exception as e:
            return f"Fehler bei Preis-vs-MACD Erklärung: {e}"
    
    def _calculate_divergence_strength(self, validation_data):
        """Berechnet einen Stärke-Score für die Divergenz (0-100)"""
        try:
            score = 50  # Basis-Score
            
            # Anzahl der lokalen Minima (mehr = stärker)
            minima_count = len(validation_data.get('local_minima', []))
            score += min(minima_count * 5, 20)
            
            # Zeitliche Nähe der Minima (näher = stärker)
            if minima_count >= 2:
                last_minima = validation_data['local_minima'][-2:]
                if len(last_minima) == 2:
                    time_diff = abs(last_minima[1]['relative_index'] - last_minima[0]['relative_index'])
                    if time_diff <= 5:
                        score += 15
                    elif time_diff <= 10:
                        score += 10
                    else:
                        score += 5
            
            # Kontext-Daten-Qualität
            context_quality = len(validation_data.get('context_data', []))
            if context_quality >= 15:
                score += 15
            elif context_quality >= 10:
                score += 10
            
            return min(score, 100)
            
        except Exception as e:
            return 50  # Fallback-Score
    
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