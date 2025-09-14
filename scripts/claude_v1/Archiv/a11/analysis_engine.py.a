"""
Analysis Engine für Bullish Divergence Analyzer - KORRIGIERT
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
        """🔧 ALTE Methode - lädt aus Datei"""
        try:
            df = self.load_data(filepath)
            if df is None:
                return None
            return self.analyze_dataframe(df, variants, date_range)
        except Exception as e:
            logger.error(f"Analyse-Fehler (File): {e}")
            return None
    
    def analyze_dataframe(self, df, variants, date_range=None):
        """🔧 NEUE Methode - arbeitet direkt mit DataFrame"""
        try:
            if not self.modules_loaded:
                return None
            
            if df is None or df.empty:
                logger.error("Leerer DataFrame")
                return None
            
            logger.info(f"🔍 Analysiere DataFrame: {len(df)} Zeilen")
            
            # Optional: Datum-Filter
            if date_range:
                df = self._filter_date_range(df, date_range)
            
            # Technische Indikatoren berechnen
            logger.info("Berechne Indikatoren...")
            df = self.modules_loaded['Initialize_RSI_EMA_MACD'](df)
            
            if df is None:
                logger.error("Indikatoren-Berechnung fehlgeschlagen")
                return None
            
            # Lokale Extrema
            logger.info("Finde lokale Extrema...")
            try:
                self.modules_loaded['Local_Max_Min'](df)
            except Exception as e:
                logger.warning(f"Lokale Extrema Warnung: {e}")
                # Weiter versuchen auch wenn Local_Max_Min Probleme hat
            
            # Ergebnisse sammeln
            results = {}
            performance = {}
            
            # Für jede Variante analysieren
            for variant in variants:
                logger.info(f"Analysiere: {variant['name']}")
                
                df_var = df.copy()
                
                # Divergenz-Analyse
                try:
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
                        
                except Exception as e:
                    logger.error(f"Variante {variant['name']} fehlgeschlagen: {e}")
                    results[variant['id']] = {'classic': [], 'hidden': [], 'total': 0}
            
            # Chart-Daten vorbereiten
            chart_data = self._prepare_chart_data(df)
            
            return {
                'chartData': chart_data,
                'results': results,
                'performance': performance
            }
            
        except Exception as e:
            logger.error(f"Analyse-Fehler (DataFrame): {e}")
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
        """🔧 ERWEITERTE Divergenz-Extraktion mit besserer Validierung"""
        classic = []
        hidden = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if row.get('CBullD_gen', 0) == 1:
                # Classic Divergence mit verbesserter Validierung
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
                # Hidden Divergence mit verbesserter Validierung
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
        
        logger.info(f"🎯 Divergenzen gefunden: {len(classic)} Classic, {len(hidden)} Hidden")
        
        return {
            'classic': classic,
            'hidden': hidden,
            'total': len(classic) + len(hidden)
        }
    
    def _get_validation_details(self, df, current_index, div_type):
        """🔧 VERBESSERTE Validierungsdetails mit robusterer Datensammlung"""
        try:
            # Erweiteter Kontext-Fenster
            window_size = 15  # Größeres Fenster für bessere Analyse
            start_idx = max(0, current_index - window_size)
            end_idx = min(len(df), current_index + window_size + 1)
            
            context_data = df.iloc[start_idx:end_idx].copy()
            
            validation_details = {
                'signal_type': div_type,
                'signal_index': current_index,
                'window_start': start_idx,
                'window_end': end_idx - 1,
                'context_data': [],
                'local_minima': [],
                'divergence_explanation': '',
                'strength_score': 0,
                'data_quality': 'good'  # ✅ Datenqualität-Flag
            }
            
            # 🔍 VERBESSERTE Kontext-Daten-Sammlung
            for idx, (_, row) in enumerate(context_data.iterrows()):
                try:
                    data_point = {
                        'relative_index': start_idx + idx,
                        'is_signal': (start_idx + idx) == current_index,
                        'date': str(row.get('date', 'N/A')),
                        'low': float(row.get('low', 0)),
                        'high': float(row.get('high', 0)),
                        'close': float(row.get('close', 0)),
                        'rsi': float(row.get('RSI', 0)),
                        'macd': float(row.get('macd_histogram', 0)),
                        'is_price_min': bool(row.get('local_min_price', 0)) if 'local_min_price' in row else False,
                        'is_rsi_min': bool(row.get('local_min_RSI', 0)) if 'local_min_RSI' in row else False,
                        'is_macd_min': bool(row.get('local_min_MACD', 0)) if 'local_min_MACD' in row else False,
                        'has_data': not (np.isnan(row.get('RSI', np.nan)) or np.isnan(row.get('macd_histogram', np.nan)))
                    }
                    validation_details['context_data'].append(data_point)
                    
                    # Lokale Minima sammeln
                    if data_point['is_price_min'] or data_point['is_rsi_min'] or data_point['is_macd_min']:
                        validation_details['local_minima'].append(data_point)
                        
                except Exception as dp_error:
                    logger.warning(f"Datenpunkt-Fehler bei Index {start_idx + idx}: {dp_error}")
                    validation_details['data_quality'] = 'partial'
            
            # 🔍 DATENQUALITÄT prüfen
            valid_data_points = [dp for dp in validation_details['context_data'] if dp['has_data']]
            data_completeness = len(valid_data_points) / len(validation_details['context_data']) if validation_details['context_data'] else 0
            
            if data_completeness < 0.7:
                validation_details['data_quality'] = 'poor'
                validation_details['divergence_explanation'] = f"⚠️ UNVOLLSTÄNDIGE DATEN: Nur {data_completeness:.1%} der Indikatoren verfügbar."
            elif len(validation_details['local_minima']) < 2:
                validation_details['data_quality'] = 'insufficient_minima'
                validation_details['divergence_explanation'] = f"⚠️ UNVOLLSTÄNDIGE ANALYSE: Nur {len(validation_details['local_minima'])} lokale Minima gefunden (benötigt: ≥2)."
            else:
                # Vollständige Analyse möglich
                signal_row = df.iloc[current_index]
                if div_type == 'classic':
                    validation_details['divergence_explanation'] = self._explain_classic_divergence(validation_details, signal_row)
                else:
                    validation_details['divergence_explanation'] = self._explain_hidden_divergence(validation_details, signal_row)
            
            # Stärke-Score berechnen
            validation_details['strength_score'] = self._calculate_divergence_strength(validation_details)
            
            return validation_details
            
        except Exception as e:
            logger.warning(f"Validierungsdetails-Fehler: {e}")
            return {
                'signal_type': div_type,
                'error': str(e),
                'strength_score': 0,
                'data_quality': 'error',
                'divergence_explanation': f"❌ FEHLER bei Validierung: {str(e)}"
            }
    
    def _explain_classic_divergence(self, validation_data, signal_row):
        """🔧 ERWEITERTE Erklärung für Classic Bullish Divergence"""
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
                
                # Berechne Divergenz-Stärke
                price_change = ((last_price_min['low'] - prev_price_min['low']) / prev_price_min['low']) * 100
                rsi_change = last_rsi_min['rsi'] - prev_rsi_min['rsi']
                
                explanation = f"""🔍 CLASSIC BULLISH DIVERGENCE (Preis vs. RSI):

📊 KURZE ZUSAMMENFASSUNG:
  • {price_direction} vs. {rsi_direction} im RSI
  • Preis-Änderung: {price_change:+.2f}%
  • RSI-Änderung: {rsi_change:+.2f} Punkte

📈 DETAILLIERTE ANALYSE:

💰 PREIS-BEWEGUNG: {price_direction}
  • Vorheriges Tief: {prev_price_min['low']:.4f} am {prev_price_min['date'][:10]}
  • Aktuelles Tief: {last_price_min['low']:.4f} am {last_price_min['date'][:10]}
  • Preis-Differenz: {last_price_min['low'] - prev_price_min['low']:.4f} ({price_change:+.2f}%)

📊 RSI-BEWEGUNG: {rsi_direction}  
  • Vorheriges RSI-Tief: {prev_rsi_min['rsi']:.2f}
  • Aktuelles RSI-Tief: {last_rsi_min['rsi']:.2f}
  • RSI-Differenz: {rsi_change:+.2f} Punkte

✅ DIVERGENZ-LOGIK:
  • Preis macht {price_direction} → {'Schwäche' if price_change < 0 else 'Stärke'}
  • RSI macht {rsi_direction} → {'Momentum-Stärke' if rsi_change > 0 else 'weitere Schwäche'}
  • Interpretation: {'BULLISH - RSI zeigt weniger Verkaufsdruck!' if rsi_change > 0 else 'Schwaches Signal - RSI bestätigt Preis-Schwäche'}

🎯 TRADING-SIGNIFIKANZ:
  • Zeitabstand: {abs(last_price_min['relative_index'] - prev_price_min['relative_index'])} Kerzen
  • Signal-Stärke: {'Stark' if abs(rsi_change) > 5 else 'Schwach' if abs(rsi_change) < 2 else 'Mittel'}"""
                
                return explanation.strip()
            
            elif len(price_minima) < 2:
                return f"⚠️ Classic Divergence erkannt, aber nur {len(price_minima)} Preis-Minimum(a) gefunden (benötigt: ≥2)"
            else:
                return f"⚠️ Classic Divergence erkannt, aber nur {len(rsi_minima)} RSI-Minimum(a) gefunden (benötigt: ≥2)"
            
        except Exception as e:
            return f"❌ Fehler bei Classic Divergence Erklärung: {e}"
    
    def _explain_hidden_divergence(self, validation_data, signal_row):
        """🔧 ERWEITERTE Erklärung für Hidden Bullish Divergence"""
        try:
            # Analysiere MACD-Minima und Preis-Bewegung
            macd_minima = [p for p in validation_data['local_minima'] if p['is_macd_min']]
            price_minima = [p for p in validation_data['local_minima'] if p['is_price_min']]
            
            if len(macd_minima) >= 2:
                last_macd_min = macd_minima[-1]
                prev_macd_min = macd_minima[-2]
                
                macd_direction = "höheres Tief" if last_macd_min['macd'] > prev_macd_min['macd'] else "tieferes Tief"
                macd_change = last_macd_min['macd'] - prev_macd_min['macd']
                macd_change_percent = (macd_change / abs(prev_macd_min['macd'])) * 100 if prev_macd_min['macd'] != 0 else 0
                
                explanation = f"""🔍 HIDDEN BULLISH DIVERGENCE (Preis vs. MACD):

📊 KURZE ZUSAMMENFASSUNG:
  • MACD macht {macd_direction}
  • MACD-Änderung: {macd_change:+.4f} ({macd_change_percent:+.1f}%)
  • Hidden Divergence = Momentum-Stärkung

📈 DETAILLIERTE ANALYSE:

⚡ MACD HISTOGRAM: {macd_direction}
  • Vorheriges MACD-Tief: {prev_macd_min['macd']:.4f}
  • Aktuelles MACD-Tief: {last_macd_min['macd']:.4f}
  • MACD-Differenz: {macd_change:+.4f} ({macd_change_percent:+.1f}%)

💡 MOMENTUM-ANALYSE:
  • MACD zeigt {'Stärke-Zunahme' if macd_change > 0 else 'weitere Schwäche'}
  • Histogram-Verbesserung: {'Ja' if macd_change > 0 else 'Nein'}
  • Beide Werte negativ: {'Ja' if last_macd_min['macd'] < 0 and prev_macd_min['macd'] < 0 else 'Nein'}

🎯 HIDDEN DIVERGENCE BEDEUTUNG:
  • Definition: Unterliegende Momentum-Stärke trotz Preis-Konsolidierung
  • Signal: {'BULLISH - MACD zeigt versteckte Stärke!' if macd_change > 0 else 'SCHWACH - MACD bestätigt Schwäche'}
  • Zeitabstand: {abs(last_macd_min['relative_index'] - prev_macd_min['relative_index'])} Kerzen

🔋 MOMENTUM-STÄRKE:
  • Verbesserung: {macd_change_percent:+.1f}%
  • Bewertung: {'Stark' if abs(macd_change_percent) > 10 else 'Schwach' if abs(macd_change_percent) < 5 else 'Mittel'}"""
                
                return explanation.strip()
            
            return f"⚠️ Hidden Divergence erkannt, aber nur {len(macd_minima)} MACD-Minimum(a) gefunden (benötigt: ≥2)"
            
        except Exception as e:
            return f"❌ Fehler bei Hidden Divergence Erklärung: {e}"
    
    def _calculate_divergence_strength(self, validation_data):
        """🔧 ERWEITERTE Stärke-Berechnung mit Datenqualität"""
        try:
            score = 30  # Niedrigerer Basis-Score
            
            # Datenqualität-Bonus
            if validation_data.get('data_quality') == 'good':
                score += 20
            elif validation_data.get('data_quality') == 'partial':
                score += 10
            else:
                score -= 10  # Malus für schlechte Datenqualität
            
            # Anzahl der lokalen Minima (mehr = stärker)
            minima_count = len(validation_data.get('local_minima', []))
            score += min(minima_count * 8, 25)
            
            # Zeitliche Nähe der Minima (näher = stärker)
            if minima_count >= 2:
                last_minima = validation_data['local_minima'][-2:]
                if len(last_minima) == 2:
                    time_diff = abs(last_minima[1]['relative_index'] - last_minima[0]['relative_index'])
                    if time_diff <= 5:
                        score += 20
                    elif time_diff <= 10:
                        score += 15
                    elif time_diff <= 20:
                        score += 10
                    else:
                        score += 5
            
            # Kontext-Daten-Qualität
            context_quality = len(validation_data.get('context_data', []))
            if context_quality >= 20:
                score += 15
            elif context_quality >= 15:
                score += 10
            elif context_quality >= 10:
                score += 5
            
            return min(max(score, 0), 100)  # Zwischen 0 und 100
            
        except Exception as e:
            return 25  # Fallback-Score
    
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
                'hidden_signals': len(divergences['hidden']),
                'avg_strength_classic': 0,
                'avg_strength_hidden': 0
            }
            
            # Durchschnittliche Stärke berechnen
            if divergences['classic']:
                classic_strengths = [d.get('validation', {}).get('strength_score', 0) for d in divergences['classic']]
                metrics['avg_strength_classic'] = sum(classic_strengths) / len(classic_strengths)
            
            if divergences['hidden']:
                hidden_strengths = [d.get('validation', {}).get('strength_score', 0) for d in divergences['hidden']]
                metrics['avg_strength_hidden'] = sum(hidden_strengths) / len(hidden_strengths)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Performance-Berechnung fehlgeschlagen: {e}")
            return {'total_signals': 0, 'classic_signals': 0, 'hidden_signals': 0}
