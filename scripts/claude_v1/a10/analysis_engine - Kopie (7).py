import pandas as pd
import numpy as np
import logging
from datetime import datetime
import uuid
import os
import sys

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisEngine:
    def __init__(self):
        self.modules_loaded = {}
        self.session_data = {}  # Dictionary zur Speicherung der Session-Daten
        self._load_modules()

    def _load_modules(self):
        """Dynamisches Laden der Analyse-Module."""
        try:
            # Versuche die Module direkt zu importieren
            try:
                import Initialize_RSI_EMA_MACD
                self.modules_loaded['Initialize_RSI_EMA_MACD'] = Initialize_RSI_EMA_MACD
                logger.info("Modul Initialize_RSI_EMA_MACD erfolgreich geladen")
            except ImportError as e:
                logger.error(f"Fehler beim Import von Initialize_RSI_EMA_MACD: {e}")

            try:
                import Local_Maximas_Minimas
                self.modules_loaded['Local_Max_Min'] = Local_Maximas_Minimas
                logger.info("Modul Local_Maximas_Minimas erfolgreich geladen")
            except ImportError as e:
                logger.error(f"Fehler beim Import von Local_Maximas_Minimas: {e}")

            try:
                import CBullDivg_Analysis_vectorized
                self.modules_loaded['CBullDivg_analysis'] = CBullDivg_Analysis_vectorized
                logger.info("Modul CBullDivg_Analysis_vectorized erfolgreich geladen")
            except ImportError as e:
                logger.error(f"Fehler beim Import von CBullDivg_Analysis_vectorized: {e}")

            try:
                import DivergenceArrows
                self.modules_loaded['DivergenceArrows'] = DivergenceArrows
                logger.info("Modul DivergenceArrows erfolgreich geladen")
            except ImportError as e:
                logger.warning(f"DivergenceArrows nicht gefunden, verwende Fallback (keine Annotationen): {e}")

            logger.info(f"Module geladen: {list(self.modules_loaded.keys())}")
            
        except Exception as e:
            logger.error(f"Allgemeiner Fehler beim Laden der Module: {e}")

    def _call_module_function(self, module_name, function_name, *args, **kwargs):
        """Hilfsfunktion zum sicheren Aufrufen von Modul-Funktionen."""
        try:
            module = self.modules_loaded.get(module_name)
            if not module:
                logger.error(f"Modul {module_name} nicht geladen")
                return None
            
            # Versuche die Funktion auf verschiedene Arten aufzurufen
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                return func(*args, **kwargs)
            elif hasattr(module, 'run'):
                return module.run(*args, **kwargs)
            elif hasattr(module, 'execute'):
                return module.execute(*args, **kwargs)
            elif callable(module):
                return module(*args, **kwargs)
            else:
                logger.error(f"Keine aufrufbare Funktion in {module_name} gefunden")
                return None
        except Exception as e:
            logger.error(f"Fehler beim Aufrufen von {module_name}.{function_name}: {e}")
            return None

    def load_data(self, file_path):
        """Lädt die CSV- oder Parquet-Datei und speichert sie in der Session."""
        try:
            # Validierung des Dateipfads
            if not os.path.exists(file_path):
                logger.error(f"Datei {file_path} existiert nicht")
                return {'success': False, 'error': f"Datei {file_path} nicht gefunden"}

            # Dateityp prüfen
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"Ungültige Dateierweiterung: {file_extension}")
                return {'success': False, 'error': f"Ungültige Dateierweiterung: {file_extension}"}

            logger.info(f"Datei {file_path} geladen: {len(df)} Zeilen")

            # Prüfen, ob die erforderlichen Spalten vorhanden sind
            required_columns = ['date', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Fehlende Spalten in der Datei: {missing_columns}")
                return {'success': False, 'error': f"Fehlende Spalten: {missing_columns}"}

            # Datum in datetime umwandeln
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                logger.error(f"Fehler beim Konvertieren der 'date'-Spalte: {e}")
                return {'success': False, 'error': f"Ungültiges Datumsformat in der 'date'-Spalte"}

            # Prüfen auf NaN-Werte in kritischen Spalten
            nan_counts = df[required_columns].isna().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"NaN-Werte in der Datei: {nan_counts.to_dict()}")
                # Entferne NaN-Zeilen statt Fehler zu werfen
                df = df.dropna(subset=required_columns)
                logger.info(f"NaN-Zeilen entfernt. Verbleibende Zeilen: {len(df)}")

            # Session-ID generieren
            session_id = str(uuid.uuid4())
            self.session_data[session_id] = df
            logger.info(f"Session {session_id} erstellt mit {len(df)} Zeilen")

            return {
                'success': True,
                'session_id': session_id,
                'info': {'rows': len(df), 'columns': list(df.columns)}
            }
        except Exception as e:
            logger.error(f"Fehler beim Laden der Datei {file_path}: {e}")
            return {'success': False, 'error': str(e)}

    def _filter_date_range(self, df, date_range):
        """Filtert den DataFrame nach einem Datumsbereich."""
        try:
            start_date, end_date = date_range
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            filtered_df = df[mask].copy()
            logger.info(f"Datumsfilter angewendet: {len(filtered_df)} Zeilen nach Filterung")
            return filtered_df
        except Exception as e:
            logger.error(f"Fehler beim Filtern des Datumsbereichs: {e}")
            return df

    def _prepare_chart_data(self, df):
        """Bereitet Daten für Plotly-Charts vor."""
        try:
            chart_data = {
                'dates': df['date'].astype(str).tolist(),
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'close': df['close'].tolist()
            }
            
            # Optionale Indikatoren
            for col in ['ema20', 'ema50', 'ema100', 'ema200', 'rsi', 'macd_histogram', 'extrema_strength']:
                if col in df.columns:
                    chart_data[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).tolist()
            
            logger.info(f"Chart-Daten vorbereitet: {len(chart_data['dates'])} Zeilen")
            return chart_data
        except Exception as e:
            logger.error(f"Fehler beim Vorbereiten der Chart-Daten: {e}")
            return {}

    def _extract_divergences(self, df):
        """Extrahiert Divergenz-Daten aus dem DataFrame."""
        try:
            divergences = {'classic': [], 'hidden': []}
            for div_type in ['classic', 'hidden']:
                col = f"{div_type}_divergence"
                if col in df.columns:
                    mask = df[col] == True
                    for idx in df[mask].index:
                        try:
                            divergences[div_type].append({
                                'index': int(idx),
                                'date': str(df.loc[idx, 'date']),
                                'low': float(df.loc[idx, 'low']),
                                'rsi': float(df.loc[idx, 'RSI']) if 'RSI' in df.columns else 0.0,
                                'macd': float(df.loc[idx, 'macd_histogram']) if 'macd_histogram' in df.columns else 0.0,
                                'validation': {
                                    'strength_score': float(df.loc[idx, f"{div_type}_strength"]) if f"{div_type}_strength" in df.columns else 50.0,
                                    'data_quality': 'good' if not df.loc[idx].isna().any() else 'poor'
                                }
                            })
                        except Exception as e:
                            logger.warning(f"Fehler beim Extrahieren der Divergenz in Zeile {idx}: {e}")
                            continue
            logger.info(f"Divergenzen extrahiert: Classic={len(divergences['classic'])}, Hidden={len(divergences['hidden'])}")
            return divergences
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Divergenzen: {e}")
            return {'classic': [], 'hidden': []}

    def calculate_extrema_strength(self, df):
        """Berechnet die Stärke der Extrema."""
        try:
            if 'local_min_price' not in df.columns or 'local_max_price' not in df.columns:
                logger.warning("Keine lokalen Extrema-Spalten gefunden")
                df['extrema_strength'] = pd.Series([0.0] * len(df), index=df.index)
                return df
            
            df['extrema_strength'] = 0.0
            min_mask = df['local_min_price'] == True
            max_mask = df['local_max_price'] == True
            
            for idx in df[min_mask].index:
                try:
                    window = 5
                    start_idx = max(0, idx - window)
                    end_idx = min(len(df), idx + window + 1)
                    window_data = df.iloc[start_idx:end_idx]
                    price_range = window_data['high'].max() - window_data['low'].min()
                    if price_range > 0:
                        strength = ((window_data['close'].iloc[-1] - window_data['low'].min()) / price_range) * 100
                        df.loc[idx, 'extrema_strength'] = strength
                except Exception as e:
                    logger.warning(f"Fehler bei der Extrema-Stärke-Berechnung für Minimum in Zeile {idx}: {e}")
                    continue
            
            for idx in df[max_mask].index:
                try:
                    window = 5
                    start_idx = max(0, idx - window)
                    end_idx = min(len(df), idx + window + 1)
                    window_data = df.iloc[start_idx:end_idx]
                    price_range = window_data['high'].max() - window_data['low'].min()
                    if price_range > 0:
                        strength = ((window_data['high'].max() - window_data['close'].iloc[-1]) / price_range) * 100
                        df.loc[idx, 'extrema_strength'] = strength
                except Exception as e:
                    logger.warning(f"Fehler bei der Extrema-Stärke-Berechnung für Maximum in Zeile {idx}: {e}")
                    continue
            
            logger.info(f"Extrema-Stärke berechnet: {df['extrema_strength'].describe().to_dict()}")
            return df
        except Exception as e:
            logger.error(f"Fehler bei der Extrema-Stärke-Berechnung: {e}")
            df['extrema_strength'] = pd.Series([0.0] * len(df), index=df.index)
            return df

    def validate_divergence_performance(self, df, divergences):
        """Validiert die Performance von Divergenzen."""
        try:
            look_ahead = 5
            for div_type in ['classic', 'hidden']:
                for div in divergences[div_type]:
                    idx = div['index']
                    if idx + look_ahead >= len(df):
                        div['validation_result'] = {'status': 'pending', 'message': 'Zu wenig Daten für Validierung'}
                        continue
                    
                    try:
                        future_prices = df.iloc[idx:idx + look_ahead + 1]['close']
                        current_price = df.iloc[idx]['close']
                        price_change = ((future_prices.max() - current_price) / current_price) * 100
                        
                        div['validation_result'] = {
                            'status': 'success' if price_change > 2 else 'failed',
                            'message': f"Preisänderung: {price_change:.2f}% in {look_ahead} Kerzen"
                        }
                        
                        # Kontext-Daten
                        div['validation']['context_data'] = []
                        for i in range(max(0, idx - 2), min(len(df), idx + 3)):
                            div['validation']['context_data'].append({
                                'relative_index': i - idx,
                                'date': str(df.iloc[i]['date']),
                                'low': float(df.iloc[i]['low']),
                                'rsi': float(df.iloc[i]['RSI']) if 'RSI' in df.columns else 0.0,
                                'macd': float(df.iloc[i]['macd_histogram']) if 'macd_histogram' in df.columns else 0.0,
                                'is_signal': i == idx,
                                'is_price_min': bool(df.iloc[i].get('local_min_price', False)),
                                'is_rsi_min': bool(df.iloc[i].get('rsi_min', False)),
                                'is_macd_min': bool(df.iloc[i].get('macd_min', False))
                            })
                    except Exception as e:
                        logger.warning(f"Fehler bei der Validierung der Divergenz in Zeile {idx}: {e}")
                        div['validation_result'] = {'status': 'failed', 'message': f"Validierungsfehler: {e}"}
            
            logger.info(f"Divergenzen validiert")
            return divergences
        except Exception as e:
            logger.error(f"Fehler bei der Divergenz-Validierung: {e}")
            return divergences

    def _calculate_performance(self, df, divergences):
        """Berechnet die Performance-Statistiken."""
        try:
            performance = {'classic': 0, 'hidden': 0}
            for div_type in ['classic', 'hidden']:
                success_count = sum(1 for div in divergences[div_type] 
                                  if div.get('validation_result', {}).get('status') == 'success')
                total_count = len(divergences[div_type])
                performance[div_type] = (success_count / total_count * 100) if total_count > 0 else 0
            logger.info(f"Performance berechnet: {performance}")
            return performance
        except Exception as e:
            logger.error(f"Fehler bei der Performance-Berechnung: {e}")
            return {'classic': 0, 'hidden': 0}

    def analyze_dataframe(self, session_id, variants, date_range=None):
        """Hauptmethode zur Analyse des DataFrames."""
        try:
            if not self.modules_loaded:
                logger.warning("Keine Module geladen - Verwende Fallback-Analyse")
                # Fahre trotzdem fort mit Basis-Funktionalität
            
            # DataFrame aus der Session laden
            if session_id not in self.session_data:
                logger.error(f"Session {session_id} nicht gefunden")
                return {'success': False, 'error': f"Session {session_id} nicht gefunden"}
            
            df = self.session_data[session_id].copy()
            if df is None or df.empty:
                logger.error("Leerer DataFrame in der Session")
                return {'success': False, 'error': "Leerer DataFrame"}
            
            logger.info(f"🔍 Analysiere DataFrame aus Session {session_id}: {len(df)} Zeilen")
            
            # Optional: Datum-Filter
            if date_range:
                df = self._filter_date_range(df, date_range)
            
            # Technische Indikatoren berechnen
            logger.info("Berechne Indikatoren...")
            if 'Initialize_RSI_EMA_MACD' in self.modules_loaded:
                result = self._call_module_function('Initialize_RSI_EMA_MACD', 'Initialize_RSI_EMA_MACD', df)
                if result is not None:
                    df = result
                    logger.info(f"Indikatoren berechnet. Spalten: {df.columns.tolist()}")
                else:
                    logger.warning("Indikatoren-Berechnung fehlgeschlagen, verwende Fallback")
                    # Fallback: Erstelle minimale Indikatoren
                    df['RSI'] = 50.0  # Dummy RSI
                    df['macd_histogram'] = 0.0  # Dummy MACD
            else:
                logger.warning("Initialize_RSI_EMA_MACD nicht geladen, verwende Fallback")
                df['RSI'] = 50.0
                df['macd_histogram'] = 0.0
            
            # Lokale Extrema
            logger.info("Finde lokale Extrema...")
            if 'Local_Max_Min' in self.modules_loaded:
                self._call_module_function('Local_Max_Min', 'Local_Max_Min', df)
            else:
                logger.warning("Local_Max_Min nicht geladen, überspringe")
                df['local_min_price'] = False
                df['local_max_price'] = False
            
            df = self.calculate_extrema_strength(df)
            
            # Ergebnisse sammeln
            results = {}
            performance = {}
            
            # Für jede Variante analysieren
            for variant in variants:
                logger.info(f"Analysiere: {variant['name']}")
                df_var = df.copy()
                
                # Divergenz-Analyse
                try:
                    if 'CBullDivg_analysis' in self.modules_loaded:
                        self._call_module_function(
                            'CBullDivg_analysis', 'CBullDivg_Analysis_vectorized',
                            df_var, variant['window'], variant['candleTol'], variant['macdTol']
                        )
                    else:
                        logger.warning(f"CBullDivg_analysis nicht geladen für {variant['name']}")
                        # Erstelle leere Divergenz-Spalten
                        df_var['classic_divergence'] = False
                        df_var['hidden_divergence'] = False
                    
                    divergences = self._extract_divergences(df_var)
                    logger.info(f"Divergenzen für {variant['name']}: Classic={len(divergences['classic'])}, Hidden={len(divergences['hidden'])}")
                    
                    # Validierung
                    divergences = self.validate_divergence_performance(df_var, divergences)
                    results[variant['id']] = {
                        'classic': divergences['classic'],
                        'hidden': divergences['hidden'],
                        'total': len(divergences['classic']) + len(divergences['hidden'])
                    }
                    
                    # Annotationen
                    annotations = {'candlestick': [], 'rsi': [], 'macd': []}
                    if 'DivergenceArrows' in self.modules_loaded:
                        try:
                            arrow_result = self._call_module_function(
                                'DivergenceArrows', 'generate_arrows',
                                df_var, divergences, variant['window'], variant['name'], bullish=True
                            )
                            if arrow_result:
                                annotations = arrow_result
                        except Exception as e:
                            logger.warning(f"Fehler bei Annotationen für {variant['name']}: {e}")
                    else:
                        logger.warning(f"DivergenceArrows nicht geladen für {variant['name']}, verwende leere Annotationen")
                    
                    results[variant['id']]['annotations'] = annotations
                    
                    if variant.get('calculate_performance'):
                        performance[variant['id']] = self._calculate_performance(df_var, divergences)
                        
                except Exception as e:
                    logger.error(f"Variante {variant['name']} fehlgeschlagen: {e}")
                    results[variant['id']] = {
                        'classic': [], 
                        'hidden': [], 
                        'total': 0, 
                        'annotations': {'candlestick': [], 'rsi': [], 'macd': []}
                    }
            
            # Chart-Daten vorbereiten
            chart_data = self._prepare_chart_data(df)
            logger.info(f"Chart-Daten: {len(chart_data['dates'])} Zeilen")
            
            return {
                'chartData': chart_data,
                'results': results,
                'performance': performance,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Analyse-Fehler (DataFrame): {e}", exc_info=True)
            return {'success': False, 'error': str(e)}