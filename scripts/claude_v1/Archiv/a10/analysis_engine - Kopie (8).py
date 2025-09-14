import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import importlib.util
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class AnalysisEngine:
    def __init__(self, modules: List[str]):
        self.modules = []
        self.data_sessions = {}
        self.load_modules(modules)

    def load_modules(self, modules: List[str]):
        for module_name in modules:
            try:
                spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.modules.append((module_name, module))
                logger.info(f"Modul {module_name} erfolgreich geladen")
            except Exception as e:
                logger.warning(f"Modul {module_name} nicht gefunden, verwende Fallback: {e}")
                if module_name == "DivergenceArrows":
                    self.modules.append((module_name, None))  # Fallback für optionales Modul

    def initialize_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            for module_name, module in self.modules:
                if module_name == "Initialize_RSI_EMA_MACD" and module:
                    df = module.initialize_technical_indicators(df)
                    if df is None:
                        logger.error("Fehler bei der Initialisierung der technischen Indikatoren")
                        return None
            return df
        except Exception as e:
            logger.error(f"Fehler bei der DataFrame-Initialisierung: {e}")
            return None

    def analyze_dataframe(self, session_id: str, variants: List[Dict], date_range: Optional[Dict] = None) -> Dict:
        try:
            if session_id not in self.data_sessions:
                logger.error(f"Session {session_id} nicht gefunden")
                return {"success": False, "error": "Session nicht gefunden"}
            
            df = self.data_sessions[session_id].copy()
            logger.info(f"Analysiere DataFrame aus Session {session_id}: {len(df)} Zeilen")
            
            # Initialisiere technische Indikatoren
            df = self.initialize_dataframe(df)
            if df is None:
                return {"success": False, "error": "Initialisierung fehlgeschlagen"}
            
            # Finde lokale Extrema
            for module_name, module in self.modules:
                if module_name == "Local_Maximas_Minimas" and module:
                    try:
                        df = module.Local_Max_Min(df)
                        logger.info("Finde lokale Extrema...")
                    except Exception as e:
                        logger.error(f"Fehler bei Local_Maximas_Minimas: {e}")
                        return {"success": False, "error": f"Fehler bei Local_Maximas_Minimas: {e}"}
            
            results = {}
            performance = {}
            base_results = None
            
            # Analysiere jede Variante
            for variant in variants:
                variant_id = str(variant.get("id", uuid.uuid4()))
                window = variant.get("window", 5)
                candle_tol = variant.get("candleTol", 0.1)
                macd_tol = variant.get("macdTol", 3.25)
                calculate_performance = variant.get("calculate_performance", True)
                
                logger.info(f"Analysiere: {variant.get('name', 'Unbenannt')}")
                
                df_variant = df.copy()
                for module_name, module in self.modules:
                    if module_name == "CBullDivg_Analysis_vectorized" and module:
                        try:
                            df_variant = module.CBullDivg_analysis(df_variant, window, candle_tol, macd_tol)
                        except Exception as e:
                            logger.error(f"Fehler bei CBullDivg_Analysis_vectorized: {e}")
                            return {"success": False, "error": f"Fehler bei CBullDivg_Analysis_vectorized: {e}"}
                
                # Extrahiere Divergenzen
                divergences = self._extract_divergences(df_variant, variant)
                if variant.get("name") == "Standard":
                    base_results = divergences
                
                # Generiere Annotationen
                annotations = {"candlestick": [], "rsi": [], "macd": []}
                for module_name, module in self.modules:
                    if module_name == "DivergenceArrows" and module:
                        try:
                            annotations = module.DivergenceArrows.generate_arrows(
                                df_variant, divergences, window, variant.get("name", "Unbenannt")
                            )
                        except Exception as e:
                            logger.warning(f"Fehler bei DivergenceArrows: {e}, verwende leere Annotationen")
                
                # Berechne Performance (falls erforderlich)
                perf = self._calculate_performance(df_variant, divergences) if calculate_performance else {}
                
                results[variant_id] = {
                    "classic": divergences.get("classic", []),
                    "hidden": divergences.get("hidden", []),
                    "total": len(divergences.get("classic", [])) + len(divergences.get("hidden", [])),
                    "annotations": annotations
                }
                performance[variant_id] = perf
                
                # Vergleich zur Basis-Variante
                if base_results and variant.get("name") != "Standard":
                    results[variant_id]["compare_to_base"] = self._compare_to_base(divergences, base_results)
            
            # Erstelle Chart-Daten
            chart_data = self._create_chart_data(df)
            
            logger.info("Analyse erfolgreich abgeschlossen")
            return {
                "success": True,
                "chartData": chart_data,
                "results": results,
                "performance": performance
            }
        
        except Exception as e:
            logger.error(f"Analyse-Fehler (DataFrame): {e}")
            return {"success": False, "error": f"Analyse-Fehler: {e}"}

    def _extract_divergences(self, df: pd.DataFrame, variant: Dict) -> Dict:
        try:
            divergences = {'classic': [], 'hidden': []}
            for div_type, col in [('classic', 'CBullD_gen'), ('hidden', 'CBullD_neg_MACD')]:
                if col in df.columns:
                    mask = df[col] == 1  # Divergenzen sind als 1 markiert
                    for idx in df[mask].index:
                        try:
                            validation = {
                                'strength_score': float(df.loc[idx, f'CBullD_Date_Gap_{"gen" if div_type == "classic" else "neg_MACD"}']) / variant.get("window", 5),
                                'data_quality': 'good' if not df.loc[idx].isna().any() else 'poor',
                                'status': 'confirmed' if df.loc[idx, f'CBullD_Lower_Low_{"gen" if div_type == "classic" else "neg_MACD"}'] > 0 else 'pending',
                                'message': f"{div_type.capitalize()} divergence detected based on {'lower low' if div_type == 'classic' else 'higher low'} in price and {'higher low' if div_type == 'classic' else 'lower low'} in RSI/MACD"
                            }
                            divergences[div_type].append({
                                'index': int(idx),
                                'date': str(df.loc[idx, 'date']),
                                'low': float(df.loc[idx, 'low']),
                                'rsi': float(df.loc[idx, 'RSI']) if 'RSI' in df.columns else 0.0,
                                'macd': float(df.loc[idx, 'macd_histogram']) if 'macd_histogram' in df.columns else 0.0,
                                'validation': validation,
                                'lower_low': float(df.loc[idx, f'CBullD_Lower_Low_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'higher_low': float(df.loc[idx, f'CBullD_Higher_Low_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'lower_low_rsi': float(df.loc[idx, f'CBullD_Lower_Low_RSI_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'higher_low_rsi': float(df.loc[idx, f'CBullD_Higher_Low_RSI_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'lower_low_macd': float(df.loc[idx, f'CBullD_Lower_Low_MACD_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'higher_low_macd': float(df.loc[idx, f'CBullD_Higher_Low_MACD_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'lower_low_date': str(df.loc[idx, f'CBullD_Lower_Low_date_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'higher_low_date': str(df.loc[idx, f'CBullD_Higher_Low_date_{"gen" if div_type == "classic" else "neg_MACD"}']),
                                'date_gap': float(df.loc[idx, f'CBullD_Date_Gap_{"gen" if div_type == "classic" else "neg_MACD"}'])
                            })
                        except Exception as e:
                            logger.warning(f"Fehler beim Extrahieren der Divergenz in Zeile {idx}: {e}")
                            continue
                logger.info(f"Divergenzen extrahiert: {div_type.capitalize()}={len(divergences[div_type])}")
            return divergences
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren der Divergenzen: {e}")
            return {'classic': [], 'hidden': []}

    def _create_chart_data(self, df: pd.DataFrame) -> Dict:
        try:
            chart_data = {
                'dates': df['date'].astype(str).tolist(),
                'open': df['open'].astype(float).tolist(),
                'high': df['high'].astype(float).tolist(),
                'low': df['low'].astype(float).tolist(),
                'close': df['close'].astype(float).tolist(),
                'rsi': df['RSI'].astype(float).tolist() if 'RSI' in df.columns else [],
                'macd_histogram': df['macd_histogram'].astype(float).tolist() if 'macd_histogram' in df.columns else []
            }
            for span in [12, 20, 26, 50, 100, 200]:
                ema_col = f'EMA_{span}'
                if ema_col in df.columns:
                    chart_data[f'ema{span}'] = df[ema_col].astype(float).tolist()
            return chart_data
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Chart-Daten: {e}")
            return {}

    def _calculate_performance(self, df: pd.DataFrame, divergences: Dict) -> Dict:
        try:
            performance = {'classic': 0.0, 'hidden': 0.0}
            for div_type in ['classic', 'hidden']:
                if divergences[div_type]:
                    total_divs = len(divergences[div_type])
                    strong_divs = sum(1 for div in divergences[div_type] if div['validation']['strength_score'] > 0.7)
                    performance[div_type] = (strong_divs / total_divs * 100) if total_divs > 0 else 0.0
            return performance
        except Exception as e:
            logger.error(f"Fehler bei der Performance-Berechnung: {e}")
            return {'classic': 0.0, 'hidden': 0.0}

    def _compare_to_base(self, divergences: Dict, base_results: Dict) -> Dict:
        try:
            compare = {'classic': [], 'hidden': []}
            for div_type in ['classic', 'hidden']:
                base_dates = {div['date'] for div in base_results.get(div_type, [])}
                for div in divergences.get(div_type, []):
                    if div['date'] not in base_dates:
                        compare[div_type].append(div)
            return compare
        except Exception as e:
            logger.error(f"Fehler beim Vergleich zur Basis-Variante: {e}")
            return {'classic': [], 'hidden': []}

    def load_data(self, df: pd.DataFrame) -> str:
        try:
            session_id = str(uuid.uuid4())
            self.data_sessions[session_id] = df
            logger.info(f"Session {session_id} erstellt mit {len(df)} Zeilen")
            return session_id
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {e}")
            return None