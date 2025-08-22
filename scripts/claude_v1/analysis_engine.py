import pandas as pd
import numpy as np
import os
import logging
import importlib.util
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def safe_float(value, default=None):
    """Convert value to float, handling NaN values safely for JSON serialization."""
    try:
        if pd.isna(value) or value is None or np.isnan(float(value)):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_json_list(series):
    """Convert pandas series to JSON-safe list, handling NaN values."""
    return [safe_float(x, None) for x in series.tolist()]

class AnalysisEngine:
    def __init__(self, modules: List[str]):
        self.modules: List[Tuple[str, object]] = []
        self.data_sessions: Dict[str, pd.DataFrame] = {}
        logger.info("🔧 Initializing AnalysisEngine with modules: %s", modules)
        self.load_modules(modules)

    def load_modules(self, modules: List[str]):
        """Load required modules dynamically."""
        for module_name in modules:
            try:
                spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
                if spec is None or spec.loader is None:
                    raise ImportError(f"Spec/Loader für {module_name} ist None")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.modules.append((module_name, module))
                logger.info(f"✅ Modul {module_name} erfolgreich geladen")
            except Exception as e:
                logger.error(f"❌ Modul {module_name} nicht gefunden/ladbar: {e}")
                raise

    def load_data(self, df: pd.DataFrame) -> Optional[str]:
        """Load data into session storage."""
        try:
            session_id = str(uuid.uuid4())
            self.data_sessions[session_id] = df
            logger.info(f"✅ Session {session_id} erstellt mit {len(df)} Zeilen")
            return session_id
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Daten: {e}")
            return None

    def initialize_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Initialize technical indicators using Initialize_RSI_EMA_MACD."""
        try:
            for module_name, module in self.modules:
                if module_name == "Initialize_RSI_EMA_MACD" and module:
                    logger.info("📊 Initializing RSI/EMA/MACD...")
                    df = module.Initialize_RSI_EMA_MACD(df)
                    if df is None:
                        logger.error("❌ Initialize_RSI_EMA_MACD returned None")
                        return None
            logger.info("✅ DataFrame initialized")
            return df
        except Exception as e:
            logger.error(f"❌ Fehler bei der DataFrame-Initialisierung: {e}")
            return None

    def analyze_dataframe(self, session_id: str, variants: List[Dict], date_range: Optional[Dict] = None) -> Dict:
        """Analyze dataframe for bullish divergences, replicating all_in_one_analyzer.py."""
        try:
            if session_id not in self.data_sessions:
                logger.error(f"❌ Session {session_id} nicht gefunden")
                return {"success": False, "error": "Session nicht gefunden"}

            df = self.data_sessions[session_id].copy()
            logger.info(f"📈 Analysiere DataFrame aus Session {session_id}: {len(df)} Zeilen")

            # Initialize technical indicators
            df = self.initialize_dataframe(df)
            if df is None:
                logger.error("❌ DataFrame initialization failed")
                return {"success": False, "error": "Initialisierung fehlgeschlagen"}

            # Calculate local maxima/minima
            for module_name, module in self.modules:
                if module_name == "Local_Maximas_Minimas" and module:
                    logger.info("📊 Calculating local maxima/minima...")
                    module.Local_Max_Min(df)

            # Analyze divergences for each variant
            results = {}
            for variant in variants:
                logger.info(f"📊 Analysiere Variante: {variant['name']}")
                df_var = df.copy()
                for module_name, module in self.modules:
                    if module_name == "CBullDivg_Analysis_vectorized" and module:
                        logger.info(f"📊 Running CBullDivg_analysis with window={variant['window']}, candleTol={variant['candleTol']}, macdTol={variant['macdTol']}")
                        module.CBullDivg_analysis(df_var, variant['window'], variant['candleTol'], variant['macdTol'])

                classic = []
                hidden = []
                for i in range(len(df_var)):
                    row = df_var.iloc[i]
                    if row.get('CBullD_gen', 0) == 1:
                        classic.append({
                            'date': str(row['date']),
                            'low': float(row['low']),
                            'rsi': float(row.get('RSI', 0)),
                            'macd': float(row.get('macd_histogram', 0))
                        })
                    if row.get('CBullD_neg_MACD', 0) == 1:
                        hidden.append({
                            'date': str(row['date']),
                            'low': float(row['low']),
                            'rsi': float(row.get('RSI', 0)),
                            'macd': float(row.get('macd_histogram', 0))
                        })

                results[variant['id']] = {
                    'classic': classic,
                    'hidden': hidden,
                    'total': len(classic) + len(hidden)
                }
                logger.info(f"✅ Variante {variant['name']} analysiert: {len(classic)} classic, {len(hidden)} hidden")

            # Prepare chart data with EMAs
            chart_data = {
                'dates': df['date'].astype(str).tolist(),
                'open': safe_json_list(df['open']),
                'high': safe_json_list(df['high']),
                'low': safe_json_list(df['low']),
                'close': safe_json_list(df['close']),
                'rsi': safe_json_list(df.get('RSI', pd.Series()).fillna(0)),
                'macd_histogram': safe_json_list(df.get('macd_histogram', pd.Series()).fillna(0)),
                'ema20': safe_json_list(df.get('EMA_20', pd.Series()).fillna(0)) if 'EMA_20' in df else None,
                'ema50': safe_json_list(df.get('EMA_50', pd.Series()).fillna(0)) if 'EMA_50' in df else None,
                'ema100': safe_json_list(df.get('EMA_100', pd.Series()).fillna(0)) if 'EMA_100' in df else None,
                'ema200': safe_json_list(df.get('EMA_200', pd.Series()).fillna(0)) if 'EMA_200' in df else None
            }

            logger.info(f"✅ Analyse abgeschlossen: {len(chart_data['dates'])} Datenpunkte")
            return {
                'success': True,
                'chartData': chart_data,
                'results': results
            }

        except Exception as e:
            logger.error(f"❌ Fehler bei der Analyse: {e}")
            return {"success": False, "error": str(e)}