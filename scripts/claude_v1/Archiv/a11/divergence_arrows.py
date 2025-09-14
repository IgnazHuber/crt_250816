import pandas as pd
import numpy as np
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DivergenceArrows:
    """Modul zur Generierung von Pfeil-Annotationen für Divergenzen."""
    
    @staticmethod
    def generate_arrows(df, divergences, window, variant_name, bullish=True):
        """Generiert Pfeil-Annotationen für Candlestick-, RSI- und MACD-Plots."""
        annotations = {
            'candlestick': [],
            'rsi': [],
            'macd': []
        }
        
        try:
            # Validierung der Eingabedaten
            required_columns = ['date', 'high', 'low', 'RSI', 'macd_histogram']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Fehlende Spalten in DataFrame: {missing_columns}")
                return annotations
            
            for div_type in ['classic', 'hidden']:
                if div_type not in divergences:
                    logger.warning(f"Keine {div_type} Divergenzen gefunden")
                    continue
                
                for div in divergences[div_type]:
                    idx = div.get('index')
                    if not isinstance(idx, int) or idx - window < 0 or idx >= len(df):
                        logger.warning(f"Ungültiger Index für Divergenz {div_type}: {idx}")
                        continue
                    
                    try:
                        # Sicherstellen, dass die benötigten Werte vorhanden und gültig sind
                        if any(np.isnan(df.iloc[idx][col]) for col in ['high', 'low', 'RSI', 'macd_histogram']):
                            logger.warning(f"Ungültige Werte in Zeile {idx} für {div_type}: {df.iloc[idx][['high', 'low', 'RSI', 'macd_histogram']].to_dict()}")
                            continue
                        
                        # Farbe basierend auf bullisch/bärisch
                        color = '#00FF00' if bullish else '#FF0000'
                        y_position_candlestick = df.iloc[idx]['high'] * 1.02 if bullish else df.iloc[idx]['low'] * 0.98
                        y_position_rsi = df.iloc[idx]['RSI'] * (1.05 if bullish else 0.95)
                        y_position_macd = df.iloc[idx]['macd_histogram'] * (1.05 if bullish else 0.95)
                        
                        # Pfeil-Länge basierend auf Fenster
                        x_start = str(df.iloc[max(0, idx - window)]['date'])
                        x_end = str(df.iloc[idx]['date'])
                        
                        # Candlestick-Pfeil
                        annotations['candlestick'].append({
                            'x': x_end,
                            'y': y_position_candlestick,
                            'xref': 'x',
                            'yref': 'y',
                            'ax': x_start,
                            'ay': y_position_candlestick,
                            'showarrow': True,
                            'arrowhead': 2,
                            'arrowcolor': color,
                            'arrowsize': 1.5,
                            'arrowwidth': 2,
                            'path': f'M {x_start},{y_position_candlestick} Q {x_end},{y_position_candlestick * (1.1 if bullish else 0.9)} {x_end},{y_position_candlestick}',
                            'text': ''
                        })
                        
                        # RSI-Pfeil
                        annotations['rsi'].append({
                            'x': x_end,
                            'y': y_position_rsi,
                            'xref': 'x',
                            'yref': 'y2',
                            'ax': x_start,
                            'ay': y_position_rsi,
                            'showarrow': True,
                            'arrowhead': 2,
                            'arrowcolor': color,
                            'arrowsize': 1.5,
                            'arrowwidth': 2,
                            'path': f'M {x_start},{y_position_rsi} Q {x_end},{y_position_rsi * (1.1 if bullish else 0.9)} {x_end},{y_position_rsi}',
                            'text': ''
                        })
                        
                        # MACD-Pfeil mit Divergenz-Namen
                        div_name = f"{variant_name[:3]}_{div_type[0].upper()}"  # z.B. "Std_C" für Standard Classic
                        annotations['macd'].append({
                            'x': x_end,
                            'y': y_position_macd,
                            'xref': 'x',
                            'yref': 'y3',
                            'ax': x_start,
                            'ay': y_position_macd,
                            'showarrow': True,
                            'arrowhead': 2,
                            'arrowcolor': color,
                            'arrowsize': 1.5,
                            'arrowwidth': 2,
                            'path': f'M {x_start},{y_position_macd} Q {x_end},{y_position_macd * (1.1 if bullish else 0.9)} {x_end},{y_position_macd}',
                            'text': div_name,
                            'textangle': 0,
                            'font': {'size': 12, 'color': color}
                        })
                        
                    except KeyError as e:
                        logger.error(f"Fehlender Schlüssel in Zeile {idx} für {div_type}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Fehler bei der Annotation-Generierung für Index {idx}, {div_type}: {e}")
                        continue
            
            logger.info(f"Annotationen generiert: Candlestick={len(annotations['candlestick'])}, RSI={len(annotations['rsi'])}, MACD={len(annotations['macd'])}")
            return annotations
            
        except Exception as e:
            logger.error(f"Fehler in generate_arrows: {e}")
            return annotations