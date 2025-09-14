"""
Vereinfachte interaktive Chartanalyse mit Plotly
Fokus auf Funktionalität ohne GUI-Komplexität
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog

# Eigene Module importieren
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min  # Original Version verwenden
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
    print("✅ Module erfolgreich importiert")
except ImportError as e:
    print(f"❌ Fehler beim Importieren der Module: {e}")
    sys.exit(1)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_data_file():
    """
    Einfacher Dateiauswahl-Dialog
    """
    root = tk.Tk()
    root.withdraw()  # Verstecke Haupt-Fenster
    
    # Startverzeichnis
    initial_dir = r"C:\Projekte\crt_250816\data\raw"
    if not Path(initial_dir).exists():
        initial_dir = os.getcwd()
    
    file_path = filedialog.askopenfilename(
        title="CSV/Parquet Datei auswählen",
        initialdir=initial_dir,
        filetypes=[
            ("CSV Dateien", "*.csv"),
            ("Parquet Dateien", "*.parquet"),
            ("Alle", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def load_and_prepare_data(file_path):
    """
    Lädt und bereitet Daten vor
    """
    logger.info(f"Lade Daten: {file_path}")
    
    # Daten laden
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
    
    # Datum konvertieren
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"✅ {len(df)} Zeilen geladen")
    return df

def calculate_all_indicators(df):
    """
    Berechnet alle technischen Indikatoren
    """
    logger.info("🔧 Berechne technische Indikatoren...")
    
    # RSI, EMA, MACD
    Initialize_RSI_EMA_MACD(df)
    
    # Lokale Extrema
    logger.info("🔍 Suche lokale Extrema...")
    Local_Max_Min(df)
    
    # Divergenzen
    logger.info("📈 Analysiere Bullish Divergenzen...")
    CBullDivg_analysis(df, 5, 0.1, 3.25)
    
    # Statistiken
    if 'CBullD_gen' in df.columns:
        classic_count = (df['CBullD_gen'] == 1).sum()
        logger.info(f"✅ Classic Bullish Divergenzen: {classic_count}")
    
    if 'CBullD_neg_MACD' in df.columns:
        neg_macd_count = (df['CBullD_neg_MACD'] == 1).sum()
        logger.info(f"✅ Negative MACD Divergenzen: {neg_macd_count}")
    
    return df

def create_plotly_chart(df):
    """
    Erstellt interaktives Plotly Chart
    """
    logger.info("📊 Erstelle interaktives Chart...")
    
    # 3-Panel Chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('BTC Preis & EMAs', 'RSI', 'MACD Histogram'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # ================== PANEL 1: CANDLESTICK & EMAs ==================
    
    # Candlestick Chart mit angepassten Farben für weißen Hintergrund
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTC Preis",
            increasing_line_color='#00cc44',  # Dunkler für weißen Hintergrund
            decreasing_line_color='#cc0000'   # Dunkler für weißen Hintergrund
        ),
        row=1, col=1
    )
    
    # EMAs mit dunkleren Farben für weißen Hintergrund
    ema_colors = {'EMA_20': '#cc3333', 'EMA_50': '#3366cc', 'EMA_100': '#cc9900', 'EMA_200': '#339966'}
    for ema, color in ema_colors.items():
        if ema in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[ema],
                    mode='lines',
                    name=ema,
                    line=dict(color=color, width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # ================== PANEL 2: RSI ==================
    
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#ffaa00', width=2)
            ),
            row=2, col=1
        )
        
        # RSI Überkauft/Überverkauft Linien
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # ================== PANEL 3: MACD ==================
    
    if 'macd_histogram' in df.columns:
        # Positive und negative Balken mit dunkleren Farben
        colors_macd = ['#00aa44' if val >= 0 else '#aa0000' for val in df['macd_histogram']]
        
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['macd_histogram'],
                name='MACD Histogram',
                marker_color=colors_macd,
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Zero-Linie
        fig.add_hline(y=0, line_color="gray", opacity=0.5, row=3, col=1)
    
    # ================== DIVERGENZ-MARKIERUNGEN ==================
    
    # Classic Bullish Divergenzen mit Nummerierung
    if 'CBullD_gen' in df.columns:
        divergence_data = df[df['CBullD_gen'] == 1]
        
        for divergence_num, (idx, row) in enumerate(divergence_data.iterrows(), 1):
            try:
                if pd.notna(row['CBullD_Lower_Low_date_gen']):
                    lower_date = pd.to_datetime(row['CBullD_Lower_Low_date_gen'])
                    higher_date = pd.to_datetime(row['CBullD_Higher_Low_date_gen'])
                    
                    # Preis-Divergenz-Markierungen mit roten/blauen Kreuzen
                    # Rotes X für Lower Low
                    fig.add_trace(
                        go.Scatter(
                            x=[lower_date],
                            y=[row['CBullD_Lower_Low_gen']],
                            mode='markers+text',
                            name=f'Classic Div #{divergence_num}',
                            marker=dict(size=12, symbol='x', color='red', line=dict(width=2)),
                            text=[f"{divergence_num}"],
                            textposition="top center",
                            textfont=dict(color='red', size=10, family='Arial Black'),
                            showlegend=False,
                            hovertemplate=f"<b>Classic Bullish Divergenz #{divergence_num}</b><br>" +
                                        f"Typ: Lower Low (Preis)<br>" +
                                        f"Datum: {lower_date.strftime('%Y-%m-%d')}<br>" +
                                        f"Preis: {row['CBullD_Lower_Low_gen']:.2f} USD<br>" +
                                        f"Zeitabstand: {row['CBullD_Date_Gap_gen']:.0f} Tage<extra></extra>"
                        ),
                        row=1, col=1
                    )
                    
                    # Blaues X für Higher Low
                    fig.add_trace(
                        go.Scatter(
                            x=[higher_date],
                            y=[row['CBullD_Higher_Low_gen']],
                            mode='markers+text',
                            name=f'Classic Div #{divergence_num}',
                            marker=dict(size=12, symbol='x', color='blue', line=dict(width=2)),
                            text=[f"{divergence_num}"],
                            textposition="top center",
                            textfont=dict(color='blue', size=10, family='Arial Black'),
                            showlegend=False,
                            hovertemplate=f"<b>Classic Bullish Divergenz #{divergence_num}</b><br>" +
                                        f"Typ: Higher Low (Preis)<br>" +
                                        f"Datum: {higher_date.strftime('%Y-%m-%d')}<br>" +
                                        f"Preis: {row['CBullD_Higher_Low_gen']:.2f} USD<br>" +
                                        f"Zeitabstand: {row['CBullD_Date_Gap_gen']:.0f} Tage<extra></extra>"
                        ),
                        row=1, col=1
                    )
                    
                    # RSI-Divergenz-Markierungen
                    if pd.notna(row['CBullD_Lower_Low_RSI_gen']):
                        # RSI Lower Low (rot)
                        fig.add_trace(
                            go.Scatter(
                                x=[lower_date],
                                y=[row['CBullD_Lower_Low_RSI_gen']],
                                mode='markers+text',
                                name=f'RSI Div #{divergence_num}',
                                marker=dict(size=10, symbol='x', color='red', line=dict(width=2)),
                                text=[f"{divergence_num}"],
                                textposition="top center",
                                textfont=dict(color='red', size=8, family='Arial Black'),
                                showlegend=False,
                                hovertemplate=f"<b>Classic Bullish Divergenz #{divergence_num}</b><br>" +
                                            f"Indikator: RSI Lower Low<br>" +
                                            f"Datum: {lower_date.strftime('%Y-%m-%d')}<br>" +
                                            f"RSI: {row['CBullD_Lower_Low_RSI_gen']:.1f}<br>" +
                                            f"Differenz: {row['CBullD_Higher_Low_RSI_gen'] - row['CBullD_Lower_Low_RSI_gen']:.1f}<extra></extra>"
                            ),
                            row=2, col=1
                        )
                        
                        # RSI Higher Low (blau)
                        fig.add_trace(
                            go.Scatter(
                                x=[higher_date],
                                y=[row['CBullD_Higher_Low_RSI_gen']],
                                mode='markers+text',
                                name=f'RSI Div #{divergence_num}',
                                marker=dict(size=10, symbol='x', color='blue', line=dict(width=2)),
                                text=[f"{divergence_num}"],
                                textposition="top center",
                                textfont=dict(color='blue', size=8, family='Arial Black'),
                                showlegend=False,
                                hovertemplate=f"<b>Classic Bullish Divergenz #{divergence_num}</b><br>" +
                                            f"Indikator: RSI Higher Low<br>" +
                                            f"Datum: {higher_date.strftime('%Y-%m-%d')}<br>" +
                                            f"RSI: {row['CBullD_Higher_Low_RSI_gen']:.1f}<br>" +
                                            f"Differenz: {row['CBullD_Higher_Low_RSI_gen'] - row['CBullD_Lower_Low_RSI_gen']:.1f}<extra></extra>"
                            ),
                            row=2, col=1
                        )
                    
                    # MACD-Divergenz-Markierungen
                    if pd.notna(row['CBullD_Lower_Low_MACD_gen']):
                        # MACD Lower Low (rot)
                        fig.add_trace(
                            go.Scatter(
                                x=[lower_date],
                                y=[row['CBullD_Lower_Low_MACD_gen']],
                                mode='markers+text',
                                name=f'MACD Div #{divergence_num}',
                                marker=dict(size=10, symbol='x', color='red', line=dict(width=2)),
                                text=[f"{divergence_num}"],
                                textposition="top center",
                                textfont=dict(color='red', size=8, family='Arial Black'),
                                showlegend=False,
                                hovertemplate=f"<b>Classic Bullish Divergenz #{divergence_num}</b><br>" +
                                            f"Indikator: MACD Lower Low<br>" +
                                            f"Datum: {lower_date.strftime('%Y-%m-%d')}<br>" +
                                            f"MACD: {row['CBullD_Lower_Low_MACD_gen']:.4f}<br>" +
                                            f"Differenz: {row['CBullD_Higher_Low_MACD_gen'] - row['CBullD_Lower_Low_MACD_gen']:.4f}<extra></extra>"
                            ),
                            row=3, col=1
                        )
                        
                        # MACD Higher Low (blau)
                        fig.add_trace(
                            go.Scatter(
                                x=[higher_date],
                                y=[row['CBullD_Higher_Low_MACD_gen']],
                                mode='markers+text',
                                name=f'MACD Div #{divergence_num}',
                                marker=dict(size=10, symbol='x', color='blue', line=dict(width=2)),
                                text=[f"{divergence_num}"],
                                textposition="top center",
                                textfont=dict(color='blue', size=8, family='Arial Black'),
                                showlegend=False,
                                hovertemplate=f"<b>Classic Bullish Divergenz #{divergence_num}</b><br>" +
                                            f"Indikator: MACD Higher Low<br>" +
                                            f"Datum: {higher_date.strftime('%Y-%m-%d')}<br>" +
                                            f"MACD: {row['CBullD_Higher_Low_MACD_gen']:.4f}<br>" +
                                            f"Differenz: {row['CBullD_Higher_Low_MACD_gen'] - row['CBullD_Lower_Low_MACD_gen']:.4f}<extra></extra>"
                            ),
                            row=3, col=1
                        )
            except Exception as e:
                logger.warning(f"Fehler bei Classic Divergenz #{divergence_num}: {e}")
                continue
    
    # Negative MACD Divergenzen mit separater Nummerierung
    if 'CBullD_neg_MACD' in df.columns:
        neg_divergence_data = df[df['CBullD_neg_MACD'] == 1]
        
        for neg_div_num, (idx, row) in enumerate(neg_divergence_data.iterrows(), 1):
            try:
                if pd.notna(row['CBullD_Lower_Low_date_neg_MACD']):
                    lower_date = pd.to_datetime(row['CBullD_Lower_Low_date_neg_MACD'])
                    higher_date = pd.to_datetime(row['CBullD_Higher_Low_date_neg_MACD'])
                    
                    # Negative MACD Divergenzen in Orange/Purple für Unterscheidung
                    # Preis-Markierungen
                    fig.add_trace(
                        go.Scatter(
                            x=[lower_date],
                            y=[row['CBullD_Lower_Low_neg_MACD']],
                            mode='markers+text',
                            name=f'Neg MACD #{neg_div_num}',
                            marker=dict(size=10, symbol='diamond', color='orange', line=dict(width=2)),
                            text=[f"N{neg_div_num}"],
                            textposition="top center",
                            textfont=dict(color='orange', size=8, family='Arial Black'),
                            showlegend=False,
                            hovertemplate=f"<b>Negative MACD Divergenz N{neg_div_num}</b><br>" +
                                        f"Typ: Lower Low (Preis)<br>" +
                                        f"Datum: {lower_date.strftime('%Y-%m-%d')}<br>" +
                                        f"Preis: {row['CBullD_Lower_Low_neg_MACD']:.2f} USD<br>" +
                                        f"Zeitabstand: {row['CBullD_Date_Gap_neg_MACD']:.0f} Tage<extra></extra>"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[higher_date],
                            y=[row['CBullD_Higher_Low_neg_MACD']],
                            mode='markers+text',
                            name=f'Neg MACD #{neg_div_num}',
                            marker=dict(size=10, symbol='diamond', color='purple', line=dict(width=2)),
                            text=[f"N{neg_div_num}"],
                            textposition="top center",
                            textfont=dict(color='purple', size=8, family='Arial Black'),
                            showlegend=False,
                            hovertemplate=f"<b>Negative MACD Divergenz N{neg_div_num}</b><br>" +
                                        f"Typ: Higher Low (Preis)<br>" +
                                        f"Datum: {higher_date.strftime('%Y-%m-%d')}<br>" +
                                        f"Preis: {row['CBullD_Higher_Low_neg_MACD']:.2f} USD<br>" +
                                        f"Zeitabstand: {row['CBullD_Date_Gap_neg_MACD']:.0f} Tage<extra></extra>"
                        ),
                        row=1, col=1
                    )
            except Exception as e:
                logger.warning(f"Fehler bei Negative MACD Divergenz N{neg_div_num}: {e}")
                continue
    
    # ================== LAYOUT ==================
    
    fig.update_layout(
        title=dict(
            text="Interaktive Technische Chartanalyse - Bitcoin Daily",
            x=0.5,
            font_size=16
        ),
        template="plotly_white",  # Weißer Hintergrund
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=900,
        hovermode='x unified'
    )
    
    # Y-Achsen mit Zoom-Funktionalität
    fig.update_yaxes(title_text="Preis (USD)", row=1, col=1, fixedrange=False)  # Y-Zoom aktiviert
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], fixedrange=False)  # Y-Zoom aktiviert
    fig.update_yaxes(title_text="MACD", row=3, col=1, fixedrange=False)  # Y-Zoom aktiviert
    fig.update_xaxes(title_text="Datum", row=3, col=1)
    
    return fig

def main():
    """
    Hauptfunktion für vereinfachte interaktive Analyse
    """
    try:
        print("🚀 Interaktive Chartanalyse startet...")
        
        # Datei auswählen
        file_path = select_data_file()
        if not file_path:
            print("❌ Keine Datei ausgewählt")
            return
        
        # Daten laden
        df = load_and_prepare_data(file_path)
        
        # Indikatoren berechnen
        df = calculate_all_indicators(df)
        
        # Chart erstellen
        fig = create_plotly_chart(df)
        
        # Chart anzeigen (öffnet Browser)
        print("📊 Öffne interaktives Chart im Browser...")
        
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'chart_analysis',
                'height': 900,
                'width': 1400,
                'scale': 2
            }
        }
        
        fig.show(config=config)
        
        print("✅ Interaktive Analyse erfolgreich!")
        print("💡 Nutze Zoom, Pan, Hover für Details")
        print("🎨 Zeichenwerkzeuge verfügbar")
        
    except Exception as e:
        logger.error(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
