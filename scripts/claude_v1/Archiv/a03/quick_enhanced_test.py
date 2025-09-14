"""
Quick Enhanced Test - Funktioniert garantiert mit den bestehenden Modulen
Erweitert das funktionierende interactive_simple.py um neue Divergenzen
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# Bestehende Module (die funktionieren)
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_simple_hidden_bullish(df):
    """
    Einfache Hidden Bullish Divergenzen
    """
    logger.info("🔍 Analysiere Hidden Bullish Divergenzen...")
    
    n = len(df)
    hidden_signals = []
    
    # Finde Hidden Bullish basierend auf RSI und Preis
    for i in range(20, n-5):  # Genug Abstand für Vergleiche
        try:
            # Suche nach zwei lokalen Lows
            window = 10
            price_lows = []
            rsi_lows = []
            dates_found = []
            
            # Finde lokale Lows in einem Fenster
            for j in range(i-window, i+window):
                if j > 5 and j < n-5:
                    # Prüfe ob lokales Low
                    if (df.iloc[j]['low'] < df.iloc[j-1]['low'] and 
                        df.iloc[j]['low'] < df.iloc[j+1]['low'] and
                        df.iloc[j]['low'] < df.iloc[j-2]['low'] and 
                        df.iloc[j]['low'] < df.iloc[j+2]['low']):
                        
                        price_lows.append(df.iloc[j]['low'])
                        rsi_lows.append(df.iloc[j]['RSI'])
                        dates_found.append(df.iloc[j]['date'])
            
            # Prüfe Hidden Bullish Bedingung
            if len(price_lows) >= 2:
                # Nehme die letzten zwei Lows
                recent_low = price_lows[-1]
                previous_low = price_lows[-2]
                recent_rsi = rsi_lows[-1]
                previous_rsi = rsi_lows[-2]
                
                # Hidden Bullish: Preis macht höhere Lows, RSI niedrigere Lows
                if (recent_low > previous_low and recent_rsi < previous_rsi and
                    abs(recent_low - previous_low) / previous_low > 0.01):  # 1% Unterschied
                    
                    hidden_signals.append({
                        'date': dates_found[-1],
                        'price': recent_low,
                        'rsi': recent_rsi,
                        'type': 'Hidden_Bullish'
                    })
        except:
            continue
    
    logger.info(f"✅ Hidden Bullish Divergenzen gefunden: {len(hidden_signals)}")
    return hidden_signals

def add_simple_bearish(df):
    """
    Einfache Bearish Divergenzen
    """
    logger.info("🔍 Analysiere Bearish Divergenzen...")
    
    n = len(df)
    bearish_signals = []
    
    # Finde Bearish basierend auf RSI und Preis
    for i in range(20, n-5):
        try:
            window = 10
            price_highs = []
            rsi_highs = []
            dates_found = []
            
            # Finde lokale Highs
            for j in range(i-window, i+window):
                if j > 5 and j < n-5:
                    # Prüfe ob lokales High
                    if (df.iloc[j]['high'] > df.iloc[j-1]['high'] and 
                        df.iloc[j]['high'] > df.iloc[j+1]['high'] and
                        df.iloc[j]['high'] > df.iloc[j-2]['high'] and 
                        df.iloc[j]['high'] > df.iloc[j+2]['high']):
                        
                        price_highs.append(df.iloc[j]['high'])
                        rsi_highs.append(df.iloc[j]['RSI'])
                        dates_found.append(df.iloc[j]['date'])
            
            # Prüfe Bearish Bedingung
            if len(price_highs) >= 2:
                recent_high = price_highs[-1]
                previous_high = price_highs[-2]
                recent_rsi = rsi_highs[-1]
                previous_rsi = rsi_highs[-2]
                
                # Bearish: Preis macht höhere Highs, RSI niedrigere Highs
                if (recent_high > previous_high and recent_rsi < previous_rsi and
                    abs(recent_high - previous_high) / previous_high > 0.01 and
                    recent_rsi > 50):  # Nur bei höheren RSI-Werten
                    
                    bearish_signals.append({
                        'date': dates_found[-1],
                        'price': recent_high,
                        'rsi': recent_rsi,
                        'type': 'Bearish'
                    })
        except:
            continue
    
    logger.info(f"✅ Bearish Divergenzen gefunden: {len(bearish_signals)}")
    return bearish_signals

def create_enhanced_chart(df, hidden_signals, bearish_signals):
    """
    Erstellt Chart mit allen Divergenz-Typen
    """
    # 3-Panel Chart wie vorher
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Preis & EMAs + ALLE Divergenzen', 'RSI', 'MACD Histogram'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Panel 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTC Preis",
            increasing_line_color='#00cc44',
            decreasing_line_color='#cc0000'
        ),
        row=1, col=1
    )
    
    # EMAs
    ema_colors = {'EMA_20': '#cc3333', 'EMA_50': '#3366cc', 'EMA_100': '#cc9900', 'EMA_200': '#339966'}
    for ema, color in ema_colors.items():
        if ema in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=df[ema], mode='lines', name=ema,
                    line=dict(color=color, width=1.5), opacity=0.8
                ),
                row=1, col=1
            )
    
    # Panel 2: RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI',
                      line=dict(color='#ff8800', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # Panel 3: MACD
    if 'macd_histogram' in df.columns:
        colors_macd = ['#00aa44' if val >= 0 else '#aa0000' for val in df['macd_histogram']]
        fig.add_trace(
            go.Bar(x=df['date'], y=df['macd_histogram'], name='MACD Histogram',
                  marker_color=colors_macd, opacity=0.7),
            row=3, col=1
        )
        fig.add_hline(y=0, line_color="gray", opacity=0.5, row=3, col=1)
    
    # CLASSIC BULLISH DIVERGENZEN (wie vorher)
    if 'CBullD_gen' in df.columns:
        divergence_data = df[df['CBullD_gen'] == 1]
        for div_num, (idx, row) in enumerate(divergence_data.iterrows(), 1):
            try:
                if pd.notna(row['CBullD_Lower_Low_date_gen']):
                    lower_date = pd.to_datetime(row['CBullD_Lower_Low_date_gen'])
                    higher_date = pd.to_datetime(row['CBullD_Higher_Low_date_gen'])
                    
                    # Rote und blaue X wie vorher
                    fig.add_trace(
                        go.Scatter(
                            x=[lower_date], y=[row['CBullD_Lower_Low_gen']],
                            mode='markers+text', name=f'Classic #{div_num}',
                            marker=dict(size=12, symbol='x', color='red', line=dict(width=2)),
                            text=[f"C{div_num}"], textposition="top center",
                            textfont=dict(color='red', size=10, family='Arial Black'),
                            showlegend=False,
                            hovertemplate=f"Classic Bullish #{div_num}<br>Datum: {lower_date.strftime('%Y-%m-%d')}<br>Preis: {row['CBullD_Lower_Low_gen']:.2f}<extra></extra>"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[higher_date], y=[row['CBullD_Higher_Low_gen']],
                            mode='markers+text', name=f'Classic #{div_num}',
                            marker=dict(size=12, symbol='x', color='blue', line=dict(width=2)),
                            text=[f"C{div_num}"], textposition="top center",
                            textfont=dict(color='blue', size=10, family='Arial Black'),
                            showlegend=False,
                            hovertemplate=f"Classic Bullish #{div_num}<br>Datum: {higher_date.strftime('%Y-%m-%d')}<br>Preis: {row['CBullD_Higher_Low_gen']:.2f}<extra></extra>"
                        ),
                        row=1, col=1
                    )
            except:
                continue
    
    # HIDDEN BULLISH DIVERGENZEN (neue Kreise)
    for i, signal in enumerate(hidden_signals, 1):
        fig.add_trace(
            go.Scatter(
                x=[signal['date']], y=[signal['price']],
                mode='markers+text', name=f'Hidden #{i}',
                marker=dict(size=10, symbol='circle', color='orange', line=dict(width=2)),
                text=[f"H{i}"], textposition="top center",
                textfont=dict(color='orange', size=9, family='Arial Black'),
                showlegend=False,
                hovertemplate=f"Hidden Bullish #{i}<br>Datum: {signal['date'].strftime('%Y-%m-%d')}<br>Preis: {signal['price']:.2f}<br>RSI: {signal['rsi']:.1f}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # BEARISH DIVERGENZEN (neue Dreiecke)
    for i, signal in enumerate(bearish_signals, 1):
        fig.add_trace(
            go.Scatter(
                x=[signal['date']], y=[signal['price']],
                mode='markers+text', name=f'Bearish #{i}',
                marker=dict(size=10, symbol='triangle-up', color='darkred', line=dict(width=2)),
                text=[f"B{i}"], textposition="top center",
                textfont=dict(color='darkred', size=9, family='Arial Black'),
                showlegend=False,
                hovertemplate=f"Bearish #{i}<br>Datum: {signal['date'].strftime('%Y-%m-%d')}<br>Preis: {signal['price']:.2f}<br>RSI: {signal['rsi']:.1f}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Layout
    fig.update_layout(
        title=dict(text="Erweiterte Divergenz-Analyse - Alle Typen", x=0.5, font_size=16),
        template="plotly_white", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50), height=900, hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Preis (USD)", row=1, col=1, fixedrange=False)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], fixedrange=False)
    fig.update_yaxes(title_text="MACD", row=3, col=1, fixedrange=False)
    fig.update_xaxes(title_text="Datum", row=3, col=1)
    
    return fig

def main():
    """
    Hauptfunktion für schnellen Test
    """
    logger.info("🚀 Quick Enhanced Divergence Test")
    
    # Datei auswählen
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Wähle BTC-Daten für erweiterte Analyse",
        filetypes=[("CSV Dateien", "*.csv"), ("Parquet Dateien", "*.parquet")]
    )
    root.destroy()
    
    if not file_path:
        print("❌ Keine Datei ausgewählt")
        return
    
    # Daten laden
    logger.info(f"📊 Lade Daten...")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, low_memory=False)
    else:
        df = pd.read_parquet(file_path)
    
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"✅ {len(df)} Zeilen geladen")
    
    # Basis-Indikatoren
    logger.info("🔧 Berechne Indikatoren...")
    Initialize_RSI_EMA_MACD(df)
    Local_Max_Min(df)
    
    # Standard-Divergenzen
    logger.info("📈 Standard-Divergenzen...")
    CBullDivg_analysis(df, 5, 0.1, 3.25)
    classic_count = (df['CBullD_gen'] == 1).sum()
    neg_macd_count = (df['CBullD_neg_MACD'] == 1).sum()
    
    # Neue Divergenzen
    hidden_signals = add_simple_hidden_bullish(df)
    bearish_signals = add_simple_bearish(df)
    
    # Zusammenfassung
    logger.info("\n📊 GESAMTERGEBNISSE:")
    logger.info("-" * 40)
    logger.info(f"Classic Bullish  : {classic_count:>3}")
    logger.info(f"Negative MACD    : {neg_macd_count:>3}")
    logger.info(f"Hidden Bullish   : {len(hidden_signals):>3} <- NEU!")
    logger.info(f"Classic Bearish  : {len(bearish_signals):>3} <- NEU!")
    logger.info(f"TOTAL           : {classic_count + neg_macd_count + len(hidden_signals) + len(bearish_signals):>3}")
    
    # Chart erstellen
    logger.info("\n📊 Erstelle erweitertes Chart...")
    fig = create_enhanced_chart(df, hidden_signals, bearish_signals)
    fig.show()
    
    # Excel Export
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_file = f"quick_enhanced_analysis_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Zusammenfassung
        summary = pd.DataFrame([
            ['Classic Bullish', classic_count],
            ['Negative MACD', neg_macd_count],
            ['Hidden Bullish', len(hidden_signals)],
            ['Classic Bearish', len(bearish_signals)],
            ['TOTAL', classic_count + neg_macd_count + len(hidden_signals) + len(bearish_signals)]
        ], columns=['Divergence Type', 'Count'])
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    logger.info(f"📋 Excel exportiert: {excel_file}")
    logger.info("✅ Erweiterte Analyse abgeschlossen!")

if __name__ == "__main__":
    main()
