#!/usr/bin/env python3
"""
Divergenz Analyzer GUI
Grafische Oberfläche für die Analysis Engine
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime
import mplfinance as mpf
from analysis_engine import AnalysisEngine, AnalysisVariant, DivergenceType
import threading
import json


class DivergenceAnalyzerGUI:
    """Haupt-GUI-Klasse"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Divergenz Analyzer v2.0")
        self.root.geometry("1400x900")
        
        # Style
        self.setup_styles()
        
        # Engine
        self.engine = AnalysisEngine()
        self.current_file = None
        
        # GUI aufbauen
        self.create_widgets()
        
    def setup_styles(self):
        """Konfiguriert Styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Farben
        self.colors = {
            'bg': '#f0f0f0',
            'primary': '#667eea',
            'success': '#48bb78',
            'danger': '#f56565',
            'warning': '#ed8936',
            'dark': '#2d3748',
            'light': '#ffffff'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Button Styles
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 10, 'bold'))
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
    def create_widgets(self):
        """Erstellt alle GUI-Elemente"""
        # Header
        self.create_header()
        
        # Control Panel
        self.create_control_panel()
        
        # Chart Area
        self.create_chart_area()
        
        # Results Panel
        self.create_results_panel()
        
        # Status Bar
        self.create_status_bar()
        
    def create_header(self):
        """Erstellt Header-Bereich"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(header_frame, 
                        text="📊 DIVERGENZ ANALYZER",
                        font=('Arial', 24, 'bold'),
                        bg=self.colors['primary'],
                        fg='white')
        title.pack(pady=20)
        
    def create_control_panel(self):
        """Erstellt Control Panel"""
        control_frame = ttk.LabelFrame(self.root, text="Steuerung", padding=15)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Datei-Auswahl
        file_frame = tk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Datei:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="Keine Datei geladen", 
                                   relief=tk.SUNKEN, width=50)
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(file_frame, text="📁 Datei wählen", 
                  command=self.load_file,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        # Varianten-Auswahl
        variant_frame = tk.Frame(control_frame)
        variant_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(variant_frame, text="Variante:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.variant_var = tk.IntVar(value=1)
        variants = [
            (1, "Klassisch (konservativ)"),
            (2, "Erweitert (mittel)"),
            (3, "Sensitiv (aggressiv)")
        ]
        
        for val, text in variants:
            ttk.Radiobutton(variant_frame, text=text, 
                           variable=self.variant_var, 
                           value=val).pack(side=tk.LEFT, padx=10)
        
        # Indikator-Auswahl
        indicator_frame = tk.Frame(control_frame)
        indicator_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(indicator_frame, text="Indikator:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.indicator_var = tk.StringVar(value='rsi')
        indicators = ['rsi', 'macd', 'stochastic']
        
        self.indicator_combo = ttk.Combobox(indicator_frame, 
                                           textvariable=self.indicator_var,
                                           values=indicators,
                                           state='readonly',
                                           width=15)
        self.indicator_combo.pack(side=tk.LEFT, padx=10)
        
        # Action Buttons
        action_frame = tk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="🔍 Analysieren", 
                  command=self.run_analysis,
                  style='Success.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="💾 Export JSON", 
                  command=lambda: self.export_results('json')).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="📊 Export CSV", 
                  command=lambda: self.export_results('csv')).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="🔄 Zurücksetzen", 
                  command=self.reset).pack(side=tk.LEFT, padx=5)
        
    def create_chart_area(self):
        """Erstellt Chart-Bereich"""
        # Notebook für Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Price Chart Tab
        self.price_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.price_frame, text="Preis-Chart")
        
        # Indicator Chart Tab
        self.indicator_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.indicator_frame, text="Indikator-Chart")
        
        # Combined Chart Tab
        self.combined_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.combined_frame, text="Kombiniert")
        
    def create_results_panel(self):
        """Erstellt Ergebnis-Panel"""
        results_frame = ttk.LabelFrame(self.root, text="Ergebnisse", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Statistik-Labels
        stats_frame = tk.Frame(results_frame)
        stats_frame.pack(fill=tk.X)
        
        self.stats_labels = {}
        stats = ['Total', 'Bullisch', 'Bärisch', 'Versteckt', 'Ø Stärke', 'Ø Konfidenz']
        
        for i, stat in enumerate(stats):
            frame = tk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, padx=20)
            
            tk.Label(frame, text=stat, font=('Arial', 9)).pack()
            label = tk.Label(frame, text="0", font=('Arial', 14, 'bold'))
            label.pack()
            self.stats_labels[stat] = label
        
        # Divergenz-Liste
        list_frame = tk.Frame(results_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        self.tree = ttk.Treeview(list_frame, 
                                columns=('Datum', 'Typ', 'Preis', 'Indikator', 'Stärke', 'Konfidenz'),
                                show='headings',
                                yscrollcommand=scrollbar.set,
                                height=8)
        
        # Spalten konfigurieren
        columns = {
            'Datum': 100,
            'Typ': 120,
            'Preis': 80,
            'Indikator': 80,
            'Stärke': 80,
            'Konfidenz': 80
        }
        
        for col, width in columns.items():
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree.yview)
        
        # Doppelklick-Event
        self.tree.bind('<Double-1>', self.on_divergence_select)
        
    def create_status_bar(self):
        """Erstellt Status-Bar"""
        self.status_frame = tk.Frame(self.root, bg=self.colors['dark'], height=30)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(self.status_frame, 
                                    text="Bereit",
                                    bg=self.colors['dark'],
                                    fg='white',
                                    anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.status_frame, 
                                       length=200,
                                       mode='indeterminate')
        
    def load_file(self):
        """Lädt Datei"""
        filename = filedialog.askopenfilename(
            title="Datei wählen",
            filetypes=[
                ("Unterstützte Dateien", "*.csv;*.parquet"),
                ("CSV Dateien", "*.csv"),
                ("Parquet Dateien", "*.parquet"),
                ("Alle Dateien", "*.*")
            ]
        )
        
        if filename:
            self.current_file = filename
            success = self.engine.load_data(filename)
            
            if success:
                self.file_label.config(text=filename.split('/')[-1])
                self.update_status(f"✅ Datei geladen: {len(self.engine.data)} Zeilen")
                self.plot_price_chart()
            else:
                messagebox.showerror("Fehler", "Datei konnte nicht geladen werden!")
                self.update_status("❌ Fehler beim Laden der Datei")
    
    def run_analysis(self):
        """Führt Analyse durch"""
        if self.engine.data is None:
            messagebox.showwarning("Warnung", "Bitte erst eine Datei laden!")
            return
        
        # Progress starten
        self.progress.pack(side=tk.RIGHT, padx=10)
        self.progress.start()
        self.update_status("🔍 Analyse läuft...")
        
        # Analyse in Thread
        def analyze():
            try:
                self.engine.set_variant(self.variant_var.get())
                divergences = self.engine.analyze(indicator=self.indicator_var.get())
                
                # GUI im Hauptthread aktualisieren
                self.root.after(0, self.display_results, divergences)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Fehler", str(e)))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, self.progress.pack_forget)
        
        thread = threading.Thread(target=analyze)
        thread.start()
    
    def display_results(self, divergences):
        """Zeigt Ergebnisse an"""
        # Liste leeren
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Divergenzen einfügen
        for div in divergences:
            self.tree.insert('', 'end', values=(
                div.timestamp.strftime('%Y-%m-%d'),
                div.type.value,
                f"{div.price:.2f}",
                f"{div.indicator_value:.2f}",
                f"{div.strength:.3f}",
                f"{div.confidence:.1%}"
            ))
        
        # Statistiken aktualisieren
        summary = self.engine.get_summary()
        self.stats_labels['Total'].config(text=str(summary.get('total', 0)))
        self.stats_labels['Bullisch'].config(text=str(summary.get('bullish', 0)))
        self.stats_labels['Bärisch'].config(text=str(summary.get('bearish', 0)))
        self.stats_labels['Versteckt'].config(text=str(summary.get('hidden_bullish', 0) + 
                                                      summary.get('hidden_bearish', 0)))
        self.stats_labels['Ø Stärke'].config(text=f"{summary.get('avg_strength', 0):.3f}")
        self.stats_labels['Ø Konfidenz'].config(text=f"{summary.get('avg_confidence', 0):.1%}")
        
        # Charts aktualisieren
        self.plot_results()
        
        self.update_status(f"✅ Analyse abgeschlossen: {len(divergences)} Divergenzen gefunden")
    
    def plot_price_chart(self):
        """Plottet Preis-Chart"""
        if self.engine.data is None:
            return
        
        # Clear previous plots
        for widget in self.price_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Plot candlesticks wenn möglich
        if all(col in self.engine.data.columns for col in ['open', 'high', 'low', 'close']):
            # Vereinfachter Candlestick Plot
            dates = self.engine.data.index
            ax.plot(dates, self.engine.data['close'], label='Close', color='blue', linewidth=1.5)
            
            # Add volume wenn vorhanden
            if 'volume' in self.engine.data.columns:
                ax2 = ax.twinx()
                ax2.bar(dates, self.engine.data['volume'], alpha=0.3, color='gray')
                ax2.set_ylabel('Volume')
        else:
            ax.plot(self.engine.data.index, self.engine.data['close'], 
                   label='Close', color='blue', linewidth=1.5)
        
        ax.set_xlabel('Datum')
        ax.set_ylabel('Preis')
        ax.set_title('Preis-Chart')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, self.price_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_results(self):
        """Plottet Ergebnisse"""
        if not self.engine.divergences:
            return
        
        # Clear previous plots
        for widget in self.combined_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(12, 8))
        
        # Preis mit Divergenz-Markern
        ax1 = fig.add_subplot(211)
        ax1.plot(self.engine.data.index, self.engine.data['close'], 
                color='blue', linewidth=1, label='Preis')
        
        # Divergenz-Marker hinzufügen
        for div in self.engine.divergences:
            color = 'green' if 'bullish' in div.type.value else 'red'
            marker = '^' if 'bullish' in div.type.value else 'v'
            ax1.scatter(div.timestamp, div.price, 
                       color=color, marker=marker, s=100, zorder=5)
        
        ax1.set_ylabel('Preis')
        ax1.set_title('Preis mit Divergenz-Signalen')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Indikator
        ax2 = fig.add_subplot(212)
        if 'indicator' in self.engine.data.columns:
            ax2.plot(self.engine.data.index, self.engine.data['indicator'], 
                    color='orange', linewidth=1, label=self.indicator_var.get().upper())
            
            # Divergenz-Marker
            for div in self.engine.divergences:
                color = 'green' if 'bullish' in div.type.value else 'red'
                marker = '^' if 'bullish' in div.type.value else 'v'
                ax2.scatter(div.timestamp, div.indicator_value, 
                           color=color, marker=marker, s=100, zorder=5)
        
        ax2.set_xlabel('Datum')
        ax2.set_ylabel('Indikator')
        ax2.set_title(f'{self.indicator_var.get().upper()} mit Divergenz-Signalen')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.combined_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_divergence_select(self, event):
        """Handler für Divergenz-Auswahl"""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            values = item['values']
            
            # Details-Fenster
            detail_window = tk.Toplevel(self.root)
            detail_window.title("Divergenz Details")
            detail_window.geometry("500x400")
            
            text = tk.Text(detail_window, wrap=tk.WORD, padx=10, pady=10)
            text.pack(fill=tk.BOTH, expand=True)
            
            # Details formatieren
            details = f"""
DIVERGENZ DETAILS
================

Datum: {values[0]}
Typ: {values[1]}
Preis: {values[2]}
Indikator: {values[3]}
Stärke: {values[4]}
Konfidenz: {values[5]}

Erklärung:
----------
"""
            # Finde passende Divergenz
            for div in self.engine.divergences:
                if div.timestamp.strftime('%Y-%m-%d') == values[0]:
                    details += f"{div.reason}\n\n"
                    details += "Zusätzliche Informationen:\n"
                    for key, value in div.additional_info.items():
                        details += f"  {key}: {value}\n"
                    break
            
            text.insert('1.0', details)
            text.config(state=tk.DISABLED)
    
    def export_results(self, format):
        """Exportiert Ergebnisse"""
        if not self.engine.divergences:
            messagebox.showwarning("Warnung", "Keine Ergebnisse zum Exportieren!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=f".{format}",
            filetypes=[
                (f"{format.upper()} Dateien", f"*.{format}"),
                ("Alle Dateien", "*.*")
            ]
        )
        
        if filename:
            exported = self.engine.export_results(format=format, filename=filename)
            if exported:
                messagebox.showinfo("Export", f"Erfolgreich exportiert nach:\n{exported}")
                self.update_status(f"✅ Exportiert: {filename}")
    
    def reset(self):
        """Setzt alles zurück"""
        self.engine = AnalysisEngine()
        self.current_file = None
        self.file_label.config(text="Keine Datei geladen")
        
        # Clear results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Reset stats
        for label in self.stats_labels.values():
            label.config(text="0")
        
        # Clear charts
        for frame in [self.price_frame, self.indicator_frame, self.combined_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        self.update_status("Zurückgesetzt")
    
    def update_status(self, message):
        """Aktualisiert Status-Bar"""
        self.status_label.config(text=message)


def main():
    """Hauptfunktion"""
    root = tk.Tk()
    app = DivergenceAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
