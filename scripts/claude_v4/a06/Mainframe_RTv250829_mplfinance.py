import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
import csv
import glob
import subprocess
import sys
from matplotlib.patches import Circle
from matplotlib.widgets import Button
try:
    import openpyxl
except ImportError:
    openpyxl = None
    
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min

def get_input_file():
    try:
        ps_command = '''
Add-Type -AssemblyName System.Windows.Forms
$openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
$openFileDialog.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
$openFileDialog.Title = "Select CSV input file"
$openFileDialog.InitialDirectory = "C:\\Projekte\\crt_250816\\data\\raw"
$result = $openFileDialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $openFileDialog.FileName
}
'''
        result = subprocess.run(["powershell", "-Command", ps_command], capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            return get_input_file_console()
    except Exception:
        return get_input_file_console()

def get_input_file_console():
    default_file = r"C:\Projekte\crt_250816\data\raw\btc_1day_candlesticks_all.csv"
    print(f"\nDefault file: {default_file}")
    print("Press Enter to use default, or choose from available files:")
    
    csv_files = []
    search_paths = ["*.csv", "../*/*.csv", "../../data/raw/*.csv", "../../data/*.csv"]
    for pattern in search_paths:
        csv_files.extend(glob.glob(pattern))
    
    for i, file in enumerate(csv_files, 1):
        print(f"{i}: {file}")
    print(f"{len(csv_files) + 1}: Enter custom path")
    print(f"0 or Enter: Use default ({default_file})")
    
    while True:
        try:
            choice = input(f"\nSelect file (0-{len(csv_files) + 1}): ").strip()
            if choice == "" or choice == "0":
                return default_file
            choice_num = int(choice)
            if 1 <= choice_num <= len(csv_files):
                return csv_files[choice_num - 1]
            elif choice_num == len(csv_files) + 1:
                custom_path = input("Enter full path to CSV file (or Enter for default): ")
                return custom_path if custom_path.strip() else default_file
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, EOFError):
            return default_file

def get_analysis_parameters():
    import sys
    if len(sys.argv) > 2:
        try:
            candle_percent = float(sys.argv[2])
            macd_percent = float(sys.argv[3]) if len(sys.argv) > 3 else 3.25
            return candle_percent, macd_percent, None
        except (ValueError, IndexError):
            pass
    
    try:
        print("\nEnter analysis parameters (or press Enter for defaults):")
        candle_input = input("Enter Candle tolerance % (default: 0.1): ").strip()
        candle_percent = 0.1 if candle_input == "" else float(candle_input)
        macd_input = input("Enter MACD tolerance % (default: 3.25): ").strip()
        macd_percent = 3.25 if macd_input == "" else float(macd_input)
        
        print("\nOptional second variant for comparison:")
        candle_input2 = input("Enter second Candle tolerance % (leave empty for none): ").strip()
        macd_input2 = input("Enter second MACD tolerance % (leave empty for none): ").strip()
        
        variant2 = None
        if candle_input2 or macd_input2:
            try:
                candle_percent2 = float(candle_input2) if candle_input2 else candle_percent
                macd_percent2 = float(macd_input2) if macd_input2 else macd_percent
                variant2 = (candle_percent2, macd_percent2)
            except ValueError as e:
                print(f"Invalid input for second variant: {e}")
                variant2 = None
        
        return candle_percent, macd_percent, variant2
    except EOFError:
        return 0.1, 3.25, None

def get_analysis_type():
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower().strip()
        if choice in ['a', 'b', 'c', 'd', 'e', 'f']:
            return choice
    
    try:
        print("\nSelect analysis type:")
        print("a: CBullDivg_Analysis (Classic Bullish Divergence)")
        print("b: BullDivg_x2_analysis (Extended Bullish Divergence)")
        print("c: HBearDivg_analysis (Hidden Bearish Divergence)")
        print("d: HBullDivg_analysis (Hidden Bullish Divergence)")
        print("e: All analyses (a-d)")
        print("f: DOE (Design of Experiments)")
        
        while True:
            choice = input("\nEnter your choice (a-f): ").lower().strip()
            if choice in ['a', 'b', 'c', 'd', 'e', 'f']:
                return choice
            print("Invalid choice. Please enter a, b, c, d, e, or f.")
    except EOFError:
        return 'e'

def run_analysis(df, analysis_type, window=5, candle_tol=0.1, macd_tol=3.25):
    results = {}
    if analysis_type == 'a' or analysis_type == 'e' or analysis_type == 'f':
        CBullDivg_analysis(df, window, candle_tol, macd_tol)
        results['CBullDivg'] = True
    if analysis_type == 'b' or analysis_type == 'e' or analysis_type == 'f':
        CBullDivg_x2_analysis(df, window, candle_tol, macd_tol)
        results['CBullDivg_x2'] = True
    if analysis_type == 'c' or analysis_type == 'e' or analysis_type == 'f':
        HBearDivg_analysis(df, window, candle_tol, macd_tol)
        results['HBearDivg'] = True
    if analysis_type == 'd' or analysis_type == 'e' or analysis_type == 'f':
        HBullDivg_analysis(df, window, candle_tol, macd_tol)
        results['HBullDivg'] = True
    return results

def plot_markers_on_subplot(ax, df, analysis_results, variant_name, subplot_type, y_column):
    """Plot markers on a specific subplot with proper legend grouping and new marker styles"""
    from matplotlib.patches import Circle
    import matplotlib.patches as patches
    
    legend_entries = set()
    marker_positions = []
    
    # Define marker styles: bullish=green, bearish=red, CBullDivg=cross, CBullDivg_x2=triangle, HBullDivg=square, HBearDivg=rhombic
    marker_styles = {
        'CBullDivg': '+',        # cross
        'CBullDivg_x2': '^',     # triangle  
        'HBullDivg': 's',        # square
        'HBearDivg': 'D'         # rhombic (diamond)
    }
    
    for i in range(2, len(df)):
        # CBullDivg Classic (Bullish = Green)
        if 'CBullDivg' in analysis_results and "CBullD_gen" in df.columns and df["CBullD_gen"].iloc[i] == 1:
            y_val = df[y_column].iloc[i] if y_column in df.columns else df["CBullD_Lower_Low_gen"].iloc[i]
            date1 = pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i])
            date2 = pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i])
            
            label_key = f'{variant_name} CBullDivg Classic'
            if label_key not in legend_entries:
                ax.scatter(date1, y_val, marker='+', s=150, c='green', label=label_key, linewidths=2)
                legend_entries.add(label_key)
            else:
                ax.scatter(date1, y_val, marker='+', s=150, c='green', linewidths=2)
            ax.scatter(date2, y_val, marker='+', s=150, c='green', linewidths=2)
            marker_positions.extend([(date1, y_val), (date2, y_val)])
            
        # CBullDivg Hidden (Bullish = Green)
        if 'CBullDivg' in analysis_results and "CBullD_neg_MACD" in df.columns and df["CBullD_neg_MACD"].iloc[i] == 1:
            y_val = df[y_column].iloc[i] if y_column in df.columns else df["CBullD_Lower_Low_neg_MACD"].iloc[i]
            date1 = pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i])
            date2 = pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i])
            
            label_key = f'{variant_name} CBullDivg Hidden'
            if label_key not in legend_entries:
                ax.scatter(date1, y_val, marker='+', s=120, c='green', alpha=0.7, label=label_key, linewidths=2)
                legend_entries.add(label_key)
            else:
                ax.scatter(date1, y_val, marker='+', s=120, c='green', alpha=0.7, linewidths=2)
            ax.scatter(date2, y_val, marker='+', s=120, c='green', alpha=0.7, linewidths=2)
            marker_positions.extend([(date1, y_val), (date2, y_val)])
                      
        # CBullDivg_x2 Extended (Bullish = Green)
        if 'CBullDivg_x2' in analysis_results and "CBullD_x2_gen" in df.columns and df["CBullD_x2_gen"].iloc[i] == 1:
            y_val = df[y_column].iloc[i] if y_column in df.columns else df["CBullD_x2_Lower_Low_gen"].iloc[i]
            date1 = pd.to_datetime(df["CBullD_x2_Lower_Low_date_gen"][i])
            date2 = pd.to_datetime(df["CBullD_x2_Higher_Low_date_gen"][i])
            
            label_key = f'{variant_name} CBullDivg_x2 Classic'
            if label_key not in legend_entries:
                ax.scatter(date1, y_val, marker='^', s=120, c='green', label=label_key)
                legend_entries.add(label_key)
            else:
                ax.scatter(date1, y_val, marker='^', s=120, c='green')
            ax.scatter(date2, y_val, marker='^', s=120, c='green')
            marker_positions.extend([(date1, y_val), (date2, y_val)])
                      
        # HBearDivg (Bearish = Red)
        if 'HBearDivg' in analysis_results and "HBearD_gen" in df.columns and df["HBearD_gen"].iloc[i] == 1:
            y_val = df[y_column].iloc[i] if y_column in df.columns else df["HBearD_Higher_High_gen"].iloc[i]
            date1 = pd.to_datetime(df["HBearD_Higher_High_date_gen"][i])
            date2 = pd.to_datetime(df["HBearD_Lower_High_date_gen"][i])
            
            label_key = f'{variant_name} HBearDivg Classic'
            if label_key not in legend_entries:
                ax.scatter(date1, y_val, marker='D', s=120, c='red', label=label_key)
                legend_entries.add(label_key)
            else:
                ax.scatter(date1, y_val, marker='D', s=120, c='red')
            ax.scatter(date2, y_val, marker='D', s=120, c='red')
            marker_positions.extend([(date1, y_val), (date2, y_val)])
                      
        # HBullDivg (Bullish = Green)
        if 'HBullDivg' in analysis_results and "HBullD_gen" in df.columns and df["HBullD_gen"].iloc[i] == 1:
            y_val = df[y_column].iloc[i] if y_column in df.columns else df["HBullD_Lower_Low_gen"].iloc[i]
            date1 = pd.to_datetime(df["HBullD_Lower_Low_date_gen"][i])
            date2 = pd.to_datetime(df["HBullD_Higher_Low_date_gen"][i])
            
            label_key = f'{variant_name} HBullDivg Classic'
            if label_key not in legend_entries:
                ax.scatter(date1, y_val, marker='s', s=120, c='green', label=label_key)
                legend_entries.add(label_key)
            else:
                ax.scatter(date1, y_val, marker='s', s=120, c='green')
            ax.scatter(date2, y_val, marker='s', s=120, c='green')
            marker_positions.extend([(date1, y_val), (date2, y_val)])
            
        # HBullDivg Hidden (Bullish = Green)  
        if 'HBullDivg' in analysis_results and "HBullD_neg_MACD" in df.columns and df["HBullD_neg_MACD"].iloc[i] == 1:
            y_val = df[y_column].iloc[i] if y_column in df.columns else df["HBullD_Lower_Low_neg_MACD"].iloc[i]
            date1 = pd.to_datetime(df["HBullD_Lower_Low_date_neg_MACD"][i])
            date2 = pd.to_datetime(df["HBullD_Higher_Low_date_neg_MACD"][i])
            
            label_key = f'{variant_name} HBullDivg Hidden'
            if label_key not in legend_entries:
                ax.scatter(date1, y_val, marker='s', s=100, c='green', alpha=0.7, label=label_key)
                legend_entries.add(label_key)
            else:
                ax.scatter(date1, y_val, marker='s', s=100, c='green', alpha=0.7)
            ax.scatter(date2, y_val, marker='s', s=100, c='green', alpha=0.7)
            marker_positions.extend([(date1, y_val), (date2, y_val)])
    
    return marker_positions

def export_markers_to_csv(df, filename, analysis_results, candle_percent, macd_percent):
    markers = []
    counts = {}
    
    for analysis in ['CBullDivg', 'CBullDivg_x2', 'HBearDivg', 'HBullDivg']:
        if analysis in analysis_results:
            counts[analysis] = {'classic': 0, 'hidden': 0, 'total': 0}
    
    for i in range(len(df)):
        if 'CBullDivg' in analysis_results:
            if "CBullD_gen" in df.columns and i < len(df) and df["CBullD_gen"].iloc[i] == 1:
                markers.append({'Type': 'CBullDivg_Classic', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['CBullDivg']['classic'] += 1
                counts['CBullDivg']['total'] += 1
            if "CBullD_neg_MACD" in df.columns and i < len(df) and df["CBullD_neg_MACD"].iloc[i] == 1:
                markers.append({'Type': 'CBullDivg_Hidden', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['CBullDivg']['hidden'] += 1
                counts['CBullDivg']['total'] += 1
        
        if 'CBullDivg_x2' in analysis_results:
            if "CBullD_x2_gen" in df.columns and i < len(df) and df["CBullD_x2_gen"].iloc[i] == 1:
                markers.append({'Type': 'CBullDivg_x2_Classic', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['CBullDivg_x2']['classic'] += 1
                counts['CBullDivg_x2']['total'] += 1
        
        if 'HBearDivg' in analysis_results:
            if "HBearD_gen" in df.columns and i < len(df) and df["HBearD_gen"].iloc[i] == 1:
                markers.append({'Type': 'HBearDivg_Classic', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['HBearDivg']['classic'] += 1
                counts['HBearDivg']['total'] += 1
        
        if 'HBullDivg' in analysis_results:
            if "HBullD_gen" in df.columns and i < len(df) and df["HBullD_gen"].iloc[i] == 1:
                markers.append({'Type': 'HBullDivg_Classic', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['HBullDivg']['classic'] += 1
                counts['HBullDivg']['total'] += 1
            if "HBullD_neg_MACD" in df.columns and i < len(df) and df["HBullD_neg_MACD"].iloc[i] == 1:
                markers.append({'Type': 'HBullDivg_Hidden', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['HBullDivg']['hidden'] += 1
                counts['HBullDivg']['total'] += 1

    print("\n=== ANALYSIS SUMMARY ===")
    for analysis_name, count_data in counts.items():
        print(f"{analysis_name}: Classic={count_data['classic']}, Hidden={count_data['hidden']}, Total={count_data['total']}")
    print(f"\nGrand Total markers found: {len(markers)}")
    
    if markers:
        os.makedirs('results', exist_ok=True)
        results_filename = os.path.join('results', filename)
        
        df_markers = pd.DataFrame(markers)
        df_markers.to_csv(results_filename, index=False)
        print(f"\nMarkers exported to {results_filename}")
        
        if openpyxl:
            xlsx_filename = results_filename.replace('.csv', '.xlsx')
            df_xlsx = df_markers.copy()
            df_xlsx['Date'] = pd.to_datetime(df_xlsx['Date']).dt.tz_localize(None)
            
            with pd.ExcelWriter(xlsx_filename, engine='openpyxl') as writer:
                df_xlsx_sorted = df_xlsx.sort_values(['Type', 'Date'])
                df_xlsx_sorted.to_excel(writer, sheet_name='All_Markers', index=False)
                for marker_type in df_xlsx['Type'].unique():
                    subset = df_xlsx[df_xlsx['Type'] == marker_type].sort_values('Date')
                    subset.to_excel(writer, sheet_name=marker_type, index=False)
            print(f"Enhanced XLSX exported to {xlsx_filename}")
    
    return counts

# Main execution
csv_file_path = get_input_file()
if not csv_file_path:
    print("No file selected. Exiting.")
    exit()

analysis_type = get_analysis_type()

if analysis_type != 'f':
    candle_percent, macd_percent, variant2 = get_analysis_parameters()

print(f"Loading data from: {csv_file_path}")
df = pd.read_csv(csv_file_path, low_memory=False)

print("Initializing indicators...")
Initialize_RSI_EMA_MACD(df)
Local_Max_Min(df)

window = 5

if analysis_type != 'f':
    print(f"Running analysis with parameters: window={window}, candle_tol={candle_percent}, macd_tol={macd_percent}")
    analysis_results = run_analysis(df, analysis_type, window, candle_percent, macd_percent)

    # Prepare data for mplfinance
    df['Date'] = pd.to_datetime(df['date'])
    df.set_index('Date', inplace=True)
    ohlc_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Create figure with 3 subplots - WHITE BACKGROUND, GREEN RISING CANDLES
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), facecolor='#242320', sharex=True)
    fig.suptitle('Technical Analysis with Divergence Markers', color='white', fontsize=16)
    
    # Style all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#242320')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
    
    # Plot 1: CANDLESTICKS with GREEN RISING BODIES
    for i in range(len(ohlc_df)):
        open_price = ohlc_df['open'].iloc[i]
        close_price = ohlc_df['close'].iloc[i]
        high_price = ohlc_df['high'].iloc[i]
        low_price = ohlc_df['low'].iloc[i]
        date = ohlc_df.index[i]
        
        # Green for rising, red for falling candlesticks
        color = '#44ff44' if close_price > open_price else '#ff4444'
        ax1.plot([date, date], [low_price, high_price], color=color, linewidth=1)  # Wick
        
        # Body with GREEN for rising candles
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        rect = plt.Rectangle((mdates.date2num(date) - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor=color, alpha=0.8)
        ax1.add_patch(rect)
    
    ax1.set_title('Price Chart', color='white')
    ax1.set_ylabel('Price', color='white')
    
    # Plot 2: RSI with orange line
    ax2.plot(df.index, df['RSI'], color='orange', linewidth=2, label='RSI')
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI', color='white')
    ax2.set_ylabel('RSI', color='white')
    
    # Plot 3: MACD histogram
    colors = ['green' if x > 0 else 'red' for x in df['macd_histogram']]
    ax3.bar(df.index, df['macd_histogram'], color=colors, alpha=0.7, width=1)
    ax3.set_title('MACD Histogram', color='white')
    ax3.set_ylabel('MACD', color='white')
    
    # Plot EMAs on price chart
    ax1.plot(df.index, df['EMA_20'], color='yellow', alpha=0.7, linewidth=1, label='EMA 20')
    ax1.plot(df.index, df['EMA_50'], color='cyan', alpha=0.7, linewidth=1, label='EMA 50')
    ax1.plot(df.index, df['EMA_100'], color='magenta', alpha=0.7, linewidth=1, label='EMA 100')
    ax1.plot(df.index, df['EMA_200'], color='orange', alpha=0.7, linewidth=1, label='EMA 200')
    
    # Add markers to all three subplots with VARIANT GROUPING
    main_positions_ax1 = plot_markers_on_subplot(ax1, df, analysis_results, "V1", "price", "close")
    main_positions_ax2 = plot_markers_on_subplot(ax2, df, analysis_results, "V1", "rsi", "RSI")  
    main_positions_ax3 = plot_markers_on_subplot(ax3, df, analysis_results, "V1", "macd", "macd_histogram")
    
    # Export results
    output_filename = f"markers_output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    counts = export_markers_to_csv(df, output_filename, analysis_results, candle_percent, macd_percent)
    
    # CONSOLIDATED MASTER XLSX - ALL RESULTS IN ONE FILE
    os.makedirs('results', exist_ok=True)
    master_xlsx = os.path.join('results', f"MASTER_ALL_RESULTS_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    all_data = []
    
    for analysis in counts:
        all_data.append({
            'Run_Type': 'Main',
            'Candle_Percent': candle_percent,
            'MACD_Percent': macd_percent,
            'Analysis_Type': analysis,
            'Classic_Count': counts[analysis]['classic'],
            'Hidden_Count': counts[analysis]['hidden'],
            'Total_Count': counts[analysis]['total']
        })
    
    # Second variant if specified
    if variant2:
        print(f"\nRunning second variant: candle={variant2[0]}, macd={variant2[1]}")
        df_var2 = df.copy()
        variant2_results = run_analysis(df_var2, analysis_type, window, variant2[0], variant2[1])
        
        # Plot variant 2 markers with V2 grouping
        variant2_positions_ax1 = plot_markers_on_subplot(ax1, df_var2, variant2_results, "V2", "price", "close")
        variant2_positions_ax2 = plot_markers_on_subplot(ax2, df_var2, variant2_results, "V2", "rsi", "RSI")
        variant2_positions_ax3 = plot_markers_on_subplot(ax3, df_var2, variant2_results, "V2", "macd", "macd_histogram")
        
        variant2_counts = export_markers_to_csv(df_var2, f"variant2_{variant2[0]}_{variant2[1]}.csv", variant2_results, variant2[0], variant2[1])
        
        # Add variant2 to master data
        for analysis in variant2_counts:
            all_data.append({
                'Run_Type': 'Variant2',
                'Candle_Percent': variant2[0],
                'MACD_Percent': variant2[1], 
                'Analysis_Type': analysis,
                'Classic_Count': variant2_counts[analysis]['classic'],
                'Hidden_Count': variant2_counts[analysis]['hidden'],
                'Total_Count': variant2_counts[analysis]['total']
            })
        
        # TRANSPARENT YELLOW/BLUE CIRCLES for additional/missing markers
        import matplotlib.dates as mdates
        
        # Convert positions to sets for comparison
        main_positions_set = set(main_positions_ax1)
        variant2_positions_set = set(variant2_positions_ax1)
        
        additional = variant2_positions_set - main_positions_set
        missing = main_positions_set - variant2_positions_set
        
        # YELLOW circles for additional markers (OUTLINE ONLY, TRANSPARENT FILL)
        for date, price in additional:
            circle = Circle((mdates.date2num(date), price), radius=price*0.015, 
                          facecolor='none', edgecolor='yellow', linewidth=3, alpha=0.8)
            ax1.add_patch(circle)
        
        # BLUE circles for missing markers (OUTLINE ONLY, TRANSPARENT FILL)  
        for date, price in missing:
            circle = Circle((mdates.date2num(date), price), radius=price*0.015,
                          facecolor='none', edgecolor='blue', linewidth=3, alpha=0.8)  
            ax1.add_patch(circle)
        
        print(f"Added {len(additional)} yellow circles, {len(missing)} blue circles")
    
    # MASTER XLSX WITH ALL DATA LOGICALLY GROUPED
    if openpyxl and all_data:
        with pd.ExcelWriter(master_xlsx, engine='openpyxl') as writer:
            df_master = pd.DataFrame(all_data)
            
            # Sheet 1: All results sorted by parameters
            df_sorted = df_master.sort_values(['Run_Type', 'Candle_Percent', 'MACD_Percent', 'Analysis_Type'])
            df_sorted.to_excel(writer, sheet_name='All_Results_Sorted', index=False)
            
            # Sheet 2: Summary by parameters
            summary = df_master.groupby(['Run_Type', 'Candle_Percent', 'MACD_Percent']).agg({
                'Classic_Count': 'sum', 'Hidden_Count': 'sum', 'Total_Count': 'sum'
            }).reset_index()
            summary.to_excel(writer, sheet_name='Parameter_Summary', index=False)
            
            # Sheet 3: By analysis type
            for analysis_type in df_master['Analysis_Type'].unique():
                subset = df_master[df_master['Analysis_Type'] == analysis_type]
                subset.to_excel(writer, sheet_name=f'{analysis_type}_Results', index=False)
        
        print(f"MASTER consolidated results: {master_xlsx}")
    
    # MARKER COUNT TEXT BOX - UPPER LEFT (XLSX FORMAT)
    count_lines = []
    for name, data in counts.items():
        count_lines.append(f"{name}:")
        count_lines.append(f"  Classic: {data['classic']}")
        count_lines.append(f"  Hidden: {data['hidden']}")
        count_lines.append(f"  Total: {data['total']}")
    count_lines.append(f"Grand Total: {sum(data['total'] for data in counts.values())}")
    count_text = "\\n".join(count_lines)
    
    ax1.text(0.02, 0.98, count_text, transform=ax1.transAxes, color='white', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8),
             verticalalignment='top', fontsize=9)
    
    # LEGEND IN UPPER RIGHT CORNER (GROUPED BY VARIANT) - TOGGABLE
    legend = ax1.legend(loc='upper right', facecolor='black', edgecolor='white', 
                       labelcolor='white', fontsize=10)
    legend.set_picker(True)
    
    # Make legend entries toggable
    def on_legend_pick(event):
        legend_item = event.artist
        label = legend_item.get_label()
        
        # Find all artists with this label and toggle visibility
        for ax in [ax1, ax2, ax3]:
            for artist in ax.get_children():
                if hasattr(artist, 'get_label') and artist.get_label() == label:
                    artist.set_visible(not artist.get_visible())
        
        # Update legend
        legend_item.set_alpha(0.5 if not legend_item.get_visible() else 1.0)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('pick_event', on_legend_pick)
    
    # WORKING RESET/REFRESH BUTTONS
    def reset_view(event):
        ax1.set_xlim(df.index.min(), df.index.max())
        ax2.set_xlim(df.index.min(), df.index.max())
        ax3.set_xlim(df.index.min(), df.index.max())
        ax1.set_ylim(df['low'].min() * 0.98, df['high'].max() * 1.02)
        ax2.set_ylim(0, 100)
        ax3.set_ylim(df['macd_histogram'].min() * 1.1, df['macd_histogram'].max() * 1.1)
        fig.canvas.draw()
    
    def refresh_plot(event):
        fig.canvas.draw()
    
    # Add reset/refresh buttons
    ax_reset = plt.axes([0.85, 0.95, 0.08, 0.04])
    ax_refresh = plt.axes([0.75, 0.95, 0.08, 0.04])
    btn_reset = Button(ax_reset, 'Reset', color='red', hovercolor='darkred')
    btn_refresh = Button(ax_refresh, 'Refresh', color='green', hovercolor='darkgreen')
    btn_reset.on_clicked(reset_view)
    btn_refresh.on_clicked(refresh_plot)
    
    plt.tight_layout()
    plt.show()

else:
    # DOE functionality - CSV INPUT ONLY
    print("Running DOE analysis...")
    try:
        doe_file = input("Enter DOE parameter file (default: doe_parameters_example.csv): ").strip()
        if not doe_file:
            doe_file = "doe_parameters_example.csv"
    except EOFError:
        doe_file = "doe_parameters_example.csv"
    
    if os.path.exists(doe_file):
        doe_params = pd.read_csv(doe_file)
        print(f"Loaded {len(doe_params)} parameter combinations")
        
        all_results = []
        for idx, row in doe_params.iterrows():
            candle_percent = row['candle_percent']
            macd_percent = row['macd_percent']
            print(f"\nDOE run {idx+1}/{len(doe_params)}: candle={candle_percent}, macd={macd_percent}")
            
            df_copy = df.copy()
            analysis_results = run_analysis(df_copy, 'e', window, candle_percent, macd_percent)
            counts = export_markers_to_csv(df_copy, f"doe_{candle_percent}_{macd_percent}.csv", analysis_results, candle_percent, macd_percent)
            
            for analysis_name, count_data in counts.items():
                all_results.append({
                    'candle_percent': candle_percent,
                    'macd_percent': macd_percent,
                    'analysis_type': analysis_name,
                    'classic_count': count_data['classic'],
                    'hidden_count': count_data['hidden'],
                    'total_count': count_data['total']
                })
        
        # SINGLE DOE RESULTS XLSX WITH LOGICAL GROUPING
        os.makedirs('results', exist_ok=True)
        doe_summary_file = os.path.join('results', f"DOE_COMPLETE_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        
        if openpyxl:
            with pd.ExcelWriter(doe_summary_file, engine='openpyxl') as writer:
                doe_df = pd.DataFrame(all_results)
                # All results sorted
                doe_sorted = doe_df.sort_values(['candle_percent', 'macd_percent', 'analysis_type'])
                doe_sorted.to_excel(writer, sheet_name='All_DOE_Results', index=False)
                
                # Best parameters summary
                best_summary = doe_df.groupby(['candle_percent', 'macd_percent']).agg({
                    'classic_count': 'sum', 'hidden_count': 'sum', 'total_count': 'sum'
                }).reset_index().sort_values(['total_count'], ascending=False)
                best_summary.to_excel(writer, sheet_name='Best_Parameters', index=False)
                
                # By analysis type
                for analysis_type in doe_df['analysis_type'].unique():
                    subset = doe_df[doe_df['analysis_type'] == analysis_type].sort_values(['candle_percent', 'macd_percent'])
                    subset.to_excel(writer, sheet_name=f'{analysis_type}', index=False)
            
            print(f"\nDOE COMPLETE results: {doe_summary_file}")
        
        # Plot best DOE result with same features
        doe_df = pd.DataFrame(all_results)
        best_params = doe_df.groupby(['candle_percent', 'macd_percent'])['total_count'].sum().idxmax()
        print(f"\nPlotting best DOE parameters: candle={best_params[0]}, macd={best_params[1]}")
        
        # Create plot for best DOE parameters
        df_best = df.copy()
        best_results = run_analysis(df_best, 'e', window, best_params[0], best_params[1])
        
        # Create figure with 3 subplots - same as main plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), facecolor='#242320', sharex=True)
        fig.suptitle(f'DOE Best Result: Candle {best_params[0]}%, MACD {best_params[1]}%', color='white', fontsize=16)
        
        # Style all axes
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#242320')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.grid(True, alpha=0.2)
        
        # Plot candlesticks with green bodies
        for i in range(len(df_best)):
            date = df_best.index[i]
            open_price, high_price, low_price, close_price = df_best[['open', 'high', 'low', 'close']].iloc[i]
            
            ax1.plot([date, date], [low_price, high_price], color='white', linewidth=1)
            body_bottom = min(open_price, close_price)
            body_height = abs(close_price - open_price)
            color = '#44ff44' if close_price > open_price else '#ff4444'
            rect = plt.Rectangle((mdates.date2num(date) - 0.3, body_bottom), 0.6, body_height, 
                               facecolor=color, edgecolor=color, alpha=0.8)
            ax1.add_patch(rect)
        
        ax1.set_title('Price with Divergence Markers', color='white')
        ax1.set_ylabel('Price', color='white')
        
        # Plot indicators
        ax2.plot(df_best.index, df_best['RSI'], color='purple', label='RSI')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_title('RSI with Divergence Markers', color='white')
        ax2.set_ylabel('RSI', color='white')
        
        ax3.plot(df_best.index, df_best['macd_histogram'], color='cyan', label='MACD Histogram')
        ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax3.set_title('MACD Histogram with Divergence Markers', color='white')
        ax3.set_ylabel('MACD Histogram', color='white')
        ax3.set_xlabel('Date', color='white')
        
        # Add markers to all subplots
        plot_markers_on_subplot(ax1, df_best, best_results, "Best DOE", "price", "close")
        plot_markers_on_subplot(ax2, df_best, best_results, "Best DOE", "rsi", "RSI")
        plot_markers_on_subplot(ax3, df_best, best_results, "Best DOE", "macd", "macd_histogram")
        
        # Legend and info box
        best_counts = export_markers_to_csv(df_best, f"doe_best_{best_params[0]}_{best_params[1]}.csv", best_results, best_params[0], best_params[1])
        count_lines = []
        for name, data in best_counts.items():
            count_lines.append(f"{name}:")
            count_lines.append(f"  Classic: {data['classic']}")
            count_lines.append(f"  Hidden: {data['hidden']}")
            count_lines.append(f"  Total: {data['total']}")
        count_lines.append(f"Grand Total: {sum(data['total'] for data in best_counts.values())}")
        count_text = "\\n".join(count_lines)
        
        ax1.text(0.02, 0.98, count_text, transform=ax1.transAxes, color='white', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8),
                 verticalalignment='top', fontsize=9)
        
        legend = ax1.legend(loc='upper right', facecolor='black', edgecolor='white', 
                           labelcolor='white', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        print("DOE plotting completed")
    else:
        print(f"DOE parameter file not found: {doe_file}")