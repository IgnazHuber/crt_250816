import finplot as fplt
import pandas as pd
import os
import csv
import glob
import subprocess
import sys
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
        
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=60
        )
        
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
    search_paths = [
        "*.csv",
        "../*/*.csv", 
        "../../data/raw/*.csv",
        "../../data/*.csv"
    ]
    
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
        except ValueError:
            print("Please enter a valid number or press Enter for default.")

def get_analysis_parameters():
    print("\nEnter analysis parameters (or press Enter for defaults):")
    
    candle_input = input("Enter Candle tolerance % (default: 0.1): ").strip()
    candle_percent = 0.1 if candle_input == "" else float(candle_input)
    
    macd_input = input("Enter MACD tolerance % (default: 3.25): ").strip()
    macd_percent = 3.25 if macd_input == "" else float(macd_input)
    
    # Optional second variant
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
            print("Skipping second variant...")
            variant2 = None
    
    return candle_percent, macd_percent, variant2

def get_analysis_type():
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

def run_analysis(df, analysis_type, window=5, candle_tol=0.1, macd_tol=3.25):
    results = {}
    
    if analysis_type == 'a' or analysis_type == 'e' or analysis_type == 'f':
        print("Running CBullDivg_Analysis...")
        CBullDivg_analysis(df, window, candle_tol, macd_tol)
        results['CBullDivg'] = True
    
    if analysis_type == 'b' or analysis_type == 'e' or analysis_type == 'f':
        print("Running CBullDivg_x2_analysis...")
        CBullDivg_x2_analysis(df, window, candle_tol, macd_tol)
        results['CBullDivg_x2'] = True
    
    if analysis_type == 'c' or analysis_type == 'e' or analysis_type == 'f':
        print("Running HBearDivg_analysis...")
        HBearDivg_analysis(df, window, candle_tol, macd_tol)
        results['HBearDivg'] = True
    
    if analysis_type == 'd' or analysis_type == 'e' or analysis_type == 'f':
        print("Running HBullDivg_analysis...")
        HBullDivg_analysis(df, window, candle_tol, macd_tol)
        results['HBullDivg'] = True
    
    return results

def plot_markers(df, analysis_results):
    # X markers - outline only with + style
    for i in range(2, len(df)):
        if 'CBullDivg' in analysis_results:
            if "CBullD_gen" in df.columns and i < len(df) and df["CBullD_gen"].iloc[i] == 1:
                fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]), df["CBullD_Lower_Low_gen"][i], style="+", ax=ax1, color="red", width=3, legend="X Red")
                fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]), df["CBullD_Higher_Low_gen"][i], style="+", ax=ax1, color="blue", width=3, legend="X Blue")
                fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]), df["CBullD_Lower_Low_RSI_gen"][i], style="+", ax=ax2, color="red", width=3)
                fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]), df["CBullD_Higher_Low_RSI_gen"][i], style="+", ax=ax2, color="blue", width=3)
                fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]), df["CBullD_Lower_Low_MACD_gen"][i], style="+", ax=ax3, color="red", width=3)
                fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]), df["CBullD_Higher_Low_MACD_gen"][i], style="+", ax=ax3, color="blue", width=3)
            
            if "CBullD_neg_MACD" in df.columns and i < len(df) and df["CBullD_neg_MACD"].iloc[i] == 1:
                fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]), df["CBullD_Lower_Low_neg_MACD"][i], style="x", ax=ax1, color="red", width=2, legend="X Hidden Red")
                fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]), df["CBullD_Higher_Low_neg_MACD"][i], style="x", ax=ax1, color="blue", width=2, legend="X Hidden Blue")
                fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]), df["CBullD_Lower_Low_RSI_neg_MACD"][i], style="x", ax=ax2, color="red", width=2)
                fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]), df["CBullD_Higher_Low_RSI_neg_MACD"][i], style="x", ax=ax2, color="blue", width=2)
                fplt.plot(pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]), df["CBullD_Lower_Low_MACD_neg_MACD"][i], style="x", ax=ax3, color="red", width=2)
                fplt.plot(pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]), df["CBullD_Higher_Low_MACD_neg_MACD"][i], style="x", ax=ax3, color="blue", width=2)
        
        # Triangle markers - outline only 
        if 'CBullDivg_x2' in analysis_results:
            if "CBullD_x2_gen" in df.columns and i < len(df) and df["CBullD_x2_gen"].iloc[i] == 1:
                fplt.plot(pd.to_datetime(df["CBullD_x2_Lower_Low_date_gen"][i]), df["CBullD_x2_Lower_Low_gen"][i], style="^", ax=ax1, color="red", width=1, legend="△ Red")
                fplt.plot(pd.to_datetime(df["CBullD_x2_Higher_Low_date_gen"][i]), df["CBullD_x2_Higher_Low_gen"][i], style="^", ax=ax1, color="blue", width=1, legend="△ Blue")
                fplt.plot(pd.to_datetime(df["CBullD_x2_Lower_Low_date_gen"][i]), df["CBullD_x2_Lower_Low_RSI_gen"][i], style="^", ax=ax2, color="red", width=1)
                fplt.plot(pd.to_datetime(df["CBullD_x2_Higher_Low_date_gen"][i]), df["CBullD_x2_Higher_Low_RSI_gen"][i], style="^", ax=ax2, color="blue", width=1)
                fplt.plot(pd.to_datetime(df["CBullD_x2_Lower_Low_date_gen"][i]), df["CBullD_x2_Lower_Low_MACD_gen"][i], style="^", ax=ax3, color="red", width=1)
                fplt.plot(pd.to_datetime(df["CBullD_x2_Higher_Low_date_gen"][i]), df["CBullD_x2_Higher_Low_MACD_gen"][i], style="^", ax=ax3, color="blue", width=1)
        
        # Square markers - outline only
        if 'HBearDivg' in analysis_results:
            if "HBearD_gen" in df.columns and i < len(df) and df["HBearD_gen"].iloc[i] == 1:
                fplt.plot(pd.to_datetime(df["HBearD_Higher_High_date_gen"][i]), df["HBearD_Higher_High_gen"][i], style="s", ax=ax1, color="red", width=1, legend="□ Red")
                fplt.plot(pd.to_datetime(df["HBearD_Lower_High_date_gen"][i]), df["HBearD_Lower_High_gen"][i], style="s", ax=ax1, color="blue", width=1, legend="□ Blue")
                fplt.plot(pd.to_datetime(df["HBearD_Higher_High_date_gen"][i]), df["HBearD_Higher_High_RSI_gen"][i], style="s", ax=ax2, color="red", width=1)
                fplt.plot(pd.to_datetime(df["HBearD_Lower_High_date_gen"][i]), df["HBearD_Lower_High_RSI_gen"][i], style="s", ax=ax2, color="blue", width=1)
                fplt.plot(pd.to_datetime(df["HBearD_Higher_High_date_gen"][i]), df["HBearD_Higher_High_MACD_gen"][i], style="s", ax=ax3, color="red", width=1)
                fplt.plot(pd.to_datetime(df["HBearD_Lower_High_date_gen"][i]), df["HBearD_Lower_High_MACD_gen"][i], style="s", ax=ax3, color="blue", width=1)
        
        # Circle markers - outline only
        if 'HBullDivg' in analysis_results:
            if "HBullD_gen" in df.columns and i < len(df) and df["HBullD_gen"].iloc[i] == 1:
                fplt.plot(pd.to_datetime(df["HBullD_Lower_Low_date_gen"][i]), df["HBullD_Lower_Low_gen"][i], style="o", ax=ax1, color="red", width=1, legend="○ Red")
                fplt.plot(pd.to_datetime(df["HBullD_Higher_Low_date_gen"][i]), df["HBullD_Higher_Low_gen"][i], style="o", ax=ax1, color="blue", width=1, legend="○ Blue")
                fplt.plot(pd.to_datetime(df["HBullD_Lower_Low_date_gen"][i]), df["HBullD_Lower_Low_RSI_gen"][i], style="o", ax=ax2, color="red", width=1)
                fplt.plot(pd.to_datetime(df["HBullD_Higher_Low_date_gen"][i]), df["HBullD_Higher_Low_RSI_gen"][i], style="o", ax=ax2, color="blue", width=1)
                fplt.plot(pd.to_datetime(df["HBullD_Lower_Low_date_gen"][i]), df["HBullD_Lower_Low_MACD_gen"][i], style="o", ax=ax3, color="red", width=1)
                fplt.plot(pd.to_datetime(df["HBullD_Higher_Low_date_gen"][i]), df["HBullD_Higher_Low_MACD_gen"][i], style="o", ax=ax3, color="blue", width=1)

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
            if "CBullD_x2_neg_MACD" in df.columns and i < len(df) and df["CBullD_x2_neg_MACD"].iloc[i] == 1:
                markers.append({'Type': 'CBullDivg_x2_Hidden', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['CBullDivg_x2']['hidden'] += 1
                counts['CBullDivg_x2']['total'] += 1
        
        if 'HBearDivg' in analysis_results:
            if "HBearD_gen" in df.columns and i < len(df) and df["HBearD_gen"].iloc[i] == 1:
                markers.append({'Type': 'HBearDivg_Classic', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['HBearDivg']['classic'] += 1
                counts['HBearDivg']['total'] += 1
            if "HBearD_neg_MACD" in df.columns and i < len(df) and df["HBearD_neg_MACD"].iloc[i] == 1:
                markers.append({'Type': 'HBearDivg_Hidden', 'Date': df['date'].iloc[i], 'Candle_Percent': candle_percent, 'MACD_Percent': macd_percent})
                counts['HBearDivg']['hidden'] += 1
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
            
            # Group and sort for better organization
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

    # Plot setup - WHITE TEXT, GREEN CANDLES
    fplt.background = fplt.odd_plot_background = "#242320"
    fplt.cross_hair_color = "#eefa"
    fplt.foreground = "white"
    fplt.candle_bull_color = '#44ff44'
    fplt.candle_bear_color = '#ff4444'
    
    fplt.zoom = True
    fplt.pan = True
    
    ax1, ax2, ax3 = fplt.create_plot("Technical Analysis", rows=3)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    candles = df[["date", "open", "close", "high", "low", "macd_histogram"]]
    
    fplt.candlestick_ochl(candles, ax=ax1)
    
    # Orange RSI
    fplt.plot(df.RSI, color="orange", width=2, ax=ax2, legend="RSI")
    fplt.set_y_range(0, 100, ax=ax2)
    
    # MACD
    fplt.volume_ocv(df[["date", "open", "close", "macd_histogram"]], ax=ax3, colorfunc=fplt.strength_colorfilter)
    
    # EMAs
    df.EMA_20.plot(ax=ax1, legend="20-EMA")
    df.EMA_50.plot(ax=ax1, legend="50-EMA") 
    df.EMA_100.plot(ax=ax1, legend="100-EMA")
    df.EMA_200.plot(ax=ax1, legend="200-EMA")
    
    plot_markers(df, analysis_results)
    
    output_filename = f"markers_output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    counts = export_markers_to_csv(df, output_filename, analysis_results, candle_percent, macd_percent)
    
    # TEXT BOX WITH MARKER COUNTS - UPPER LEFT CORNER
    count_text = "\n".join([f"{name}: {data['total']}" for name, data in counts.items()])
    fplt.add_text((df.index[10], df['high'].max() * 0.95), count_text, ax=ax1, color='white', fontsize=14)
    
    # Run second variant if specified
    if variant2:
        print(f"\nRunning second variant: candle={variant2[0]}, macd={variant2[1]}")
        df_var2 = df.copy()
        variant2_results = run_analysis(df_var2, analysis_type, window, variant2[0], variant2[1])
        
        # Add variant2 markers in orange/cyan
        for i in range(2, len(df_var2)):
            if 'CBullDivg' in variant2_results and "CBullD_gen" in df_var2.columns and df_var2["CBullD_gen"].iloc[i] == 1:
                fplt.plot(pd.to_datetime(df_var2["CBullD_Lower_Low_date_gen"][i]), df_var2["CBullD_Lower_Low_gen"][i], style="+", ax=ax1, color="orange", width=2)
                fplt.plot(pd.to_datetime(df_var2["CBullD_Higher_Low_date_gen"][i]), df_var2["CBullD_Higher_Low_gen"][i], style="+", ax=ax1, color="cyan", width=2)
        
        variant2_counts = export_markers_to_csv(df_var2, f"variant2_{variant2[0]}_{variant2[1]}.csv", variant2_results, variant2[0], variant2[1])
        variant2_text = f"Variant2: " + ", ".join([f"{name}={data['total']}" for name, data in variant2_counts.items()])
        fplt.add_text((df.index[10], df['high'].max() * 0.85), variant2_text, ax=ax1, color='yellow', fontsize=12)
    
    # Comparison circles for variants b,c,d vs a (YELLOW/BLUE OUTLINE CIRCLES)
    if analysis_type in ['b', 'c', 'd']:
        print("\nRunning reference CBullDivg for comparison...")
        df_ref = df.copy()
        CBullDivg_analysis(df_ref, window, candle_percent, macd_percent)
        
        ref_positions = set()
        current_positions = set()
        
        for i in range(len(df_ref)):
            if "CBullD_gen" in df_ref.columns and df_ref["CBullD_gen"].iloc[i] == 1:
                ref_positions.add(i)
            if "CBullD_neg_MACD" in df_ref.columns and df_ref["CBullD_neg_MACD"].iloc[i] == 1:
                ref_positions.add(i)
        
        for analysis_type_key in analysis_results:
            for i in range(len(df)):
                if analysis_type_key == 'CBullDivg_x2':
                    if "CBullD_x2_gen" in df.columns and df["CBullD_x2_gen"].iloc[i] == 1:
                        current_positions.add(i)
                elif analysis_type_key == 'HBearDivg':
                    if "HBearD_gen" in df.columns and df["HBearD_gen"].iloc[i] == 1:
                        current_positions.add(i)
                elif analysis_type_key == 'HBullDivg':
                    if "HBullD_gen" in df.columns and df["HBullD_gen"].iloc[i] == 1:
                        current_positions.add(i)
        
        # YELLOW circles for additional markers, BLUE circles for missing markers - OUTLINE ONLY
        additional = current_positions - ref_positions
        missing = ref_positions - current_positions
        
        for pos in additional:
            if pos < len(df):
                fplt.plot(pd.to_datetime(df["date"].iloc[pos]), df["high"].iloc[pos] * 1.02, 
                         style="o", ax=ax1, color="yellow", width=1, legend="Additional")
        
        for pos in missing:
            if pos < len(df):
                fplt.plot(pd.to_datetime(df["date"].iloc[pos]), df["low"].iloc[pos] * 0.98, 
                         style="o", ax=ax1, color="blue", width=1, legend="Missing")
    
    # RESET BUTTON - UPPER RIGHT CORNER
    fplt.add_text((df.index[-50], df['high'].max() * 0.95), "RESET", ax=ax1, color='red', fontsize=16)
    fplt.add_text((df.index[-50], df['high'].max() * 0.92), "(R key)", ax=ax1, color='red', fontsize=10)
    
    fplt.show()
else:
    # DOE functionality - CSV INPUT ONLY
    print("Running DOE analysis...")
    doe_file = input("Enter DOE parameter file (default: doe_parameters_example.csv): ").strip()
    if not doe_file:
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
        
        # CONSOLIDATED DOE RESULTS IN SINGLE XLSX
        doe_results_df = pd.DataFrame(all_results)
        doe_summary_file = f"DOE_Complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        if openpyxl:
            with pd.ExcelWriter(doe_summary_file, engine='openpyxl') as writer:
                # All results sorted by parameters
                doe_sorted = doe_results_df.sort_values(['candle_percent', 'macd_percent', 'analysis_type'])
                doe_sorted.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Summary by parameters
                summary = doe_results_df.groupby(['candle_percent', 'macd_percent']).agg({
                    'classic_count': 'sum',
                    'hidden_count': 'sum', 
                    'total_count': 'sum'
                }).reset_index().sort_values(['total_count'], ascending=False)
                summary.to_excel(writer, sheet_name='Parameter_Summary', index=False)
                
                # By analysis type
                for analysis_type in doe_results_df['analysis_type'].unique():
                    subset = doe_results_df[doe_results_df['analysis_type'] == analysis_type].copy()
                    subset = subset.sort_values(['candle_percent', 'macd_percent'])
                    subset.to_excel(writer, sheet_name=f'{analysis_type}', index=False)
            
            print(f"\nDOE complete results exported to {doe_summary_file}")
        
        # Plot best parameter combination
        best_params = doe_results_df.groupby(['candle_percent', 'macd_percent'])['total_count'].sum().idxmax()
        print(f"\nPlotting best parameters: candle={best_params[0]}, macd={best_params[1]}")
        
        df_best = df.copy()
        best_results = run_analysis(df_best, 'e', window, best_params[0], best_params[1])
        
        # Plot DOE results
        fplt.background = fplt.odd_plot_background = "#242320"
        fplt.cross_hair_color = "#eefa"
        fplt.foreground = "white"
        fplt.candle_bull_color = '#44ff44'
        fplt.candle_bear_color = '#ff4444'
        fplt.zoom = True
        fplt.pan = True
        
        ax1, ax2, ax3 = fplt.create_plot(f"DOE Best: C={best_params[0]}, M={best_params[1]}", rows=3)
        df_best["date"] = pd.to_datetime(df_best["date"], format="mixed")
        candles = df_best[["date", "open", "close", "high", "low", "macd_histogram"]]
        
        fplt.candlestick_ochl(candles, ax=ax1)
        fplt.plot(df_best.RSI, color="orange", width=2, ax=ax2, legend="RSI")
        fplt.set_y_range(0, 100, ax=ax2)
        fplt.volume_ocv(df_best[["date", "open", "close", "macd_histogram"]], ax=ax3, colorfunc=fplt.strength_colorfilter)
        
        df_best.EMA_20.plot(ax=ax1, legend="20-EMA")
        df_best.EMA_50.plot(ax=ax1, legend="50-EMA") 
        df_best.EMA_100.plot(ax=ax1, legend="100-EMA")
        df_best.EMA_200.plot(ax=ax1, legend="200-EMA")
        
        plot_markers(df_best, best_results)
        
        best_counts = export_markers_to_csv(df_best, f"doe_best.csv", best_results, best_params[0], best_params[1])
        count_text = "\n".join([f"{name}: {data['total']}" for name, data in best_counts.items()])
        fplt.add_text((df_best.index[10], df_best['high'].max() * 0.95), count_text, ax=ax1, color='white', fontsize=14)
        
        fplt.add_text((df_best.index[-50], df_best['high'].max() * 0.95), "RESET", ax=ax1, color='red', fontsize=16)
        
        fplt.show()
    else:
        print(f"DOE parameter file not found: {doe_file}")