import finplot as fplt
import pandas as pd
import os
import csv
import glob
import subprocess
import sys
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min

def get_input_file():
    try:
        # Use PowerShell to open file dialog
        ps_command = '''
Add-Type -AssemblyName System.Windows.Forms
$openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
$openFileDialog.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
$openFileDialog.Title = "Select CSV input file"
$openFileDialog.InitialDirectory = "C:\Projekte\crt_250816\data\raw"
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
            # Fallback to console selection
            return get_input_file_console()
    except Exception:
        # Fallback to console selection
        return get_input_file_console()

def get_input_file_console():
    print("\nAvailable CSV files in current directory and subdirectories:")
    csv_files = []
    
    # Search for CSV files in common locations
    search_paths = [
        "*.csv",
        "../*/*.csv", 
        "../../data/raw/*.csv",
        "../../data/*.csv"
    ]
    
    for pattern in search_paths:
        csv_files.extend(glob.glob(pattern))
    
    if not csv_files:
        # Manual input if no files found
        print("No CSV files found automatically.")
        file_path = input("Please enter the full path to your CSV file: ")
        return file_path
    
    # Display available files
    for i, file in enumerate(csv_files, 1):
        print(f"{i}: {file}")
    
    print(f"{len(csv_files) + 1}: Enter custom path")
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(csv_files) + 1}): ")
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(csv_files):
                return csv_files[choice_num - 1]
            elif choice_num == len(csv_files) + 1:
                return input("Enter full path to CSV file: ")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_analysis_parameters():
    print("\nEnter analysis parameters:")
    
    while True:
        try:
            candle_percent = float(input("Enter Candle tolerance % (e.g., 0.1): "))
            if candle_percent > 0:
                break
            print("Candle % must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            macd_percent = float(input("Enter MACD tolerance % (e.g., 3.25): "))
            if macd_percent > 0:
                break
            print("MACD % must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    return candle_percent, macd_percent

def get_analysis_type():
    print("\nSelect analysis type:")
    print("a: CBullDivg_Analysis (Classic Bullish Divergence)")
    print("b: BullDivg_x2_analysis (Extended Bullish Divergence)")
    print("c: HBearDivg_analysis (Hidden Bearish Divergence)")
    print("d: HBullDivg_analysis (Hidden Bullish Divergence)")
    print("e: All analyses (a-d)")
    print("f: DOE (Design of Experiments) - Run multiple parameter combinations")
    
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

def get_doe_parameters():
    print("\nSelect DOE parameters file:")
    doe_files = glob.glob("*.csv") + glob.glob("doe*.csv")
    
    if doe_files:
        for i, file in enumerate(doe_files, 1):
            print(f"{i}: {file}")
        print(f"{len(doe_files) + 1}: Enter custom path")
        
        while True:
            try:
                choice = input(f"\nSelect DOE file (1-{len(doe_files) + 1}): ")
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(doe_files):
                    return doe_files[choice_num - 1]
                elif choice_num == len(doe_files) + 1:
                    return input("Enter full path to DOE CSV file: ")
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        return input("Enter path to DOE parameters CSV file: ")

def run_doe_analysis(df, doe_file, window=5):
    doe_df = pd.read_csv(doe_file)
    all_results = []
    
    print(f"\nRunning DOE with {len(doe_df)} parameter combinations...")
    
    for idx, row in doe_df.iterrows():
        candle_percent = row['candle_percent']
        macd_percent = row['macd_percent']
        
        print(f"\nRun {idx+1}/{len(doe_df)}: Candle%={candle_percent}, MACD%={macd_percent}")
        
        # Create copy of dataframe for this run
        df_copy = df.copy()
        
        # Run all analyses
        analysis_results = run_analysis(df_copy, 'e', window, candle_percent, macd_percent)
        
        # Export results with run identifier
        output_filename = f"doe_run_{idx+1:02d}_c{candle_percent}_m{macd_percent}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_markers_to_csv(df_copy, output_filename, analysis_results, candle_percent, macd_percent)
        
        all_results.append({
            'run': idx+1,
            'candle_percent': candle_percent,
            'macd_percent': macd_percent,
            'df': df_copy,
            'results': analysis_results
        })
    
    return all_results

def export_markers_to_csv(df, filename, analysis_results, candle_percent, macd_percent):
    markers = []
    counts = {}
    
    # Initialize counts for each analysis type
    for analysis in ['CBullDivg', 'CBullDivg_x2', 'HBearDivg', 'HBullDivg']:
        if analysis in analysis_results:
            counts[analysis] = {'classic': 0, 'hidden': 0, 'total': 0}
    
    for i in range(len(df)):
        if 'CBullDivg' in analysis_results:
            if "CBullD_gen" in df.columns and i < len(df) and df["CBullD_gen"].iloc[i] == 1:
                markers.append({
                    'Type': 'CBullDivg_Classic',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['CBullDivg']['classic'] += 1
                counts['CBullDivg']['total'] += 1
            if "CBullD_neg_MACD" in df.columns and i < len(df) and df["CBullD_neg_MACD"].iloc[i] == 1:
                markers.append({
                    'Type': 'CBullDivg_Hidden',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['CBullDivg']['hidden'] += 1
                counts['CBullDivg']['total'] += 1
        
        if 'CBullDivg_x2' in analysis_results:
            if "CBullD_x2_gen" in df.columns and i < len(df) and df["CBullD_x2_gen"].iloc[i] == 1:
                markers.append({
                    'Type': 'CBullDivg_x2_Classic',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['CBullDivg_x2']['classic'] += 1
                counts['CBullDivg_x2']['total'] += 1
            if "CBullD_x2_neg_MACD" in df.columns and i < len(df) and df["CBullD_x2_neg_MACD"].iloc[i] == 1:
                markers.append({
                    'Type': 'CBullDivg_x2_Hidden',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['CBullDivg_x2']['hidden'] += 1
                counts['CBullDivg_x2']['total'] += 1
        
        if 'HBearDivg' in analysis_results:
            if "HBearD_gen" in df.columns and i < len(df) and df["HBearD_gen"].iloc[i] == 1:
                markers.append({
                    'Type': 'HBearDivg_Classic',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['HBearDivg']['classic'] += 1
                counts['HBearDivg']['total'] += 1
            if "HBearD_neg_MACD" in df.columns and i < len(df) and df["HBearD_neg_MACD"].iloc[i] == 1:
                markers.append({
                    'Type': 'HBearDivg_Hidden',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['HBearDivg']['hidden'] += 1
                counts['HBearDivg']['total'] += 1
        
        if 'HBullDivg' in analysis_results:
            if "HBullD_gen" in df.columns and i < len(df) and df["HBullD_gen"].iloc[i] == 1:
                markers.append({
                    'Type': 'HBullDivg_Classic',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['HBullDivg']['classic'] += 1
                counts['HBullDivg']['total'] += 1
            if "HBullD_neg_MACD" in df.columns and i < len(df) and df["HBullD_neg_MACD"].iloc[i] == 1:
                markers.append({
                    'Type': 'HBullDivg_Hidden',
                    'Date': df['date'].iloc[i],
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
                counts['HBullDivg']['hidden'] += 1
                counts['HBullDivg']['total'] += 1
    
    if markers:
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        results_filename = os.path.join('results', filename)
        
        with open(results_filename, 'w', newline='') as csvfile:
            fieldnames = ['Type', 'Date', 'Candle_Percent', 'MACD_Percent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write summary as first rows
            for analysis_name, count_data in counts.items():
                writer.writerow({
                    'Type': f'{analysis_name}_Summary',
                    'Date': f'Classic:{count_data["classic"]} Hidden:{count_data["hidden"]} Total:{count_data["total"]}',
                    'Candle_Percent': candle_percent,
                    'MACD_Percent': macd_percent
                })
            
            # Empty row separator
            writer.writerow({'Type': '---', 'Date': '---', 'Candle_Percent': '---', 'MACD_Percent': '---'})
            
            # Write actual markers
            for marker in markers:
                writer.writerow(marker)
        print(f"\nMarkers exported to {results_filename}")
        
        # Console summary output
        print("\\n=== ANALYSIS SUMMARY ===")
        for analysis_name, count_data in counts.items():
            print(f"{analysis_name}: Classic={count_data['classic']}, Hidden={count_data['hidden']}, Total={count_data['total']}")
        print(f"\\nGrand Total markers found: {len(markers)}")
        
        # Console output
        print("\n--- Found Markers ---")
        for marker in markers:
            print(f"{marker['Type']}: {marker['Date']} (Candle%: {marker['Candle_Percent']}, MACD%: {marker['MACD_Percent']})")
    else:
        print("No markers found to export.")

csv_file_path = get_input_file()
if not csv_file_path:
    print("No file selected. Exiting.")
    exit()

analysis_type = get_analysis_type()

if analysis_type == 'f':
    doe_file = get_doe_parameters()
    print(f"Loading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    print("Initializing indicators...")
    Initialize_RSI_EMA_MACD(df)
    Local_Max_Min(df)
    
    window = 5
    all_doe_results = run_doe_analysis(df, doe_file, window)
    
    print(f"\nDOE completed. {len(all_doe_results)} runs finished.")
    print("Results exported to individual CSV files in results/ folder.")
    exit()
else:
    candle_percent, macd_percent = get_analysis_parameters()

print(f"Loading data from: {csv_file_path}")
df = pd.read_csv(csv_file_path, low_memory=False)

print("Initializing indicators...")
Initialize_RSI_EMA_MACD(df)
Local_Max_Min(df)

window = 5

print(f"Running analysis with parameters: window={window}, candle_tol={candle_percent}, macd_tol={macd_percent}")
analysis_results = run_analysis(df, analysis_type, window, candle_percent, macd_percent)


fplt.background = fplt.odd_plot_background = "#242320"  # Adjust Plot Background colour
fplt.cross_hair_color = "#eefa"  # Adjust Crosshair colour

# Enable zoom and pan
fplt.zoom = True
fplt.pan = True

# Plotting Chart----------------------------------------------
# Plotting Candlesticks---------------------------------------
ax1, ax2, ax3 = fplt.create_plot("Chart", rows=3)
df["date"] = pd.to_datetime(df["date"], format="mixed")
candles = df[["date", "open", "close", "high", "low", "macd_histogram"]]
# candles = df[['date', 'open', 'close', 'high', 'low']]
fplt.candlestick_ochl(candles, ax=ax1)  # Plotting candlestick chart using

# Plotting RSI
fplt.plot(df.RSI, color="#000000", width=2, ax=ax2, legend="RSI")
fplt.set_y_range(0, 100, ax=ax2)  # Setting y-axis range
# fplt.add_horizontal_band(0, 100, color='#FFFFFF', ax=ax2)  # Changing background color to white
# fplt.add_horizontal_band(30, 70, color='#ffcccc', ax=ax2)  # Adding band for 30-70 RSI
fplt.add_horizontal_band(
    0, 1, color="#000000", ax=ax2
)  # Dummy band to mark the ending of the plot
fplt.add_horizontal_band(
    99, 100, color="#000000", ax=ax2
)  # Dummy band to mark the ending of the plot

# Plotting the MACD
fplt.volume_ocv(
    df[["date", "open", "close", "macd_histogram"]],
    ax=ax3,
    colorfunc=fplt.strength_colorfilter,
)

# Plotting EMAs-----------------------------------------------
df.EMA_20.plot(
    ax=ax1, legend="20-EMA"
)  # Plotting exponential moving average period = 20
df.EMA_50.plot(
    ax=ax1, legend="50-EMA"
)  # Plotting exponential moving average period = 50
df.EMA_100.plot(
    ax=ax1, legend="100-EMA"
)  # Plotting exponential moving average period = 100
df.EMA_200.plot(
    ax=ax1, legend="200-EMA"
)  # Plotting exponential moving average period = 200

def get_marker_positions(df, analysis_type):
    positions = set()
    
    if analysis_type == 'CBullDivg':
        for i in range(len(df)):
            if "CBullD_gen" in df.columns and i < len(df) and df["CBullD_gen"].iloc[i] == 1:
                positions.add(i)
            if "CBullD_neg_MACD" in df.columns and i < len(df) and df["CBullD_neg_MACD"].iloc[i] == 1:
                positions.add(i)
    elif analysis_type == 'CBullDivg_x2':
        for i in range(len(df)):
            if "CBullD_x2_gen" in df.columns and i < len(df) and df["CBullD_x2_gen"].iloc[i] == 1:
                positions.add(i)
            if "CBullD_x2_neg_MACD" in df.columns and i < len(df) and df["CBullD_x2_neg_MACD"].iloc[i] == 1:
                positions.add(i)
    elif analysis_type == 'HBearDivg':
        for i in range(len(df)):
            if "HBearD_gen" in df.columns and i < len(df) and df["HBearD_gen"].iloc[i] == 1:
                positions.add(i)
            if "HBearD_neg_MACD" in df.columns and i < len(df) and df["HBearD_neg_MACD"].iloc[i] == 1:
                positions.add(i)
    elif analysis_type == 'HBullDivg':
        for i in range(len(df)):
            if "HBullD_gen" in df.columns and i < len(df) and df["HBullD_gen"].iloc[i] == 1:
                positions.add(i)
            if "HBullD_neg_MACD" in df.columns and i < len(df) and df["HBullD_neg_MACD"].iloc[i] == 1:
                positions.add(i)
    
    return positions

def plot_markers(df, analysis_results, show_comparison=False, reference_df=None, reference_results=None):
    for i in range(2, len(df)):
        # CBullDivg Analysis markers
        if 'CBullDivg' in analysis_results:
            if "CBullD_gen" in df.columns and i < len(df) and df["CBullD_gen"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]),
                    df["CBullD_Lower_Low_gen"][i],
                    style="x",
                    ax=ax1,
                    color="red",
                    width=3,
                    legend="CBull Classic Low"
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]),
                    df["CBullD_Higher_Low_gen"][i],
                    style="x",
                    ax=ax1,
                    color="blue",
                    width=3,
                    legend="CBull Classic High"
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]),
                    df["CBullD_Lower_Low_RSI_gen"][i],
                    style="x",
                    ax=ax2,
                    color="red",
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]),
                    df["CBullD_Higher_Low_RSI_gen"][i],
                    style="x",
                    ax=ax2,
                    color="blue",
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]),
                    df["CBullD_Lower_Low_MACD_gen"][i],
                    style="x",
                    ax=ax3,
                    color="red",
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]),
                    df["CBullD_Higher_Low_MACD_gen"][i],
                    style="x",
                    ax=ax3,
                    color="blue",
                )

            if "CBullD_neg_MACD" in df.columns and i < len(df) and df["CBullD_neg_MACD"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]),
                    df["CBullD_Lower_Low_neg_MACD"][i],
                    style="x",
                    ax=ax1,
                    color="red",
                    width=3,
                    legend="CBull Hidden Low"
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]),
                    df["CBullD_Higher_Low_neg_MACD"][i],
                    style="x",
                    ax=ax1,
                    color="blue",
                    width=3,
                    legend="CBull Hidden High"
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]),
                    df["CBullD_Lower_Low_RSI_neg_MACD"][i],
                    style="x",
                    ax=ax2,
                    color="red",
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]),
                    df["CBullD_Higher_Low_RSI_neg_MACD"][i],
                    style="x",
                    ax=ax2,
                    color="blue",
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]),
                    df["CBullD_Lower_Low_MACD_neg_MACD"][i],
                    style="x",
                    ax=ax3,
                    color="red",
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]),
                    df["CBullD_Higher_Low_MACD_neg_MACD"][i],
                    style="x",
                    ax=ax3,
                    color="blue",
                )
        
        # CBullDivg_x2 Analysis markers
        if 'CBullDivg_x2' in analysis_results:
            if "CBullD_x2_gen" in df.columns and i < len(df) and df["CBullD_x2_gen"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["CBullD_x2_Lower_Low_date_gen"][i]),
                    df["CBullD_x2_Lower_Low_gen"][i],
                    style="^",
                    ax=ax1,
                    color="orange",
                    width=4,
                    legend="CBull_x2 Classic Low"
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_x2_Higher_Low_date_gen"][i]),
                    df["CBullD_x2_Higher_Low_gen"][i],
                    style="^",
                    ax=ax1,
                    color="cyan",
                    width=4,
                    legend="CBull_x2 Classic High"
                )
            if "CBullD_x2_neg_MACD" in df.columns and i < len(df) and df["CBullD_x2_neg_MACD"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["CBullD_x2_Lower_Low_date_neg_MACD"][i]),
                    df["CBullD_x2_Lower_Low_neg_MACD"][i],
                    style="^",
                    ax=ax1,
                    color="orange",
                    width=4,
                    legend="CBull_x2 Hidden Low"
                )
                fplt.plot(
                    pd.to_datetime(df["CBullD_x2_Higher_Low_date_neg_MACD"][i]),
                    df["CBullD_x2_Higher_Low_neg_MACD"][i],
                    style="^",
                    ax=ax1,
                    color="cyan",
                    width=4,
                    legend="CBull_x2 Hidden High"
                )
        
        # HBearDivg Analysis markers
        if 'HBearDivg' in analysis_results:
            if "HBearD_gen" in df.columns and i < len(df) and df["HBearD_gen"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["HBearD_Higher_High_date_gen"][i]),
                    df["HBearD_Higher_High_gen"][i],
                    style="v",
                    ax=ax1,
                    color="purple",
                    width=4,
                    legend="HBear Classic High"
                )
                fplt.plot(
                    pd.to_datetime(df["HBearD_Lower_High_date_gen"][i]),
                    df["HBearD_Lower_High_gen"][i],
                    style="v",
                    ax=ax1,
                    color="yellow",
                    width=4,
                    legend="HBear Classic Low"
                )
            if "HBearD_neg_MACD" in df.columns and i < len(df) and df["HBearD_neg_MACD"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["HBearD_Higher_High_date_neg_MACD"][i]),
                    df["HBearD_Higher_High_neg_MACD"][i],
                    style="v",
                    ax=ax1,
                    color="purple",
                    width=4,
                    legend="HBear Hidden High"
                )
                fplt.plot(
                    pd.to_datetime(df["HBearD_Lower_High_date_neg_MACD"][i]),
                    df["HBearD_Lower_High_neg_MACD"][i],
                    style="v",
                    ax=ax1,
                    color="yellow",
                    width=4,
                    legend="HBear Hidden Low"
                )
        
        # HBullDivg Analysis markers
        if 'HBullDivg' in analysis_results:
            if "HBullD_gen" in df.columns and i < len(df) and df["HBullD_gen"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["HBullD_Lower_Low_date_gen"][i]),
                    df["HBullD_Lower_Low_gen"][i],
                    style="s",
                    ax=ax1,
                    color="green",
                    width=4,
                    legend="HBull Classic Low"
                )
                fplt.plot(
                    pd.to_datetime(df["HBullD_Higher_Low_date_gen"][i]),
                    df["HBullD_Higher_Low_gen"][i],
                    style="s",
                    ax=ax1,
                    color="lime",
                    width=4,
                    legend="HBull Classic High"
                )
            if "HBullD_neg_MACD" in df.columns and i < len(df) and df["HBullD_neg_MACD"].iloc[i] == 1:
                fplt.plot(
                    pd.to_datetime(df["HBullD_Lower_Low_date_neg_MACD"][i]),
                    df["HBullD_Lower_Low_neg_MACD"][i],
                    style="s",
                    ax=ax1,
                    color="green",
                    width=4,
                    legend="HBull Hidden Low"
                )
                fplt.plot(
                    pd.to_datetime(df["HBullD_Higher_Low_date_neg_MACD"][i]),
                    df["HBullD_Higher_Low_neg_MACD"][i],
                    style="s",
                    ax=ax1,
                    color="lime",
                    width=4,
                    legend="HBull Hidden High"
                )

plot_markers(df, analysis_results)

# Add comparison circles for variants 2-4 vs variant 1 (CBullDivg)
if analysis_type in ['b', 'c', 'd'] and len(analysis_results) == 1:
    # Run reference analysis (variant a) for comparison
    print("\\nRunning reference analysis (CBullDivg) for comparison...")
    df_ref = df.copy()
    CBullDivg_analysis(df_ref, window, candle_percent, macd_percent)
    ref_results = {'CBullDivg': True}
    
    ref_positions = get_marker_positions(df_ref, 'CBullDivg')
    current_type = list(analysis_results.keys())[0]
    current_positions = get_marker_positions(df, current_type)
    
    # Mark additional markers (yellow circles)
    additional_markers = current_positions - ref_positions
    for pos in additional_markers:
        if pos < len(df):
            fplt.plot(
                pd.to_datetime(df["date"].iloc[pos]),
                df["high"].iloc[pos] * 1.02,
                style="o",
                ax=ax1,
                color="yellow",
                width=6,
                legend="Additional vs CBull"
            )
    
    # Mark missing markers (blue circles)
    missing_markers = ref_positions - current_positions
    for pos in missing_markers:
        if pos < len(df):
            fplt.plot(
                pd.to_datetime(df["date"].iloc[pos]),
                df["low"].iloc[pos] * 0.98,
                style="o",
                ax=ax1,
                color="blue",
                width=6,
                legend="Missing vs CBull"
            )

output_filename = f"markers_output_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
export_markers_to_csv(df, output_filename, analysis_results, candle_percent, macd_percent)

fplt.show()
