import pandas as pd
import warnings
import os
import numpy as np
from Initialize_RSI_EMA_MACD_vectorized import Initialize_RSI_EMA_MACD
from CS_Type import Candlestick_Type
from Level_1_Maximas_Minimas import Level_1_Max_Min
from HBearDivg_analysis_vectorized import HBearDivg_analysis
from HBullDivg_analysis_vectorized import HBullDivg_analysis
from CBearDivg_analysis_vectorized import CBearDivg_analysis
from CBullDivg_analysis_vectorized import CBullDivg_analysis
from CBullDivg_x2_analysis_vectorized import CBullDivg_x2_analysis
from Goldenratio_vectorized import calculate_golden_ratios
from Support_Resistance_vectorized import calculate_support_levels
import multiprocessing as mp
import json
import sys
import subprocess
import signal
import time
import glob
import argparse

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented*')

# Paths (set via GUI; no hardcoded defaults)
csv_file_path = ''
output_dir = ''

# Preferences file to remember last used paths
_PREFS_PATH = os.path.join(os.path.dirname(__file__), '.io_paths.json')

def _load_prefs(path=_PREFS_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _save_prefs(prefs, path=_PREFS_PATH):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Interactive CSV chooser via Explorer; falls back to default if canceled.
def _choose_csv_or_fail(default_path):
    prefs = _load_prefs()
    env_sel = os.environ.get('CSV_FILE_PATH_SELECTED')
    if env_sel and os.path.isfile(env_sel):
        return env_sel
    if os.path.isfile(default_path):
        return default_path
    # In child processes, avoid GUI prompts and rely on inherited env/default
    if mp.current_process().name != 'MainProcess':
        raise FileNotFoundError(f"CSV not found for child process. Ensure parent selected a file or set CSV_FILE_PATH_SELECTED env.")
    # Main process: prompt via Explorer
    def _win_open_file_dialog():
        try:
            import sys, ctypes
            from ctypes import wintypes
            if sys.platform != 'win32':
                return None
            class OPENFILENAMEW(ctypes.Structure):
                _fields_ = [
                    ("lStructSize", wintypes.DWORD),
                    ("hwndOwner", wintypes.HWND),
                    ("hInstance", wintypes.HINSTANCE),
                    ("lpstrFilter", wintypes.LPCWSTR),
                    ("lpstrCustomFilter", wintypes.LPWSTR),
                    ("nMaxCustFilter", wintypes.DWORD),
                    ("nFilterIndex", wintypes.DWORD),
                    ("lpstrFile", wintypes.LPWSTR),
                    ("nMaxFile", wintypes.DWORD),
                    ("lpstrFileTitle", wintypes.LPWSTR),
                    ("nMaxFileTitle", wintypes.DWORD),
                    ("lpstrInitialDir", wintypes.LPCWSTR),
                    ("lpstrTitle", wintypes.LPCWSTR),
                    ("Flags", wintypes.DWORD),
                    ("nFileOffset", wintypes.WORD),
                    ("nFileExtension", wintypes.WORD),
                    ("lpstrDefExt", wintypes.LPCWSTR),
                    ("lCustData", wintypes.LPARAM),
                    ("lpfnHook", wintypes.LPVOID),
                    ("lpTemplateName", wintypes.LPCWSTR),
                    ("pvReserved", wintypes.LPVOID),
                    ("dwReserved", wintypes.DWORD),
                    ("FlagsEx", wintypes.DWORD),
                ]
            GetOpenFileNameW = ctypes.windll.comdlg32.GetOpenFileNameW
            ofn = OPENFILENAMEW()
            buf = ctypes.create_unicode_buffer(65536)
            ofn.lStructSize = ctypes.sizeof(ofn)
            ofn.hwndOwner = None
            ofn.lpstrFilter = "CSV files\0*.csv\0All files\0*.*\0\0"
            ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
            ofn.nMaxFile = len(buf)
            ofn.lpstrFileTitle = None
            ofn.nMaxFileTitle = 0
            last_csv_dir = prefs.get('last_csv_dir')
            init_base = last_csv_dir if (last_csv_dir and os.path.isdir(last_csv_dir)) else os.path.dirname(default_path)
            init_dir = init_base if os.path.isdir(init_base or '') else os.getcwd()
            ofn.lpstrInitialDir = init_dir
            ofn.lpstrTitle = "Select input CSV"
            OFN_EXPLORER = 0x00080000
            OFN_FILEMUSTEXIST = 0x00001000
            OFN_PATHMUSTEXIST = 0x00000800
            ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST
            ofn.lpstrDefExt = "csv"
            if GetOpenFileNameW(ctypes.byref(ofn)):
                return buf.value
            return None
        except Exception:
            return None
    # Try Tkinter first; if unavailable, fall back to native Win32 dialog
    selected = None
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd
        last_csv_dir = _load_prefs().get('last_csv_dir')
        for _ in range(2):  # give the user a second chance if they cancel
            _root = _tk.Tk(); _root.withdraw()
            selected = _fd.askopenfilename(title='Select input CSV', initialdir=last_csv_dir if last_csv_dir else None, filetypes=[('CSV files', '*.csv')])
            _root.destroy()
            if selected:
                break
    except Exception:
        selected = _win_open_file_dialog()
    if selected and os.path.isfile(selected):
        os.environ['CSV_FILE_PATH_SELECTED'] = selected
        prefs['last_csv_dir'] = os.path.dirname(selected)
        _save_prefs(prefs)
        return selected
    raise FileNotFoundError(f"CSV not selected or not found. Please select a valid CSV in the dialog.")

def _choose_output_dir(default_dir):
    prefs = _load_prefs()
    env_sel = os.environ.get('OUTPUT_DIR_SELECTED')
    if env_sel:
        return env_sel
    # Avoid GUI in child processes
    if mp.current_process().name != 'MainProcess':
        raise FileNotFoundError("Output directory not set for child process. Ensure parent selected it or set OUTPUT_DIR_SELECTED env.")
    # Try Tkinter directory picker
    selected = None
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd
        last_out = prefs.get('last_output_dir')
        _root = _tk.Tk(); _root.withdraw()
        selected = _fd.askdirectory(title='Select output folder', initialdir=last_out if last_out else None, mustexist=False)
        _root.destroy()
    except Exception:
        # Try modern IFileDialog (FOS_PICKFOLDERS) first, then fallback to SHBrowseForFolder
        try:
            import sys, ctypes
            from ctypes import wintypes
            if sys.platform == 'win32':
                # GUID and COM helpers
                class GUID(ctypes.Structure):
                    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

                CLSID_FileOpenDialog = GUID(0xDC1C5A9C, 0xE88A, 0x4DDE, (ctypes.c_ubyte * 8)(0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7))
                IID_IFileDialog = GUID(0x42F85136, 0xDB7E, 0x439C, (ctypes.c_ubyte * 8)(0x85, 0xF1, 0xE4, 0x07, 0x5D, 0x13, 0x5F, 0xC8))
                IID_IShellItem = GUID(0x43826D1E, 0xE718, 0x42EE, (ctypes.c_ubyte * 8)(0xBC, 0x55, 0xA1, 0xE2, 0x61, 0xC3, 0x7B, 0xFE))

                FOS_PICKFOLDERS = 0x00000020
                FOS_FORCEFILESYSTEM = 0x00000040
                SIGDN_FILESYSPATH = 0x80058000
                CLSCTX_INPROC_SERVER = 0x1

                ole32 = ctypes.windll.ole32

                def _com_call(obj, index, restype, *argtypes):
                    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                    func_type = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
                    return func_type(vtbl[index])

                ole32.CoInitialize(None)
                p_dlg = ctypes.c_void_p()
                hr = ole32.CoCreateInstance(ctypes.byref(CLSID_FileOpenDialog), None, CLSCTX_INPROC_SERVER, ctypes.byref(IID_IFileDialog), ctypes.byref(p_dlg))
                if hr == 0 and p_dlg:
                    try:
                        GetOptions = _com_call(p_dlg, 10, wintypes.HRESULT, ctypes.POINTER(wintypes.DWORD))
                        opts = wintypes.DWORD(0)
                        GetOptions(p_dlg, ctypes.byref(opts))
                        new_opts = wintypes.DWORD(opts.value | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM)
                        SetOptions = _com_call(p_dlg, 9, wintypes.HRESULT, wintypes.DWORD)
                        SetOptions(p_dlg, new_opts)
                        try:
                            SetTitle = _com_call(p_dlg, 17, wintypes.HRESULT, wintypes.LPCWSTR)
                            SetTitle(p_dlg, "Select output folder")
                        except Exception:
                            pass
                        Show = _com_call(p_dlg, 3, wintypes.HRESULT, wintypes.HWND)
                        if Show(p_dlg, None) == 0:
                            GetResult = _com_call(p_dlg, 20, wintypes.HRESULT, ctypes.POINTER(ctypes.c_void_p))
                            p_item = ctypes.c_void_p()
                            if GetResult(p_dlg, ctypes.byref(p_item)) == 0 and p_item:
                                try:
                                    GetDisplayName = _com_call(p_item, 5, wintypes.HRESULT, ctypes.c_int, ctypes.POINTER(wintypes.LPWSTR))
                                    psz = wintypes.LPWSTR()
                                    if GetDisplayName(p_item, SIGDN_FILESYSPATH, ctypes.byref(psz)) == 0 and psz:
                                        selected = psz.value
                                        try:
                                            ole32.CoTaskMemFree(psz)
                                        except Exception:
                                            pass
                                finally:
                                    try:
                                        Release = _com_call(p_item, 2, wintypes.ULONG)
                                        Release(p_item)
                                    except Exception:
                                        pass
                    finally:
                        try:
                            Release = _com_call(p_dlg, 2, wintypes.ULONG)
                            Release(p_dlg)
                        except Exception:
                            pass
                        ole32.CoUninitialize()

                if not selected:
                    class BROWSEINFOW(ctypes.Structure):
                        _fields_ = [
                            ("hwndOwner", wintypes.HWND),
                            ("pidlRoot", ctypes.c_void_p),
                            ("pszDisplayName", wintypes.LPWSTR),
                            ("lpszTitle", wintypes.LPCWSTR),
                            ("ulFlags", wintypes.UINT),
                            ("lpfn", ctypes.c_void_p),
                            ("lParam", wintypes.LPARAM),
                            ("iImage", ctypes.c_int),
                        ]
                    shell32 = ctypes.windll.shell32
                    SHBrowseForFolderW = shell32.SHBrowseForFolderW
                    SHGetPathFromIDListW = shell32.SHGetPathFromIDListW
                    bi = BROWSEINFOW()
                    bi.hwndOwner = None
                    bi.pidlRoot = None
                    display_buf = ctypes.create_unicode_buffer(260)
                    bi.pszDisplayName = ctypes.cast(display_buf, wintypes.LPWSTR)
                    bi.lpszTitle = "Select output folder"
                    BIF_RETURNONLYFSDIRS = 0x00000001
                    BIF_NEWDIALOGSTYLE = 0x00000040
                    BIF_EDITBOX = 0x00000010
                    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX
                    bi.lpfn = None
                    bi.lParam = 0
                    bi.iImage = 0
                    pidl_selected = SHBrowseForFolderW(ctypes.byref(bi))
                    if pidl_selected:
                        out_buf = ctypes.create_unicode_buffer(260)
                        if SHGetPathFromIDListW(pidl_selected, out_buf):
                            selected = out_buf.value
        except Exception:
            selected = None
        # As a last resort, invoke PowerShell FolderBrowserDialog and capture selection
        if not selected:
            try:
                ps_script = (
                    "$ErrorActionPreference='Stop'; "
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    "$f.Description='Select output folder'; $f.ShowNewFolderButton=$true; "
                    "if ($f.ShowDialog() -eq 'OK') { [Console]::WriteLine($f.SelectedPath) }"
                )
                proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps_script],
                                       capture_output=True, text=True, timeout=120)
                cand = (proc.stdout or '').strip()
                if cand and os.path.isdir(cand):
                    selected = cand
            except Exception:
                selected = None
    if selected:
        os.environ['OUTPUT_DIR_SELECTED'] = selected
        prefs['last_output_dir'] = selected
        _save_prefs(prefs)
        return selected
    # If user canceled all prompts but default_dir exists, use it silently
    if default_dir and os.path.isdir(default_dir):
        return default_dir
    raise FileNotFoundError("Output directory not selected. Please choose a valid folder.")

def _confirm_start(csv_path, out_dir):
    msg = f"About to start processing.\nCSV: {csv_path}\nOutput: {out_dir}\nProceed?"
    try:
        if sys.platform == 'win32':
            import ctypes
            MB_YESNO = 0x00000004
            MB_ICONQUESTION = 0x00000020
            res = ctypes.windll.user32.MessageBoxW(None, msg, "Confirm", MB_YESNO | MB_ICONQUESTION)
            return res == 6  # IDYES
    except Exception:
        return True  # proceed if GUI confirm not available; no console input

# Input selection happens in main guard

_df_full = None
_full_parquet_path = None

def _pool_init(parquet_path):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
    global _full_parquet_path, _df_full
    _full_parquet_path = parquet_path
    _df_full = None

def _get_df_full():
    global _df_full
    if _df_full is None:
        _df_full = pd.read_parquet(_full_parquet_path)
    return _df_full

def process_i(i):
    # Slice from original dataset, run full pipeline to match legacy behaviour
    df = _get_df_full().iloc[:i].copy()

    Initialize_RSI_EMA_MACD(df)
    Level_1_Max_Min(df)
    Candlestick_Type(df)
    CBullDivg_analysis(df, 0.05, 3.25)
    CBullDivg_x2_analysis(df, 0.05, 3.25)
    HBullDivg_analysis(df, 0.05, 3.25)
    CBearDivg_analysis(df, 0.05, 3.25)
    HBearDivg_analysis(df, 0.05, 3.25)

    df = calculate_support_levels(df, lookback_years=25, pivot_threshold=0.25)
    df = calculate_golden_ratios(df)

    # Save output (matching original behaviour)
    last_date = df['date'].iloc[-1]
    last_date_sanitized = str(last_date).replace('/', '-').replace(':', '-').replace(' ', '_')
    output_file = os.path.join(output_dir, f'output_{last_date_sanitized}.parquet')
    df.tail(400).to_parquet(output_file, index=False, engine='pyarrow')

    # Write sidecar JSON to make resume detection robust
    try:
        sidecar_path = os.path.splitext(output_file)[0] + '.json'
        meta = {
            'slice_index': int(i),
            'last_date': str(last_date),
            'parquet': output_file,
        }
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass

    # Report completion of this slice
    return i

# ---- Pause/Resume support ----
_INTERRUPT_REQUESTED = False

def _sigint_handler(signum, frame):
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True

def _state_paths(csv_path, out_dir):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    state_path = os.path.join(out_dir, f".state_{base}.json")
    control_path = os.path.join(out_dir, f".control_{base}.txt")
    return state_path, control_path

def _load_state(state_path):
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['completed'] = set(data.get('completed', []))
            return data
    except Exception:
        return {'completed': set(), 'meta': {}}

def _save_state(state_path, state):
    try:
        to_dump = dict(state)
        comp = sorted(list(to_dump.get('completed', [])))
        to_dump['completed'] = comp
        to_dump['meta'] = to_dump.get('meta', {})
        to_dump['meta'].update({'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')})
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(to_dump, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _cleanup_sidecars(out_dir):
    try:
        for sc in glob.glob(os.path.join(out_dir, 'output_*.json')):
            try:
                os.remove(sc)
            except Exception:
                pass
    except Exception:
        pass

def _check_control(control_path):
    try:
        if os.path.isfile(control_path):
            with open(control_path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = (f.read() or '').lower()
            if any(k in txt for k in ['pause', 'stop', 'abort', 'quit']):
                return 'pause'
    except Exception:
        return None
    return None

def _parse_runtime():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--procs', type=str, default=None)
    parser.add_argument('--chunksize', type=int, default=None)
    args, _ = parser.parse_known_args()
    procs_env = os.environ.get('PROCESSES') or os.environ.get('PROC_COUNT')
    chunks_env = os.environ.get('CHUNKSIZE')
    procs = args.procs or procs_env or 'auto'
    try:
        chunksize = args.chunksize or (int(chunks_env) if chunks_env else 16)
    except Exception:
        chunksize = 16
    if isinstance(procs, str) and procs.lower() == 'auto':
        cpu = os.cpu_count() or 1
        procs_count = max(1, cpu - 1)
    else:
        try:
            procs_count = max(1, int(procs))
        except Exception:
            procs_count = max(1, (os.cpu_count() or 2) - 1)
    return procs_count, chunksize

if __name__ == '__main__':
    os.environ.pop('CSV_FILE_PATH_SELECTED', None)
    os.environ.pop('OUTPUT_DIR_SELECTED', None)
    csv_file_path = _choose_csv_or_fail(csv_file_path)
    output_dir = os.path.dirname(csv_file_path)

    if mp.current_process().name == 'MainProcess':
        if not _confirm_start(csv_file_path, output_dir):
            print('Aborted by user.')
            sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)

    # Parallelize the processing using multiprocessing Pool with progress output
    start_i = 200
    # Read full CSV once and expose via parquet to workers
    df_full = pd.read_csv(csv_file_path, header=0, parse_dates=['date'])
    full_parquet_path = os.path.join(output_dir, 'full_df.parquet')
    df_full.to_parquet(full_parquet_path, index=False, engine='pyarrow')
    len_df = len(df_full)
    del df_full
    end_i = len_df - 1
    total_all = max(0, end_i - start_i)

    state_path, control_path = _state_paths(csv_file_path, output_dir)
    state = _load_state(state_path)
    state['completed'] = set(i for i in state.get('completed', set()) if start_i <= i < end_i)
    if not os.path.isfile(state_path):
        try:
            sidecars = glob.glob(os.path.join(output_dir, 'output_*.json'))
            got_any = False
            for sc in sidecars:
                try:
                    with open(sc, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    idx = int(data.get('slice_index'))
                    if start_i <= idx < end_i:
                        state['completed'].add(idx)
                        got_any = True
                except Exception:
                    continue
            if not got_any:
                existing = glob.glob(os.path.join(output_dir, 'output_*.parquet'))
                cnt = len(existing)
                if cnt > 0:
                    upto = min(end_i, start_i + cnt)
                    state['completed'].update(range(start_i, upto))
        except Exception:
            pass
    remaining_iter = [i for i in range(start_i, end_i) if i not in state['completed']]
    total = len(remaining_iter)

    state['meta'] = {
        'csv': csv_file_path,
        'output_dir': output_dir,
        'start_i': start_i,
        'end_i': end_i,
        'total_all': total_all,
    }
    _save_state(state_path, state)

    signal.signal(signal.SIGINT, _sigint_handler)

    if total == 0:
        print('Nothing to process: dataset too small after start offset.')
    else:
        print(f'Starting processing {total} remaining slices (from {start_i} to {end_i - 1})...')
        def _render_progress(done_rem, tot_rem, done_all, tot_all, width=40):
            pct = 0 if tot_rem == 0 else done_rem / tot_rem
            filled = int(pct * width)
            bar = '#' * filled + '-' * (width - filled)
            pct_all = 0 if tot_all == 0 else (done_all / tot_all)
            print(f"\r[Remaining {bar}] {done_rem}/{tot_rem} ({pct*100:0.1f}%) | Overall {done_all}/{tot_all} ({pct_all*100:0.1f}%)", end='', flush=True)
        completed = 0
        _render_progress(completed, total, len(state['completed']), total_all)
        procs_count, chunksize = _parse_runtime()
        with mp.Pool(processes=procs_count, initializer=_pool_init, initargs=(full_parquet_path,)) as pool:
        # with mp.Pool(6) as pool:
            try:
                for i_done in pool.imap_unordered(process_i, remaining_iter, chunksize=chunksize):
                    state['completed'].add(i_done)
                    completed += 1
                    if _INTERRUPT_REQUESTED or _check_control(control_path):
                        print('\nPause requested. Saving state and stopping...')
                        _save_state(state_path, state)
                        break
                    if completed % 25 == 0:
                        _save_state(state_path, state)
                    _render_progress(completed, total, len(state['completed']), total_all)
            finally:
                try:
                    pool.close()
                except Exception:
                    pass
                try:
                    pool.join()
                except Exception:
                    pass
        _save_state(state_path, state)
        _cleanup_sidecars(output_dir)
        print()

    # Optional: Clean up temporary parquet file
    try:
        os.remove(full_parquet_path)
    except Exception:
        pass
