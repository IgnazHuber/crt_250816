# -*- coding: utf-8 -*-
# file: Mainframe_RT_DOE.py
"""
Mainframe – DOE-fähig + Diagnosemodus (vollständiges Modul)

- Übergibt DOE-Parameter (window, candle_tolerance, macd_tolerance) an Local_Max_Min & CBullDivg_analysis
- Adapter rufen beide Funktionen robust auf (alte/neue Signaturen)
- Zählt Divergenzen robust (Case-insensitive, Bool/Int/Float, breite Spalten-Heuristik)
- Diagnosemodus (--diagnose) zeigt Spaltenprüfungen + Value Counts
- Plot unverändert optional vorhanden, keine Features entfernt

API:
  analyze(window, candle_tolerance, macd_tolerance, input_path=None, enable_plot=False, diagnose=False) -> dict
CLI:
  python Mainframe_RT_DOE.py --input <file> --window 5 --ct 0.1 --mt 3.25 --no-plot --diagnose
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import finplot as fplt

# Projekt-Module (wie vorhanden, nichts entfernt)
try:
    from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
    from Local_Maximas_Minimas import Local_Max_Min
    from CBullDivg_Analysis_vectorized import CBullDivg_analysis
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chartanalyse.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class DivergenceParams:
    window: int = 5
    candle_tolerance: float = 0.10
    macd_tolerance: float = 3.25


# ---------- Helper ----------
def _coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols:
        df = df.rename(columns={cols["date"]: "date"})
    elif "timestamp" in cols:
        df = df.rename(columns={cols["timestamp"]: "date"})
    else:
        df["date"] = pd.RangeIndex(start=0, stop=len(df))
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).tz_convert(None)
        if df["date"].isna().all():
            df["date"] = pd.RangeIndex(start=0, stop=len(df))
    except Exception:
        pass
    return df


def _robust_count(df: Optional[pd.DataFrame], primary_names: List[str]) -> int:
    """
    Zählt '1' bzw. True in potenziellen Divergenz-Spalten.
    - primary_names: bevorzugte Spaltennamen (case-insensitive), z.B. ['CBullD_gen']
    - Fallback: sucht heuristisch nach 'cbulld' & 'gen' oder 'neg_macd' in allen Spalten
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return 0

    cols_lower = {c.lower(): c for c in df.columns}

    # 1) bevorzugte Namen (case-insensitive)
    for name in primary_names:
        ckey = cols_lower.get(name.lower())
        if ckey:
            s = df[ckey]
            try:
                return int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())
            except Exception:
                return int(s.astype(bool).sum())

    # 2) Heuristik
    # classic: irgendwas mit cbulld + gen oder classic+div/signal
    for c in df.columns:
        cl = c.lower()
        if ("cbulld" in cl and "gen" in cl) or ("classic" in cl and ("div" in cl or "signal" in cl)):
            s = pd.to_numeric(df[c], errors="coerce").fillna(0)
            return int((s > 0).sum())

    # negative macd: neg_macd / negative_macd etc.
    for c in df.columns:
        cl = c.lower()
        if ("neg" in cl and "macd" in cl) or ("negative" in cl and "macd" in cl):
            s = pd.to_numeric(df[c], errors="coerce").fillna(0)
            return int((s > 0).sum())

    return 0


# --- Adapters: robust gegen alte/neue Funktionssignaturen ---
def _call_local_max_min(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
    Ruft Local_Max_Min robust auf:
      - bevorzugt: Local_Max_Min(df, window_1=win, window_2=1)
      - Fallback:  Local_Max_Min(df)  (alte Signatur, arbeitet in-place, gibt evtl. None zurück)
    """
    try:
        res = Local_Max_Min(df, window_1=int(win), window_2=1)
        return res if isinstance(res, pd.DataFrame) else df
    except TypeError:
        # alte Implementierung ohne Parameter
        res = Local_Max_Min(df)
        return res if isinstance(res, pd.DataFrame) else df


def _call_cbulldivg(df: pd.DataFrame, win: int, ct: float, mt: float) -> pd.DataFrame:
    """
    Ruft CBullDivg_analysis robust auf:
      - positional:   CBullDivg_analysis(df, win, ct, mt)
      - keyword-args: CBullDivg_analysis(df, window=..., Candle_Tol=..., MACD_tol=...)
      - Fallback:     CBullDivg_analysis(df, win) (falls ältere Variante)
    """
    try:
        return CBullDivg_analysis(df, int(win), float(ct), float(mt))
    except TypeError:
        try:
            return CBullDivg_analysis(df, window=int(win), Candle_Tol=float(ct), MACD_tol=float(mt))
        except TypeError:
            return CBullDivg_analysis(df, int(win))


class ChartAnalyzer:
    def __init__(self, config: Optional[dict] = None, divergence_params: Optional[dict] = None, enable_plot: bool = True, diagnose: bool = False):
        self.config = config or {
            "divergence": {
                "window": DivergenceParams.window,
                "candle_tolerance": DivergenceParams.candle_tolerance,
                "macd_tolerance": DivergenceParams.macd_tolerance,
            },
            "visualization": {"background": "#FFFFFF", "crosshair_color": "#eefa"},
        }
        if divergence_params:
            self.config["divergence"].update(divergence_params)
        self.enable_plot = enable_plot
        self.diagnose = diagnose
        self.df: Optional[pd.DataFrame] = None

    # ----- Daten laden -----
    def load_data(self, file_path: str) -> bool:
        try:
            p = Path(file_path)
            if not p.exists():
                logger.error(f"Datei nicht gefunden: {p}")
                return False

            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p, low_memory=False)
            elif p.suffix.lower() == ".parquet":
                df = None
                for engine in ("pyarrow", "fastparquet", None):
                    try:
                        df = (pd.read_parquet(p, engine=engine) if engine else pd.read_parquet(p))
                        break
                    except Exception:
                        continue
                if df is None:
                    logger.error(f"Parquet konnte nicht gelesen werden: {p}")
                    return False
            else:
                logger.error(f"Nicht unterstütztes Dateiformat: {p.suffix}")
                return False

            cols = {c.lower(): c for c in df.columns}
            need = {"open", "high", "low", "close"}
            if not need.issubset(set(cols)):
                logger.error(f"Fehlende Spalten: {need - set(cols)}")
                return False

            # OHLC auf lowercase, Datum normalisieren
            for k in need:
                if cols[k] != k:
                    df.rename(columns={cols[k]: k}, inplace=True)
            df = _coerce_date(df)
            self.df = df
            logger.info(f"Daten geladen: {len(df)} Zeilen.")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden: {e}")
            return False

    # ----- Indikatoren -----
    def calculate_indicators(self) -> bool:
        try:
            if self.df is None:
                return False
            res = Initialize_RSI_EMA_MACD(self.df)
            if res is None:
                return False
            self.df = res
            return True
        except Exception as e:
            logger.error(f"Fehler bei Indikator-Berechnung: {e}")
            return False

    # ----- Extrema (DOE-window) -----
    def find_extrema(self) -> bool:
        try:
            if self.df is None:
                return False
            win = int(self.config["divergence"]["window"])
            self.df = _call_local_max_min(self.df, win)
            return True
        except Exception as e:
            logger.error(f"Fehler bei Extrema-Suche: {e}")
            return False

    # ----- Divergenzen -----
    def analyze_divergences(self) -> Tuple[bool, int, int]:
        try:
            if self.df is None:
                return False, 0, 0
            cfg = self.config["divergence"]
            win = int(cfg["window"]); ct = float(cfg["candle_tolerance"]); mt = float(cfg["macd_tolerance"])

            # robust aufrufen (positional / keyword / fallback)
            result_df = _call_cbulldivg(self.df, win, ct, mt)

            classic = _robust_count(result_df, ["CBullD_gen"])
            neg     = _robust_count(result_df, ["CBullD_neg_MACD"])

            # Fallback: falls result_df minimal war, versuche self.df
            if classic == 0:
                classic = max(classic, _robust_count(self.df, ["CBullD_gen"]))
            if neg == 0:
                neg = max(neg, _robust_count(self.df, ["CBullD_neg_MACD"]))

            # ggf. reichhaltigeres result_df übernehmen
            if isinstance(result_df, pd.DataFrame) and isinstance(self.df, pd.DataFrame) and result_df.shape[1] > self.df.shape[1]:
                self.df = result_df

            if self.diagnose:
                self._diagnose_dump(classic, neg)

            return True, int(classic), int(neg)
        except Exception as e:
            logger.error(f"Fehler bei Divergenz-Analyse: {e}")
            return False, 0, 0

    def _diagnose_dump(self, classic: int, neg: int) -> None:
        """Kleine Diagnoseausgabe für Spaltenlage und Counts."""
        if not isinstance(self.df, pd.DataFrame):
            logger.info("[diagnose] self.df nicht verfügbar.")
            return
        logger.info(f"[diagnose] Columns (erste 30): {list(self.df.columns)[:30]} ... (total {len(self.df.columns)})")
        for probe in ("CBullD_gen", "CBullD_neg_MACD"):
            for c in self.df.columns:
                if c.lower() == probe.lower():
                    vc = self.df[c].value_counts(dropna=False).head(10).to_dict()
                    logger.info(f"[diagnose] value_counts({c}): {vc}")
        logger.info(f"[diagnose] gezählt -> Classic={classic}, NegMACD={neg}")

    # ----- Chart (unverändert) -----
    def create_chart(self) -> bool:
        try:
            if self.df is None:
                return False
            cfg = self.config["visualization"]
            fplt.background = fplt.odd_plot_background = cfg["background"]
            fplt.cross_hair_color = cfg["crosshair_color"]
            ax1, ax2, ax3 = fplt.create_plot("Technische Chartanalyse", rows=3)
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
            fplt.candlestick_ochl(self.df[["date", "open", "close", "high", "low"]], ax=ax1)
            for span in (20, 50, 100, 200):
                col = f"EMA_{span}"
                if col in self.df.columns:
                    getattr(self.df, col).plot(ax=ax1, legend=f"{span}-EMA")
            if "RSI" in self.df.columns:
                fplt.plot(self.df["RSI"], width=2, ax=ax2, legend="RSI")
            if "macd_histogram" in self.df.columns:
                fplt.volume_ocv(self.df[["date", "open", "close", "macd_histogram"]], ax=ax3, colorfunc=fplt.strength_colorfilter)
            return True
        except Exception as e:
            logger.error(f"Fehler bei Chart-Erstellung: {e}")
            return False

    # ----- Komplettlauf -----
    def run_analysis(self, file_path: str) -> Tuple[bool, Dict[str, int]]:
        if not self.load_data(file_path):   return False, {"classic": 0, "neg_macd": 0}
        if not self.calculate_indicators(): return False, {"classic": 0, "neg_macd": 0}
        if not self.find_extrema():         return False, {"classic": 0, "neg_macd": 0}
        ok, c, n = self.analyze_divergences()
        if not ok:                          return False, {"classic": 0, "neg_macd": 0}
        if self.enable_plot:                self.create_chart()
        return True, {"classic": int(c), "neg_macd": int(n)}


# ======= Öffentliche API =======
def _autodetect_input() -> Optional[str]:
    for base in ("./data", "."):
        b = Path(base)
        if b.is_dir():
            for name in sorted(os.listdir(b)):
                if name.lower().endswith((".csv", ".parquet")):
                    return str(b / name)
    candidates = [
        r"C:\Projekte\crt_250816\data\raw\btc_1week_candlesticks_all.csv",
        "data/sp500_data.csv", "data/test_data.parquet", "test_data.csv",
    ]
    return next((p for p in candidates if Path(p).exists()), None)


def analyze(window: int = DivergenceParams.window,
            candle_tolerance: float = DivergenceParams.candle_tolerance,
            macd_tolerance: float = DivergenceParams.macd_tolerance,
            input_path: Optional[str] = None,
            enable_plot: bool = False,
            diagnose: bool = False) -> Dict[str, Any]:
    """
    DOE-tauglicher programmatischer Aufruf. Gibt dict mit Counts + details-DF.
    """
    input_path = input_path or _autodetect_input()
    if input_path is None:
        raise FileNotFoundError("Keine Eingabedatei gefunden. Bitte --input setzen oder Datei in ./data ablegen.")
    analyzer = ChartAnalyzer(
        divergence_params={"window": int(window), "candle_tolerance": float(candle_tolerance), "macd_tolerance": float(macd_tolerance)},
        enable_plot=enable_plot, diagnose=diagnose,
    )
    ok, stats = analyzer.run_analysis(input_path)
    classic = int(stats.get("classic", 0)); neg = int(stats.get("neg_macd", 0))
    return {"classic_count": classic, "neg_macd_count": neg, "details": analyzer.df}


def run(**kwargs) -> Dict[str, Any]:
    return analyze(**kwargs)


def main(window: int = DivergenceParams.window,
         candle_tolerance: float = DivergenceParams.candle_tolerance,
         macd_tolerance: float = DivergenceParams.macd_tolerance,
         input_path: Optional[str] = None,
         enable_plot: bool = True,
         diagnose: bool = False) -> Dict[str, Any]:
    return analyze(window, candle_tolerance, macd_tolerance, input_path, enable_plot, diagnose)


# ======= CLI =======
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mainframe RT DOE – Analyse & Chart")
    p.add_argument("--input", "-i", dest="input_path", type=str, default=None)
    p.add_argument("--window", "-w", type=int, default=DivergenceParams.window)
    p.add_argument("--ct", "--candle-tolerance", dest="candle_tolerance", type=float, default=DivergenceParams.candle_tolerance)
    p.add_argument("--mt", "--macd-tolerance", dest="macd_tolerance", type=float, default=DivergenceParams.macd_tolerance)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--diagnose", action="store_true", help="Zeigt Spalten/Counts-Infos in den Logs")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    res = main(window=args.window,
               candle_tolerance=args.candle_tolerance,
               macd_tolerance=args.macd_tolerance,
               input_path=args.input_path,
               enable_plot=(not args.no_plot),
               diagnose=args.diagnose)
    print("\n=== Zusammenfassung ===")
    print(f"Classic Divergences:       {res.get('classic_count', 0)}")
    print(f"Negative MACD Divergences: {res.get('neg_macd_count', 0)}")
    if not args.no_plot:
        try:
            fplt.show()
        except Exception as e:
            logger.error(f"fplt.show() Fehler: {e}")
