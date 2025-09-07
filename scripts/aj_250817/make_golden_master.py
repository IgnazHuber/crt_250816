# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import sys
import pandas as pd

# Projektpfade
ROOT = Path(__file__).resolve().parents[2]
AJ = ROOT / "scripts" / "aj"
if str(AJ) not in sys.path:
    sys.path.insert(0, str(AJ))

from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def extract_events(res: pd.DataFrame) -> pd.DataFrame:
    """
    Liefert stets ein DataFrame mit Spalten ['type','ts'].
    Typen: gen_LL, gen_HL, neg_LL, neg_HL
    """
    records = []

    def add(typ: str, col_date: str, col_flag: str):
        n = len(res)
        ts = pd.to_datetime(res.get(col_date, pd.Series([pd.NaT] * n)),
                            utc=True, errors="coerce")
        flag = res.get(col_flag, pd.Series([0] * n)).astype(float) == 1.0
        sel = ts[flag & ts.notna()]
        for t in sel:
            records.append({"type": typ, "ts": t})

    # klassische (RSI) Divergenz
    add("gen_LL",  "CBullD_Lower_Low_date_gen",        "CBullD_gen")
    add("gen_HL",  "CBullD_Higher_Low_date_gen",       "CBullD_gen")
    # MACD-basierte Divergenz (negativ)
    add("neg_LL",  "CBullD_Lower_Low_date_neg_MACD",   "CBullD_neg_MACD")
    add("neg_HL",  "CBullD_Higher_Low_date_neg_MACD",  "CBullD_neg_MACD")

    # Immer Spalten sicherstellen, auch wenn records leer sind
    if not records:
        return pd.DataFrame({"type": pd.Series(dtype="object"),
                             "ts":   pd.Series(dtype="datetime64[ns, UTC]")})
    return pd.DataFrame.from_records(records)


def main():
    ap = argparse.ArgumentParser("Make Golden Master")
    ap.add_argument("--input", required=True, help="CSV/Parquet")
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=3.25)
    ap.add_argument("--outdir", type=str, default="tests/golden")
    args = ap.parse_args()

    p = Path(args.input)
    df = load_any(p)

    # Falls 'date' fehlt → aus 'timestamp' übernehmen (CBullDivg erwartet date ODER timestamp)
    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.copy()
        df["date"] = df["timestamp"]

    # Indikatoren & LM vorbereiten (wie im Mainframe)
    df = Initialize_RSI_EMA_MACD(df)
    Local_Max_Min(df)

    # Analyse
    res = CBullDivg_analysis(df.copy(), args.lookback, args.alpha, args.gamma)

    # Events extrahieren (immer Spalten vorhanden)
    ev = extract_events(res)

    # Schreiben
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{p.stem}_lb{args.lookback}_a{args.alpha}_g{args.gamma}.csv"
    ev.to_csv(out, index=False)

    # Logging robust, auch bei leerem ev
    n_total = len(ev)
    n_gen = int((ev["type"] == "gen_LL").sum() + (ev["type"] == "gen_HL").sum()) if not ev.empty else 0
    n_neg = int((ev["type"] == "neg_LL").sum() + (ev["type"] == "neg_HL").sum()) if not ev.empty else 0
    print(f"Golden Master geschrieben: {out} (total={n_total}, gen={n_gen}, neg={n_neg})")


if __name__ == "__main__":
    main()
