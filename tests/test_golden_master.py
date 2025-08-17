import sys
import os
import glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AJ = ROOT / "scripts" / "aj"
if str(AJ) not in sys.path:
    sys.path.insert(0, str(AJ))

from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min


def _extract_ts_type(res: pd.DataFrame, typ: str) -> set:
    """Menge der Zeitstempel je Event-Typ für bitgenauen Vergleich."""
    n = len(res)
    maps = {
        "gen_LL": ("CBullD_Lower_Low_date_gen", "CBullD_gen"),
        "gen_HL": ("CBullD_Higher_Low_date_gen", "CBullD_gen"),
        "neg_LL": ("CBullD_Lower_Low_date_neg_MACD", "CBullD_neg_MACD"),
        "neg_HL": ("CBullD_Higher_Low_date_neg_MACD", "CBullD_neg_MACD"),
    }
    col_d, col_f = maps[typ]
    ts = pd.to_datetime(res.get(col_d, pd.Series([pd.NaT] * n)), utc=True, errors="coerce")
    flag = res.get(col_f, pd.Series([0] * n)).astype(float) == 1.0
    return set(ts[flag & ts.notna()].astype("datetime64[ns, UTC]").tolist())


def test_golden_master_parity():
    golden_files = glob.glob(str(ROOT / "tests" / "golden" / "*.csv"))
    if not golden_files:
        import pytest
        pytest.skip("Keine Golden-Master-Dateien vorhanden.")

    for gf in golden_files:
        # Leere Datei? (z.B. wenn Generator nichts schrieb)
        if os.path.getsize(gf) == 0:
            import pytest
            pytest.skip(f"Golden-Master leer: {gf} – bitte neu erzeugen.")

        meta = Path(gf).stem.split("_")
        # erwartet: <stem>_lb<lookback>_a<alpha>_g<gamma>.csv
        lookback = int(meta[-3][2:])
        alpha = float(meta[-2][1:])
        gamma = float(meta[-1][1:])
        stem = "_".join(meta[:-3])

        p = ROOT / "data" / "raw" / f"{stem}.parquet"
        if not p.exists():
            import pytest
            pytest.skip(f"Input {p} fehlt.")

        df = pd.read_parquet(p)
        df = Initialize_RSI_EMA_MACD(df)
        Local_Max_Min(df)

        res = CBullDivg_analysis(df.copy(), lookback, alpha, gamma)

        # Golden laden (robust)
        try:
            g = pd.read_csv(gf, parse_dates=["ts"])
        except pd.errors.EmptyDataError:
            import pytest
            pytest.skip(f"Golden-Master ohne Spalten: {gf} – bitte neu erzeugen.")

        for typ in ["gen_LL", "gen_HL", "neg_LL", "neg_HL"]:
            got = _extract_ts_type(res, typ)
            exp = set(g.loc[g["type"] == typ, "ts"].astype("datetime64[ns, UTC]"))
            assert got == exp, f"Mismatch {typ} in {gf}"
