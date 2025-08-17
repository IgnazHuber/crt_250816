import sys
from pathlib import Path
import pandas as pd

# Projektpfad
ROOT = Path(__file__).resolve().parents[1]
AJ = ROOT / "scripts" / "aj"
if str(AJ) not in sys.path:
    sys.path.insert(0, str(AJ))

from synth_utils import make_synthetic_series
from Local_Maximas_Minimas import Local_Max_Min
from CBullDivg_Analysis_vectorized import CBullDivg_analysis


def _extract_ts(res: pd.DataFrame, typ: str):
    """Zeitstempel der erkannten Events je Typ extrahieren."""
    n = len(res)
    if typ == "gen":
        flag = res.get("CBullD_gen", pd.Series([0] * n)).astype(float) == 1.0
        a = pd.to_datetime(
            res.get("CBullD_Higher_Low_date_gen", pd.Series([pd.NaT] * n)),
            utc=True, errors="coerce"
        )
        b = pd.to_datetime(
            res.get("CBullD_Lower_Low_date_gen", pd.Series([pd.NaT] * n)),
            utc=True, errors="coerce"
        )
    else:
        flag = res.get("CBullD_neg_MACD", pd.Series([0] * n)).astype(float) == 1.0
        a = pd.to_datetime(
            res.get("CBullD_Higher_Low_date_neg_MACD", pd.Series([pd.NaT] * n)),
            utc=True, errors="coerce"
        )
        b = pd.to_datetime(
            res.get("CBullD_Lower_Low_date_neg_MACD", pd.Series([pd.NaT] * n)),
            utc=True, errors="coerce"
        )
    ts = a.where(a.notna(), b)
    return ts[flag].dropna().to_numpy()



def test_synthetic_bruteforce_alignment():
    # künstliche Serie mit garantierten Divergenzen
    df, pivots = make_synthetic_series()

    # WICHTIG: CBullDivg_analysis erwartet 'date' ODER 'timestamp' -> wir liefern 'date'
    df["date"] = df["timestamp"]

    # WICHTIG: LM_* erzeugen (von CBullDivg_analysis vorausgesetzt)
    Local_Max_Min(df)

    # Analyse
    lookback, alpha, gamma = 5, 0.10, 3.25
    res = CBullDivg_analysis(df.copy(), lookback, alpha, gamma)

    # Mindestens die konstruierten Ereignisse erkennen (robuster Check)
    ts_gen = _extract_ts(res, "gen")
    ts_neg = _extract_ts(res, "neg_MACD")
    assert ts_gen.size >= 3
    assert ts_neg.size >= 3
