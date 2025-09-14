import pandas as pd

df1 = pd.read_csv("c://Projekte\Anirudh//ETH//output_4hour_parquet//backtest_results_ETH_4hour_100perc_with_brokerage_Basis_412s.csv")
df2 = pd.read_csv("c://Projekte\Anirudh//ETH//output_4hour_parquet//backtest_results_ETH_4hour_100perc_with_brokerage_v03_pf4_w8_88s.csv")

print(df1.shape == df2.shape)

print((df1.columns == df2.columns).all())

print(df1.equals(df2))   # True/False

vergleich = df1.compare(df2)   # zeigt Zellen, die unterschiedlich sind
print(vergleich)
