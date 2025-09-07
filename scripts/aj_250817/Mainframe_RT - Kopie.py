# from Initialize_RSI_EMA_MACD_Orig import Initialize_RSI_EMA_MACD
import finplot as fplt
import pandas as pd
from CBullDivg_Analysis_vectorized import CBullDivg_analysis
from Initialize_RSI_EMA_MACD import Initialize_RSI_EMA_MACD
from Local_Maximas_Minimas import Local_Max_Min

# csv_file_path = r'C:\Anirudh\Python\IBKR\Incremental\sp500_first_10_years.csv'
csv_file_path = r"d:\Projekte\crt_250816\data\sp500_first_10_years.csv"

# Read the CSV File
df = pd.read_csv(csv_file_path, low_memory=False)

Initialize_RSI_EMA_MACD(df)
Local_Max_Min(df)
CBullDivg_analysis(df, 5, 0.1, 3.25)

fplt.background = fplt.odd_plot_background = "#242320"  # Adjust Plot Background colour
fplt.cross_hair_color = "#eefa"  # Adjust Crosshair colour

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

for i in range(2, len(df)):
    if df["CBullD_gen"][i] == 1:
        fplt.plot(
            pd.to_datetime(df["CBullD_Lower_Low_date_gen"][i]),
            df["CBullD_Lower_Low_gen"][i],
            style="x",
            ax=ax1,
            color="red",
        )
        fplt.plot(
            pd.to_datetime(df["CBullD_Higher_Low_date_gen"][i]),
            df["CBullD_Higher_Low_gen"][i],
            style="x",
            ax=ax1,
            color="blue",
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

    if df["CBullD_neg_MACD"][i] == 1:
        fplt.plot(
            pd.to_datetime(df["CBullD_Lower_Low_date_neg_MACD"][i]),
            df["CBullD_Lower_Low_neg_MACD"][i],
            style="x",
            ax=ax1,
            color="red",
        )
        fplt.plot(
            pd.to_datetime(df["CBullD_Higher_Low_date_neg_MACD"][i]),
            df["CBullD_Higher_Low_neg_MACD"][i],
            style="x",
            ax=ax1,
            color="blue",
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

fplt.show()
