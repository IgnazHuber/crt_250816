import numpy as np
import pandas as pd
import ta

def Candlestick_Type(ohlc: pd.DataFrame):

    CL = np.float64(ohlc["low"])   # read candlestick low price
    CH = np.float64(ohlc["high"])  # read candlestick high
    CO = np.float64(ohlc["open"])  # read candlestick open
    CC = np.float64(ohlc["close"]) # read candlestick candle
    Volume = np.float64(ohlc["volume"]) # read Volume

    Bearish_Engulfing = [0] * len(CH)  # Initialize with zeros
    Shooting_Star = [0] * len(CH)  # Initialize with zeros
    Inverted_Hammer = [0] * len(CH)  # Initialize with zeros
    Bearish_Harami = [0] * len(CH)  # Initialize pattern array
    Evening_Star = [0] * len(CH)  # Initialize with zeros
    Bearish_Piercing = [0] * len(CH)  # Initialize with zeros
    Three_Black_Crows = [0] * len(CH)  # Initialize with zeros
    Tweezer_Top = [0] * len(CH)  # Initialize with zeros

    Bullish_Engulfing = [0] * len(CH)  # Initialize with zeros
    Hammer = [0] * len(CH)  # Initialize with zeros
    Bullish_Harami = [0] * len(CH)  # Initialize with zeros
    Morning_Star = [0] * len(CH)  # Initialize with zeros
    Bullish_Piercing = [0] * len(CH)  # Initialize with zeros
    Three_White_Soldiers = [0] * len(CH)  # Initialize with zeros
    Tweezer_Bottom = [0] * len(CH)  # Initialize with zeros

    # ---------------------------------------------Bullish Patterns----------------------------

    # Recognizing Bullish Engulfing
    for i in range(1, len(CO)):
        # Calculate upper and lower wick sizes
        body_1 = abs(CO[i - 1] - CC[i - 1])  # First candle body size
        upper_wick_1 = CH[i - 1] - max(CO[i - 1], CC[i - 1])  # First candle upper wick
        lower_wick_1 = min(CO[i - 1], CC[i - 1]) - CL[i - 1]  # First candle lower wick

        body_2 = abs(CO[i] - CC[i])  # Second candle body size
        upper_wick_2 = CH[i] - max(CO[i], CC[i])  # Second candle upper wick
        lower_wick_2 = min(CO[i], CC[i]) - CL[i]  # Second candle lower wick

        if CC[i - 1] < CO[i - 1] and CO[i] <= CC[i - 1] and CC[i] > CH[
            i - 1] and lower_wick_2 > upper_wick_2 and lower_wick_1 > upper_wick_1:
            Bullish_Engulfing[i] = 1
        else:
            Bullish_Engulfing[i] = 0
    ohlc['Bullish_Engulfing'] = Bullish_Engulfing

    # Recognizing Hammer
    for i in range(1, len(CO)):
        if (CH[i] - CL[i]) > 2.0 * (abs(CO[i] - CC[i])) and (CC[i] - CL[i]) / (0.001 + CH[i] - CL[i]) > 0.356 and (CO[i] - CL[i]) / (0.001 + CH[i] - CL[i]) > 0.50:
            Hammer[i] = 1
        else:
            Hammer[i] = 0
    ohlc['Hammer'] = Hammer

    # Recognizing Shooting Star
    for i in range(1, len(CO)):
        if CH[i] - max(CO[i], CC[i]) >= abs(CO[i] - CC[i]) * 2.3 and (abs(CH[i] - CO[i])/ abs((CL[i] - CC[i])+0.00000001)) > 1:
            Shooting_Star[i] = 1
        else:
            Shooting_Star[i] = 0
    ohlc['Shooting_Star'] = Shooting_Star

    # Recognizing Inverted Hammer
    for i in range(1, len(CO)):
        if (((CH[i] - CL[i]) > 2.3 * (CO[i] - CC[i])) and ((CH[i] - CC[i]) / (.001 + CH[i] - CL[i]) > 0.356) and ((CH[i] - CO[i]) / (.001 + CH[i] - CL[i]) > 0.6)):
            Inverted_Hammer[i] = 1
        else:
            Inverted_Hammer[i] = 0
    ohlc['Inverted_Hammer'] = Inverted_Hammer

    # Recognizing Bullish Harami
    for i in range(1, len(CO)):
        # Calculate upper/lower wick and body sizes
        body_1 = abs(CO[i - 1] - CC[i - 1])  # First candle body size
        body_2 = abs(CO[i] - CC[i])  # Second candle body size
        upper_wick_2 = CH[i] - max(CO[i], CC[i])  # Second candle upper wick
        lower_wick_2 = min(CO[i], CC[i]) - CL[i]  # Second candle lower wick

        if (CO[i - 1] > CC[i - 1] and CC[i] > CO[i] and CO[i - 1] > CC[i]
            and CO[i] > CC[i - 1] and CL[i] > CL[i - 1] and CH[i] < CH[i - 1]
            and CH[i] <= CO[i - 1] and CL[i] >= CC[i - 1]
            and body_2 > lower_wick_2 and body_2 > upper_wick_2 and body_1 * 0.65 > body_2
            and ((CO[i - 1] - CC[i]) / (CO[i] - CC[i - 1])) >= 1) and CL[i - 1] < CL[i - 2] and CC[i - 1] < CC[i - 2]:
            Bullish_Harami[i] = 1
        else:
            Bullish_Harami[i] = 0
    ohlc['Bullish_Harami'] = Bullish_Harami

    # Recognizing Morning Star
    for i in range(1, len(CO)):
        if (CH[i - 1] - CL[i - 1]) > 0:
            if (CO[i - 2] > CC[i - 2]  # First candle is bearish
                    and abs(CO[i - 2] - CC[i - 2]) / (CH[i - 2] - CL[i - 2]) > 0.603  # First candle has a strong body
                    and abs(CO[i - 1] - CC[i - 1]) / (
                            CH[i - 1] - CL[i - 1]) < 0.53  # Second candle has a small body (indecision)
                    and CO[i - 1] != CC[i - 1]  # Second candle is not a Doji (ensuring some body)
                    and CO[i] < CC[i] and CO[i] >= CL[i - 1]  # Third candle is bullish
                    and (CH[i - 1] - CL[i - 1]) < (CH[i] - CL[i]) and (CH[i - 1] - CL[i - 1]) < (
                            CH[i - 2] - CL[i - 2])  # length of mid candle is lesser than the other two
                    and abs(CO[i] - CC[i]) / (CH[i] - CL[i]) > 0.6  # Third candle has a strong body
                    and CC[i] > (CO[i - 2] + CC[i - 2]) / 2  # Third candle closes above 50% of first candle's body
                    and CL[i - 1] < CL[i - 2]
                    and (ohlc['Shooting_Star'][i] != 1 or ohlc['Shooting_Star'][i - 1] != 1 or ohlc['Shooting_Star'][
                        i - 2] != 1)
                    and (abs(CO[i - 2] - CC[i - 2]) > abs(CO[i - 1] - CC[i - 1]))
                    and (ohlc['Inverted_Hammer'][i] != 1 or ohlc['Inverted_Hammer'][i - 1] != 1 or
                         ohlc['Inverted_Hammer'][i - 2] != 1)
                    and CH[i] > CH[i - 1]):
                Morning_Star[i] = 1
        else:
            Morning_Star[i] = 0
    ohlc['Morning_Star'] = Morning_Star

    # Recognizing Bullish Piercing
    for i in range(1, len(CO)):
        if (CC[i - 1] < CO[i - 1] and CO[i] < CL[i - 1] and CO[i - 1] > CC[i] > CC[i - 1] + (
                (CO[i - 1] - CC[i - 1]) / 2)) \
                and CH[i] < CO[i - 1] and CL[i] < CL[i - 1] and CL[i - 1] < CL[i - 2] and CH[i - 1] < CH[i - 2] and CC[
            i - 1] < CC[i - 2] and (CC[i - 2] < CO[i - 2] or ohlc['Shooting_Star'][i - 2] == 1):
            Bullish_Piercing[i] = 1
        else:
            Bullish_Piercing[i] = 0
    ohlc['Bullish_Piercing'] = Bullish_Piercing

    # Recognizing Three White Soldiers
    for i in range(1, len(CO)):
        if (CO[i - 2] < CC[i - 2]  # First candle is bullish
                and CO[i - 1] < CC[i - 1]  # Second candle is bullish
                and CO[i] < CC[i]  # Third candle is bullish
                and CC[i - 2] < CC[i - 1] < CC[i]  # Each candle closes higher than the previous one
                and CO[i - 1] > CC[i - 2] * 0.95  # Second candle opens within first candle's body (not too low)
                and CO[i] > CC[i - 1] * 0.95  # Third candle opens within second candle's body (not too low)
                and abs(CC[i - 2] - CO[i - 2]) / (CH[i - 2] - CL[i - 2]) > 0.6  # First candle strong body
                and abs(CC[i - 1] - CO[i - 1]) / (CH[i - 1] - CL[i - 1]) > 0.6  # Second candle strong body
                and abs(CC[i] - CO[i]) / (CH[i] - CL[i]) > 0.6  # Third candle strong body
                and (CH[i - 2] - CC[i - 2]) / (CH[i - 2] - CL[i - 2]) < 0.2  # First candle closes near high
                and (CH[i - 1] - CC[i - 1]) / (CH[i - 1] - CL[i - 1]) < 0.2  # Second candle closes near high
                and (CH[i] - CC[i]) / (CH[i] - CL[i]) < 0.2  # Third candle closes near high
                and CL[i - 3] < CL[i - 4] and CC[i - 3] < CO[i - 3] and CC[i - 4] < CO[i - 4]):
            Three_White_Soldiers[i] = 1
        else:
            Three_White_Soldiers[i] = 0
    ohlc['Three_White_Soldiers'] = Three_White_Soldiers

    # Recognizing Tweezer Bottom
    for i in range(1, len(CO)):
        # Calculate upper/lower wick and body sizes
        body_1 = abs(CO[i - 1] - CC[i - 1])  # First candle body size
        upper_wick_1 = CH[i - 1] - max(CO[i - 1], CC[i - 1])  # First candle upper wick
        lower_wick_1 = min(CO[i - 1], CC[i - 1]) - CL[i - 1]  # First candle lower wick

        body_2 = abs(CO[i] - CC[i])  # Second candle body size
        upper_wick_2 = CH[i] - max(CO[i], CC[i])  # Second candle upper wick
        lower_wick_2 = min(CO[i], CC[i]) - CL[i]  # Second candle lower wick

        if (CO[i - 1] > CC[i - 1]  # First candle is bearish
                and CO[i] < CC[i]  # Second candle is bullish
                and abs(CL[i - 1] - CL[i]) / CL[i] < 0.002  # Lows are nearly equal (within 0.2% tolerance)
                and CC[i] > CO[i]  # Second candle must close above open
                and CL[i] < CC[i - 1]
                and CC[i - 2] > CC[i - 1]  # Downtrend before pattern (lower closes)
                and CH[i - 2] > CH[i - 1]  # Downtrend before pattern (lower highs)
                and lower_wick_1 >= upper_wick_1  # First candle: Lower wick > upper wick
                and lower_wick_2 >= upper_wick_2  # Second candle: Lower wick > upper wick
                and body_2 >= body_1 * 0.9 and body_1 >= body_2 * 0.5):
            Tweezer_Bottom[i] = 1
        else:
            Tweezer_Bottom[i] = 0
    ohlc['Tweezer_Bottom'] = Tweezer_Bottom

    # ---------------------------------------------Bearish Patterns----------------------------

    # Recognizing Evening Star
    for i in range(1, len(CO)):
        if (CH[i - 1] - CL[i - 1]) > 0:
            if (CO[i - 2] < CC[i - 2]  # First candle is bullish
                    and abs(CO[i - 2] - CC[i - 2]) / (CH[i - 2] - CL[i - 2]) > 0.603  # First candle has a strong body
                    and abs(CO[i - 1] - CC[i - 1]) / (
                            CH[i - 1] - CL[i - 1]) < 0.53  # Second candle has a small body (indecision)
                    and CO[i - 1] != CC[i - 1]  # Second candle is not a Doji (ensuring some body)
                    and CO[i] > CC[i] and CO[i] <= CH[i - 1]  # Third candle is bearish
                    and (CH[i - 1] - CL[i - 1]) < (CH[i] - CL[i]) and (CH[i - 1] - CL[i - 1]) < (
                            CH[i - 2] - CL[i - 2])  # Middle candle smallest
                    and abs(CO[i] - CC[i]) / (CH[i] - CL[i]) > 0.6  # Third candle has a strong body
                    and CC[i] < (CO[i - 2] + CC[i - 2]) / 2  # Third candle closes below 50% of first candle's body
                    and CH[i - 1] > CH[i - 2]
                    and (abs(CO[i - 2] - CC[i - 2]) > abs(CO[i - 1] - CC[i - 1]))
                    and (ohlc['Hammer'][i] != 1 or ohlc['Hammer'][i - 1] != 1 or ohlc['Hammer'][i - 2] != 1)
                    and CL[i] < CL[i - 1]):
                Evening_Star[i] = 1
        else:
            Evening_Star[i] = 0
    ohlc['Evening_Star'] = Evening_Star

    # Recognizing Bearish Engulfing
    for i in range(1, len(CO)):
        if CC[i-1] > CO[i-1] and CO[i] >= CC[i - 1] and CC[i] < CL[i - 1]:
            Bearish_Engulfing[i] = 1
        else:
            Bearish_Engulfing[i] = 0
    ohlc['Bearish_Engulfing'] = Bearish_Engulfing

    # Recognizing Bearish Harami Pattern
    for i in range(1, len(CO)):  # Start from i=1 to avoid index errors
        body_1 = abs(CO[i - 1] - CC[i - 1])  # First candle body
        body_2 = abs(CO[i] - CC[i])  # Second candle body
        upper_wick_2 = CH[i] - max(CO[i], CC[i])  # Second candle upper wick
        lower_wick_2 = min(CO[i], CC[i]) - CL[i]  # Second candle lower wick

        if (CO[i - 1] < CC[i - 1]  # First candle is bullish
                and CC[i] < CO[i]  # Second candle is bearish
                and CO[i] < CC[i - 1]  # Second candle's open is inside the first candle
                and CC[i] > CO[i - 1]  # Second candle's close is inside the first candle
                and body_2 < body_1  # Second candle body is smaller than the first
                and body_2 > max(lower_wick_2, upper_wick_2)  # Second candle is not a Doji
                and CH[i] <= CO[i - 1]  # Second candle's high is within the first candle
                and CL[i] >= CC[i - 1]):  # Second candle's low is within the first candle
            Bearish_Harami[i] = 1

    ohlc['Bearish_Harami'] = Bearish_Harami

    # Recognizing Bearish Piercing (Dark Cloud Cover)
    for i in range(1, len(CO)):
        if (CC[i - 1] > CO[i - 1] and CO[i] > CH[i - 1] and CO[i - 1] < CC[i] < CC[i - 1] - (
                (CC[i - 1] - CO[i - 1]) / 2)) \
                and CL[i] > CO[i - 1] and CH[i] > CH[i - 1] and CH[i - 1] > CH[i - 2] and CL[i - 1] > CL[i - 2] and CC[
            i - 1] > CC[i - 2] and (CC[i - 2] > CO[i - 2] or ohlc['Hammer'][i - 2] == 1):
            Bearish_Piercing[i] = 1
        else:
            Bearish_Piercing[i] = 0
    ohlc['Bearish_Piercing'] = Bearish_Piercing

    # Recognizing Three Black Crows
    for i in range(1, len(CO)):
        if (CO[i - 2] > CC[i - 2]  # First candle is bearish
                and CO[i - 1] > CC[i - 1]  # Second candle is bearish
                and CO[i] > CC[i]  # Third candle is bearish
                and CC[i - 2] > CC[i - 1] > CC[i]  # Each candle closes lower than the previous one
                and CO[i - 1] < CC[i - 2] * 1.05  # Second candle opens within first candle's body (not too high)
                and CO[i] < CC[i - 1] * 1.05  # Third candle opens within second candle's body (not too high)
                and abs(CC[i - 2] - CO[i - 2]) / (CH[i - 2] - CL[i - 2]) > 0.6  # First candle strong body
                and abs(CC[i - 1] - CO[i - 1]) / (CH[i - 1] - CL[i - 1]) > 0.6  # Second candle strong body
                and abs(CC[i] - CO[i]) / (CH[i] - CL[i]) > 0.6  # Third candle strong body
                and (CC[i - 2] - CL[i - 2]) / (CH[i - 2] - CL[i - 2]) < 0.2  # First candle closes near low
                and (CC[i - 1] - CL[i - 1]) / (CH[i - 1] - CL[i - 1]) < 0.2  # Second candle closes near low
                and (CC[i] - CL[i]) / (CH[i] - CL[i]) < 0.2  # Third candle closes near low
                and CL[i - 3] > CL[i - 4] and CC[i - 3] > CO[i - 3] and CC[i - 4] > CO[i - 4]):  # Prior uptrend
            Three_Black_Crows[i] = 1
        else:
            Three_Black_Crows[i] = 0

    ohlc['Three_Black_Crows'] = Three_Black_Crows

    # Recognizing Tweezer Top
    for i in range(1, len(CO)):
        # First candle components
        body_1 = abs(CO[i - 1] - CC[i - 1])
        upper_wick_1 = CH[i - 1] - max(CO[i - 1], CC[i - 1])
        lower_wick_1 = min(CO[i - 1], CC[i - 1]) - CL[i - 1]

        # Second candle components
        body_2 = abs(CO[i] - CC[i])
        upper_wick_2 = CH[i] - max(CO[i], CC[i])
        lower_wick_2 = min(CO[i], CC[i]) - CL[i]

        # Tweezer Top (Bearish Reversal)
        if (CO[i - 1] < CC[i - 1]  # First candle bullish
                and CO[i] > CC[i]  # Second candle bearish
                and abs(CH[i - 1] - CH[i]) / CH[i] < 0.002  # Highs nearly equal
                and CC[i] < CO[i]  # Second candle closes below open
                and upper_wick_1 >= lower_wick_1  # First candle: upper wick > lower wick
                and upper_wick_2 >= lower_wick_2  # Second candle: upper wick > lower wick
                and CC[i - 2] < CC[i - 1]  # Uptrend before pattern
                and CH[i - 2] < CH[i - 1]  # Higher highs before pattern
                and body_1 >= body_2 * 0.9 and body_2 >= body_1 * 0.5):  # Similar body sizes
            Tweezer_Top[i] = 1
        else:
            Tweezer_Top[i] = 0

    # Add pattern to dataframe
    ohlc['Tweezer_Top'] = Tweezer_Top

    return