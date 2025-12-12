import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
import pprint

from helpers.datasources import SQLconnect

def check_surcharge(id, SCD):
    EngDep, cursorED = SQLconnect('EngDep')
    df = pd.read_sql(f'''Select DateTime, Settlement, GroundLevel from MON_DISPLACEMENT_READINGS where 
    PointID='{id}' and Datetime >= '{SCD}' ''', EngDep)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # calculate the difference between rows along the Ground Level (mCD) column
    df['Difference'] = df['GroundLevel'].diff()

    # filter the dataframe where the difference exceeds 3 metres
    df = df[df['Difference'] < -2.5]

    # return the datetime value of the filtered dataframe
    cursorED.close()
    if len(df['DateTime']) > 0:
        return (min(df['DateTime']))
    else:
        return None

def start_monday(d):
    from datetime import datetime, timedelta
    ahead = 7 - d.weekday()
    if ahead != 7:
        return d + timedelta(days=ahead)
    else:
        return d


def prev_monday(d):
    from datetime import datetime, timedelta
    weekday = d.weekday()
    if weekday != 0:
        return d - timedelta(days=weekday)
    else:
        return d


def search(ID, d, Te, n=3):  # for a given date, search for readings within n-days before
    EngDep, cursorED = SQLconnect('EngDep')
    out = {}
    for d in d:
        if Te is not None:
            if d <= Te:
                for i in range(0, n + 1):
                    s_d = d + timedelta(days=i)
                    cursorED.execute(
                        f'''Select Settlement, Datetime from MON_DISPLACEMENT_READINGS where PointID = '{ID}' and DateTime = '{s_d}' and Settlement is not NULL ''')
                    read = cursorED.fetchone()
                    if read is not None:
                        out[read[1]] = -read[0]
                        break
        else:
            for i in range(0, n + 1):
                s_d = d + timedelta(days=i)
                cursorED.execute(
                    f'''Select Settlement, Datetime from MON_DISPLACEMENT_READINGS where PointID = '{ID}' and DateTime = '{s_d}' and Settlement is not NULL ''')
                read = cursorED.fetchone()
                if read is not None:
                    out[read[1]] = -read[0]
                    break
    return out


def Asaoka_data(id, SCD, ASD, max_date=None, asaoka_days=7, period=0, n=4):  # all plates

    EngDep, cursorED = SQLconnect('EngDep')

    Surchcompl = pd.read_sql_query(
        f'''Select PointID, Surcharge_complete_date, Asaoka_Start_Date from SettlementPlates where PointID = '{id}' and Surcharge_complete_date is not null ''',
        EngDep)
    if len(Surchcompl["PointID"]) != 0:
        if len(Surchcompl['Asaoka_Start_Date']) != 0:
            if SCD is None:
                SCD = Surchcompl['Surcharge_complete_date'][0]
            if ASD is None:
                ASD = Surchcompl['Asaoka_Start_Date'][0]

            SCD = datetime.strptime(SCD, "%Y-%m-%d")
            ASD = datetime.strptime(ASD, "%Y-%m-%d")

            Te = check_surcharge(id, SCD)

            if max_date is not None:
                max_date = datetime.strptime(max_date, "%Y-%m-%d")
                if Te is not None:
                    if max_date <= Te:
                        Te = max_date
                    else:
                        Te = Te
            else:
                max_date = datetime.now()

            cursorED.execute('SELECT TOP (1) Datetime, Settlement, GroundLevel FROM MON_DISPLACEMENT_READINGS WHERE PointID = ? AND Datetime <= ? AND Settlement is not NULL order by Datetime DESC', (id, max_date))
            fetch = cursorED.fetchone()
            Datetime, latest_settl, latest_GL = fetch[0], fetch[1], fetch[2]
            latest_settl = latest_settl * (0.001)       # convert to [m] from [mm]

            err = []

            # try:
            dtt = max_date
            startdate = dtt + timedelta(days=period)
            x = []
            y = []
            stdates = []
            prevdates = []
            if startdate > ASD:
                first_monday = start_monday(ASD)
                intervals = (startdate - first_monday).days // asaoka_days
                date = first_monday
                for w in range(0, intervals + 1):
                    prev = date - timedelta(days=asaoka_days)
                    stdates.append(date)
                    prevdates.append(prev)
                    date += timedelta(days=asaoka_days)
                    y = search(id, stdates, Te, n)
                    x = search(id, prevdates, Te, n)
            else:
                return {"PointID": id,
                        "SCD": SCD,
                        "ASD": ASD,
                        "pairs": None,
                        "dates": None,
                        "m": None,
                        "b": None,
                        "SCD_s": None,
                        "R2_score": None,
                        "Asaoka_pred": None,
                        "DOC": None,
                        "Latest_Settlement": latest_settl,
                        "Latest_GL": latest_GL,
                        "Latest_date": Datetime,
                        "max_date": max_date,
                        "Errors": 'Insufficient data-points (SCD > S1)'}

            len_x, len_y = len(x), len(y)
            if -2 <= len_x - len_y <= 2:
                min_len = min(len_x, len_y)
                x = dict(itertools.islice(x.items(), min_len))
                y = dict(itertools.islice(y.items(), min_len))

            x_list = np.array(list(x.values()))
            y_list = np.array(list(y.values()))
            t_list = np.array(list(x.keys()))
            t_minus_list = np.array(list(y.keys()))
            lst = list(map(lambda x, y: (x, y), x_list, y_list))
            date_lst = list(map(lambda p, q: (p, q), t_list, t_minus_list))

            if len(x) >= 2:
                m, b, = np.polyfit(x_list, y_list, 1)
                r_s = round((np.corrcoef(x_list, y_list)[0, 1]) ** 2, 2)
                Asaoka_pred = round((b / (1 - m)) / 1000, 3)
                Asaoka_DOC = (round(abs(latest_settl / Asaoka_pred), 4)) * 100 if 0 < (round(abs(latest_settl / Asaoka_pred), 4)) * 100 <=100 else 100
                if m == 0:
                    err.append(['Settlement curve anomaly detected (Indefinite DOC. Check SCD)'])
                elif Asaoka_DOC >= 0:
                    return {"PointID": id,
                            "SCD": SCD,
                            "ASD": ASD,
                            "pairs": lst,
                            "dates": date_lst,
                            "m": m,
                            "b": b,
                            "SCD_s": None,
                            "R2_score": r_s,
                            "Asaoka_pred": Asaoka_pred,
                            "DOC": Asaoka_DOC,
                            "Latest_Settlement": latest_settl,
                            "Latest_GL": latest_GL,
                            "Latest_date": Datetime,
                            "max_date": Te,
                            "Errors": None}
                else:
                    err.append(['Settlement curve anomaly detected (Negative DOC)'])
            elif abs(len_x - len_y) >= 3:
                err.append([f'Unable to produce regression line (axis series mismatched due to insufficient data)'])
            elif len(x) < 2 and len(x) == len(y):
                err.append(['Insufficient data-points (less than 2 coordinate pairs)'])
            # except TypeError:
            #     err.append(['TypeError: Errors in Settlement records (Data is not numeric)'])
            # except pyodbc.Error:
            #     err.append(['SQL/pyodbc Error: Unable to update database'])
            # except ArithmeticError:
            #     err.append(['ArithmeticError: Computational errors'])
            cursorED.close()
            if err:
                return {"PointID": id,
                        "SCD": SCD,
                        "ASD": ASD,
                        "pairs": None,
                        "dates": None,
                        "m": None,
                        "b": None,
                        "SCD_s": None,
                        "R2_score": None,
                        "Asaoka_pred": None,
                        "DOC": None,
                        "Latest_Settlement": latest_settl,
                        "Latest_GL": latest_GL,
                        "Latest_date": Datetime,
                        "max_date": Te,
                        "Errors": err[0]}
        else:
            return {"PointID": id,
                    "SCD": None,
                    "ASD": None,
                    "pairs": None,
                    "dates": None,
                    "m": None,
                    "b": None,
                    "SCD_s": None,
                    "R2_score": None,
                    "Asaoka_pred": None,
                    "DOC": None,
                    "Latest_Settlement": None,
                    "Latest_GL": None,
                    "Latest_date": None,
                    "max_date": None,
                    "Errors": 'SCD/ASD Date not specified'}
    else:
        return {"PointID": id,
                "SCD": None,
                "ASD": None,
                "pairs": None,
                "m": None,
                "b": None,
                "SCD_s": None,
                "R2_score": None,
                "Asaoka_pred": None,
                "DOC": None,
                "Latest_Settlement": None,
                "Latest_GL": None,
                "Latest_date": None,
                "max_date": None,
                "Errors": 'Unidentified Settlement Plate'}
