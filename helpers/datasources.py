import pyodbc
import pandas as pd
import requests
from datetime import datetime as dt

def SQLconnect(database_name):
    db_server = '172.16.181.2\geobase'
    db_name = database_name
    user = 'api'
    pw = 'api'
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + db_server + ';DATABASE=' + db_name + ';UID=' + user + ';PWD=' + pw)
    cursor = cnxn.cursor()
    return cnxn, cursor
def S_series(ids: list, max_date=None):
    EngDep, CursorED = SQLconnect('EngDep')
    data = []
    for id in ids:
        if max_date is None:
            CursorED.execute(
                '''SELECT Datetime, Settlement, GroundLevel, Remark from MON_DISPLACEMENT_READINGS where 
                PointID = ? order by DateTime ASC''', (id))
        else:
            CursorED.execute(
                '''SELECT Datetime, Settlement, GroundLevel, Remark from MON_DISPLACEMENT_READINGS where 
                PointID = ? and Datetime <= ? order by DateTime ASC''', (id, max_date))
        data_st = CursorED.fetchall()
        for i in data_st:
            int_lst = []
            dd = i[0]
            int_lst.append(id)
            int_lst.append(dd)
            int_lst.append(i[1])
            int_lst.append(i[2])
            int_lst.append(i[3])
            data.append(int_lst)
    CursorED.close()
    df_S = pd.DataFrame.from_records(data, columns=['id', 'Date', 'Settlement (mm)', 'Ground Level (mCD)', 'Remarks'])
    df_S['Date'] = df_S['Date'].dt.date
    return df_S

def SM_metrics(id: str):
    response = requests.get(f"http://172.16.181.2:8887/sm/{id}")
    if response.status_code == 200:
        data = response.json()
        return data

def SM_overview(ids: list):
    ids = ','.join(str(i) for i in ids)
    response = requests.get(f"http://172.16.181.2:8887/settlement-summary/{ids}")
    if response.status_code == 200:
        data = response.json()
        return data