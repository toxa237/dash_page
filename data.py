import pandas as pd
import pypyodbc
import numpy as np

conn = pypyodbc.connect('Driver={SQL SERVER};'
'Server=UAIEV1SQLM01.emea.media.global.loc;'
'Database=AmplifiDataDashboards;'
'Trusted_Connection=yes;')

qwery = """SELECT pp.predicted_sentiment 
FROM philipmorris_predictions pp """

df = pd.read_sql(qwery, conn)
conn.close()

print(df.value_counts())




