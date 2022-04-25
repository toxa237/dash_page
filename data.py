import pandas as pd
import pypyodbc
import numpy as np

conn = pypyodbc.connect('Driver={SQL SERVER};'
'Server=UAIEV1SQLM01.emea.media.global.loc;'
'Database=AmplifiDataDashboards;'
'Trusted_Connection=yes;')

qwery = """SELECT is2.[source], pp.predicted_sentiment 
FROM philipmorris_predictions pp 
INNER JOIN philipmorris_mention_details pmd 
ON pp.id_mention = pmd.id_mention 
LEFT JOIN id_sources is2 
ON pmd.[source] = is2.id_source_type """

df = pd.read_sql(qwery, conn)
conn.close()
df.head(100000).to_excel('q.xlsx')




