import pandas as pd
import pypyodbc
import numpy as np

# conn = pypyodbc.connect('Driver={SQL SERVER};'
# 'Server=UAIEV1SQLM01.emea.media.global.loc;'
# 'Database=AmplifiDataDashboards;'
# 'Trusted_Connection=yes;'
# 'UID=AmplifiDataRobot;'
# 'PWD=Straight6probably#;')

qwery = """SELECT pp.predicted_sentiment, pp.main_tag
FROM philipmorris_predictions pp"""

df = pd.read_excel('assets/board_pade/figure4.xlsx')

# cursor = conn.cursor()
# conn.close()

for i in range(df.shape[0]):
    print(f'''INSERT INTO rep_profile (area, category, value)
    values ('{df.loc[i, 'area']}', N'{df.loc[i, 'category']}', {int(df.loc[i, 'value'])})''')


# conn.commit()






