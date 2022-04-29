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

df = pd.read_excel('assets/board_pade/figure1.xlsx')

# cursor = conn.cursor()
# conn.close()

for i in range(df.shape[0]):
    print(f'''INSERT INTO industry (area, negative, neutral, positive, color)
    values ('{df.loc[i, 'area']}',{int(df.loc[i, 'negative'])}, {int(df.loc[i, 'neutral'])}, {int(df.loc[i, 'positive'])}, '{df.loc[i, 'color']}')''')


# conn.commit()






