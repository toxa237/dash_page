import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor
import joblib


class ONLY_PREDICTION_WITH_TRP:
    def __init__(self, data_pash):
        data = joblib.load(data_pash)
        self.MODEL = data[0]
        self.trp = data[1]
        self.pcs = data[2]
        self.month = data[3]
        self.period = data[4]
        self.end_date = data[5]
        self.time_delta = data[6]
        self.err = data[7]

    def prediction(self, new_trp: np.array):
        trp = self.trp
        pcs = self.pcs
        month = self.month
        end_date = self.end_date
        for i in range(new_trp.shape[0]):
            month = np.append(month, [end_date.month], axis=0)
            m = np.zeros(12)
            m[end_date.month - 1] = 1
            x = np.array([[trp[-self.period:]], [pcs[-self.period:]]])
            x = np.append(x, m)
            x = [np.append(x, new_trp[i] / self.err)]
            # x = self.SCELER.transform(x)
            y = self.MODEL.predict(x)
            pcs = np.append(pcs, y, axis=0)
            trp = np.append(trp, [new_trp[i] / self.err], axis=0)
            end_date += self.time_delta
        return pcs[-new_trp.shape[0]:]

    def __m(self):
        f_m = self.end_date.month
        return f_m

