import pandas as pd
import datetime
import numpy as np

df=pd.read_csv("vreme_samo.csv",dtype=type_dict)
print(df.info())
print(df.head())
df['time_0'][3]=df['time_0'][3]+datetime.timedelta(days=1)
print(df.head())