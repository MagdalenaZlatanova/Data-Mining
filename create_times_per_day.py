import pandas as pd
import numpy as np
import datetime
maxdays=30
delta=datetime.timedelta(days=0)
print(delta)
print(str(delta))
array=[delta for i in range(0, 20000)]
np_array=np.array(array, dtype=np.timedelta64)
#print(array)
df=pd.DataFrame()
for i in range(0,30):
    name='time_'+str(i)
    df[name]=np_array
df['time_0']=np_array
print(df.head())
df['time_0'][3]=df['time_0'][3]+datetime.timedelta(minutes=67)
print("new")

print(df.head())