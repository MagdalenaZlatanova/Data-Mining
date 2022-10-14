####
#log_train.csv
####

'''
1. trainをsplitする8:2
train: new_train.csv
validation: validation.csv -> label: val_label
2. all_dataはtimeを変換して
shuffle_train.csv
new_test
'''
import time
import datetime
import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from collections import Counter
train = pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\log_train.csv')
test=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\test\\log_train.csv')
label=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\truth_train.csv',header=None)
label.columns = ['enrollment_id','drop']
print(train['time'][0])
dt_0=datetime.datetime.strptime(train['time'][0], "%Y-%m-%dT%H:%M:%S")
dt_2=datetime.datetime.strptime(train['time'][200], "%Y-%m-%dT%H:%M:%S")
difference=dt_2-dt_0
print(difference)
print(2*difference)
date_time=time.strptime(train['time'][0], "%Y-%m-%dT%H:%M:%S")
f = lambda x: datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S")
train['datetime'] = train['time'].apply(f)
print("done")
print(type(train['datetime'][400]))
print(train['datetime'][400])
train.to_csv("logs.csv", index=False)




