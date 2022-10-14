import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
label=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\truth_train.csv',header=None, names=['enrollment_id','drop'], index_col='enrollment_id')
import statistics
df=pd.read_csv("features.csv", index_col=['enrollment_id'])
#indexes=list(df['Unnamed: 0'].unique())
df['drop']=np.zeros(20000,dtype=int)
for i in range (0,30):
    df.drop('time_'+str(i), inplace=True, axis=1)
for element in list(df.index):
    df['drop'][element]=label['drop'][element]
df['week_1']=sum([df['min_'+str(i)] for i in range(0,7)])
df['week_2']=sum([df['min_'+str(i)] for i in range(7,14)])
df['week_3']=sum([df['min_'+str(i)] for i in range(14,21)])
df['week_4']=sum([df['min_'+str(i)] for i in range(21,30)])
selected_df=df[['week_1','week_2','week_3','week_4','num_sessions','min_29','min_28','problem_26','video_29','video_28','discussion_29','discussion_28','drop']]
X_train,X_test,y_train,y_test=train_test_split(df.drop("num_sessions",inplace=False,axis=1), df['num_sessions'], test_size=0.2)

models = []
models.append(('Linear Regression', LinearRegression()))
models.append(('Lasso', Lasso()))
models.append(('Ridge', Ridge()))
models.append(('SVM', SVR()))
models.append(('Bayesian',BayesianRidge()))
models.append(('Decision Tree', DecisionTreeRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('Bagging', BaggingRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = explained_variance_score(y_test, predictions)
    mae = mean_absolute_error(predictions, y_test)
    mse=mean_squared_error(y_test,predictions)
    msg = "%s: %f (%f) %f" % (name, score, mae, mse)
    print(msg)