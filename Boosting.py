import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
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

from sklearn.model_selection import train_test_split,cross_val_score

X_train,X_test,y_train,y_test=train_test_split(df.drop("drop",inplace=False,axis=1), df['drop'], test_size=0.2)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
classifier=AdaBoostClassifier(n_estimators=100)
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)

print(classification_report(y_true=y_test,y_pred=predictions))
print()
print(confusion_matrix(y_test,predictions))
print()
print('Accuracy:')
print(accuracy_score(y_test,predictions))
print()
print("Roc_auc curve:")
print(roc_auc_score(y_true=y_test,y_score=classifier.predict_proba(X_test)[:,1]))