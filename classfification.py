import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
label=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\truth_train.csv',header=None, names=['enrollment_id','drop'], index_col='enrollment_id')
import statistics
class LogisticRegressionWithThreshold(LogisticRegression):
    def predict(self, X, threshold=None):
        if threshold == None: # If no threshold passed in, simply call the base class predict, effectively threshold=0.5
            return LogisticRegression.predict(self, X)
        else:
            y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
            y_pred_with_threshold = (y_scores >= threshold).astype(int)

            return y_pred_with_threshold
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
print(selected_df.head())

selected_df['access_days']=np.zeros(20000,int)

for index in selected_df.index:
    for day in range(0,30):
        if(df['min_'+str(day)][index]!=0):
            selected_df['access_days'][index]=selected_df['access_days'][index]+1

print(selected_df['access_days'])

from sklearn.neighbors import KernelDensity
print()
cormat=selected_df.corr()
print(round(cormat,2))
print()

df[['min_18','min_20','min_19','min_29','min_28']].boxplot()
plt.show()
# instantiate and fit the KDE model
sns.kdeplot(data=selected_df, x="num_sessions")
sns.kdeplot(data=selected_df,x='problem_26')
plt.show()
X_train,X_test,y_train,y_test=train_test_split(df.drop("drop",inplace=False,axis=1), df['drop'], test_size=0.2)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
logmodel=LogisticRegression(max_iter=10000)
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
print(classification_report(y_true=y_test,y_pred=predictions))
print()
print(confusion_matrix(y_true=y_test, y_pred=predictions))
print("Accuracy:")
print(accuracy_score(y_true=y_test,y_pred=predictions))
print()
print("Roc_auc_score:")
print(roc_auc_score(y_true=y_test,y_score=logmodel.predict_proba(X_test)[:,1]))


y_scores = cross_val_predict(logmodel, X_train, y_train, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

plt.plot(recalls)
plt.plot(precisions)
plt.legend(['Recall','Precision'])
plt.show()

print(logmodel.decision_function(X_test))
print()


lrt = LogisticRegressionWithThreshold()
lrt.fit(X_train, y_train)

threshold=0.4
y_pred = lrt.predict(X_test, threshold)
print("threshold: "+str(threshold))
print()
print(classification_report(y_true=y_test,y_pred=y_pred))
print()
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print()
print("roc auc score")
print(roc_auc_score(y_test,y_pred))
y_scores = logmodel.predict_proba(X_test)[:,1]
print(len(y_scores))
print(len(y_test))
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()