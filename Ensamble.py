import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score

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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
X_train,X_test,y_train,y_test=train_test_split(selected_df.drop("drop",inplace=False,axis=1), selected_df['drop'], test_size=0.2)
log_clf=LogisticRegression(max_iter=10000)
tree_clf=DecisionTreeClassifier()
KNN_clf=KNeighborsClassifier()

log_clf.fit(X_train, y_train)
tree_clf.fit(X_train, y_train)
KNN_clf.fit(X_train,y_train)

voting_clf=VotingClassifier(estimators=[('KNN',KNN_clf),('Tree',tree_clf),('LogReg',log_clf)],voting='soft')
voting_clf.fit(X_train,y_train)
preds=voting_clf.predict(X_test)

print(classification_report(y_test, preds))
print()
print("Confusion matrix")
print(confusion_matrix(y_test,preds))
print()
print("Accuracy:")
print(accuracy_score(y_test, preds))
print('roc_auc_score')
print(roc_auc_score(y_true=y_test,y_score=voting_clf.predict_proba(X_test)[:,1]))
