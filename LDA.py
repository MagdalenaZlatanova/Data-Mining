from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('final_features.csv')
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train,X_test,y_train,y_test=train_test_split(df.drop("drop",inplace=False,axis=1), df['drop'], test_size=0.2)
# Add silent=True to avoid printing out updates with each cycle
clf = make_pipeline(StandardScaler(),
                  LinearDiscriminantAnalysis())
clf.fit(X_train, y_train)

#make prediction
y_pred = clf.predict(X_test)
print(classification_report(y_true=y_test,y_pred=y_pred))
print()
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print("Accuracy:")
print(accuracy_score(y_true=y_test,y_pred=y_pred))
print()