import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('final_features.csv')
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(1.0)


selected_df=df[['week_1','week_2','week_3','week_4','num_sessions','min_29','min_28','problem_26','video_29','video_28','discussion_29','discussion_28','drop']]
X_train,X_test,y_train,y_test=train_test_split(selected_df.drop("drop",inplace=False,axis=1), df['drop'], test_size=0.2)
gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
y_pred=gpc.predict(X_test)
print(classification_report(y_true=y_test,y_pred=y_pred))
print()
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print("Accuracy:")
print(accuracy_score(y_true=y_test,y_pred=y_pred))
print()