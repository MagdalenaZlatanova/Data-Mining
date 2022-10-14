import pandas as pd
num_instances_per_class=10000

label=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\truth_train.csv',header=None)
train = pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\log_train.csv')
label.columns = ['enrollment_id','drop']
drops=0
no_drops=0
keeps=[]
for i in range (0, label.shape[0]):
    if(drops==num_instances_per_class and no_drops==num_instances_per_class):
        break
    if(label['drop'][i]==1):
        if(drops<num_instances_per_class):
            keeps.append(label['enrollment_id'][i])
            drops=drops+1
    else:
        if(no_drops<num_instances_per_class):
            keeps.append(label['enrollment_id'][i])
            no_drops=no_drops+1

new_df=train[train.enrollment_id.isin(keeps)]
new_df.to_csv("balanced.csv", index=False)