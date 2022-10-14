import pandas as pd
import numpy as np
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
print(df.head())
groups=[]
datas=[]
for g, data in df.groupby('drop'):
    print(g)
    groups.append(g)
    print(data.head())
    datas.append(data)

print("new")
print(datas[0].head())
#labels=['day_'+str(i) for i in range(0,30)]
minutes=['discussion_'+str(i) for i in range(0,30)]
min_nodrops=[statistics.mean(datas[0][day]) for day in minutes]
min_drops=[statistics.mean(datas[1][day]) for day in minutes]

weeks=['week_'+str(i) for i in range(1,5)]
week_drop=[statistics.mean(datas[1][column]) for column in weeks]
week_nodrop=[statistics.mean(datas[0][column]) for column in weeks]
labels=[column for column in list(df.columns) if column!='Unnamed: 0']
print(labels)
nodrops=[statistics.mean(datas[0][column]) for column in labels]
drops=[statistics.mean(datas[1][column]) for column in labels]
difference={labels[i]:((nodrops[i]-drops[i])/drops[i]) for i in range(0,len(drops))}
print(len(difference))
print(difference)
sorted={k: v for k, v in sorted(difference.items(), key=lambda item: item[1])}
print(sorted)
print(minutes)
print(min_nodrops)
print(min_drops)
fig, ax = plt.subplots()
ax.bar([i for i in range(0,30)], min_nodrops, label='no_drops', color='green')
ax.bar([i for i in range(0,30)],min_drops,  label='drops',color='red' )


ax.set_ylabel('discussion count')
ax.set_title('Average discussion count per day')
ax.legend()

plt.show()