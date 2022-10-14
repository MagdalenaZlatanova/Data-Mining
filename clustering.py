import pandas as pd
import numpy as np
label=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\truth_train.csv',header=None, names=['enrollment_id','drop'], index_col='enrollment_id')
import statistics
df=pd.read_csv('C:\\Users\\PCI\\Desktop\\mining proekt\\train\\enrollment_train.csv',index_col='enrollment_id')

df['drop']=np.zeros(len(list(df['course_id'])),dtype=int)
print(df.head())
for element in list(df.index):
    df['drop'][element]=label['drop'][element]

print(df.head())
first_ten=list(df['course_id'].unique())[:10]
print(first_ten)

df=df[df['course_id'].isin(first_ten)]

print(df.shape)
print(len(df['username'].unique()))

courses_df=pd.DataFrame(index=df['username'].unique())
for course in first_ten:
    courses_df[course]=np.zeros(len(df['username'].unique()),dtype=int)

print(courses_df.head())
for g, data in df.groupby('course_id',as_index=True):
    data=data.reindex(data['username'])
    #print(g)
    #print(data)
    #print([index for index in data.index])
    for index in data.index:
        if data['drop'][index]==1:
            courses_df[g][index]=1
        else:
            courses_df[g][index]=-1
print(courses_df.head())

courses_df.to_csv("courses.csv")