import pandas as pd
import numpy as np
import datetime

maxdays = 30
def add_sessions():
    array=[0 for i in range(0, 20000)]
    final_dataset['num_sessions']=np.array(array, dtype=int)
def get_day(start_day, end_day):
    result= end_day-start_day
    return str(result.days)
def find_time(enrollment, data, start_time):
    num_sessions=0
    in_session=False
    start_session=datetime.datetime(year=2000,month=4,day=5)
    end_session=datetime.datetime(year=2000,month=4,day=5)
    for i in range(0, data.shape[0]):
        if(not in_session and data['event'][i]!='page_close'):
            start_session=data['datetime'][i]
            in_session=True
        if(in_session and data['event'][i]=='page_close'):
            end_session=data['datetime'][i]
            day=get_day(start_time,start_session)
            column='time_'+day
            #print(enrollment)
            time=end_session-start_session
            if(time<datetime.timedelta(hours=10)):
                num_sessions=num_sessions+1
                final_dataset[column][enrollment]=final_dataset[column][enrollment]+time
                #print(column)
                #print(time)
            in_session=False
    final_dataset['num_sessions'][enrollment]=final_dataset['num_sessions'][enrollment]+num_sessions
def find_activities(enrollment, data, start_time):
    navigations=0
    videos=0
    access=0
    problems=0
    discussions=0
    wikis=0
    for i in range(0, data.shape[0]):
        if(data['event'][i]=='page_close'):
            continue
        day=get_day(start_time,data['datetime'][i])
        column=data['event'][i]
        name=column+'_'+day
        final_dataset[name][enrollment]=final_dataset[name][enrollment]+1

def add_activities():
    count=[0 for i in range(0,20000)]
    activities=df['event'].unique()
    print(activities)
    activities=[element for element in activities if element!='page_close']
    #activities.delete('page_close')
    for activity in activities:
        for i in range(0,30):
            name=activity+"_"+str(i)
            final_dataset[name]=np.array(count,dtype=int)

def add_times():
    delta = datetime.timedelta(days=0)
    array = [delta for i in range(0, 20000)]
    np_array = np.array(array, dtype=np.timedelta64)
    # print(array)
    for i in range(0, 30):
        name = 'time_' + str(i)
        final_dataset[name] = np_array
    #print(final_dataset.head())
    #final_dataset['time_0'][13] = final_dataset['time_0'][13] + datetime.timedelta(minutes=67)
    #print("new")
    #print(final_dataset.head())

df=pd.read_csv("balanced.csv")

print(df['enrollment_id'].unique())
final_dataset=pd.DataFrame(index=df['enrollment_id'].unique())
final_dataset['enrollment_id']=df['enrollment_id'].unique()
add_times()
add_sessions()
add_activities()
print(list(final_dataset.columns))
f = lambda x: datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S")
df['datetime'] = df['time'].apply(f)
i=0
print(final_dataset.head())
for g, data in df.groupby('enrollment_id',as_index=True):
    print(i)
    data = data.reset_index(level=0)
    data.drop(labels="index",axis=1, inplace=True)
    #print(g, data)
    #print(data.shape)
    data=data.reset_index(level=0)
    #print(data)
    start=data['datetime'][0]
    #print('enrollment')
    find_time(enrollment=g,data=data, start_time=start)

    find_activities(enrollment=g, data=data,start_time=start)
    #print(g)
    #print()
    #print(start)
    #print(start+datetime.timedelta(days=1))
    end=data['datetime'][data.shape[0]-1]
    difference=end-start
    if(i==0):
        max=difference
    elif(difference>max):
        max=difference
    #print(difference)
    #print("DONE-------------")
    i=i+1


k = lambda x: x.total_seconds()/60
for i in range(0, 30):
    final_dataset['min_'+str(i)]=final_dataset['time_'+str(i)].apply(k)
print(final_dataset.head())
final_dataset.to_csv("features.csv")