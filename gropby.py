import pandas as pd
import datetime


def get_day(start_day, end_day):
    result= end_day-start_day
    return str(result.days)
def find_time(enrollment, data, start_time):
    num_sessions=0
    in_session=False
    start_session=datetime.datetime(year=2000,month=4,day=5)
    end_session=datetime.datetime(year=2000,month=4,day=5)
    for i in range(0, data.shape[0]-1):
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
                #print(column)
                #print(time)
            in_session=False

df=pd.read_csv("balanced.csv")
max=0
i=0
f = lambda x: datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S")
df['datetime'] = df['time'].apply(f)
for g, data in df.groupby('enrollment_id',as_index=True):
    if(i>10):
        break
    data = data.reset_index(level=0)
    data.drop(labels="index",axis=1, inplace=True)
    #print(g, data)
    #print(data.shape)
    data=data.reset_index(level=0)
    print(data)
    start=data['datetime'][0]
    #print('enrollment')
    #find_time(enrollment=g,data=data, start_time=start)
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

print(max.days+1)