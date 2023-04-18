import pandas as pd
from datetime import datetime
import numpy as np
data=pd.read_csv("../dataset/dataset.csv")

data.drop(['PatientId',"AppointmentID","Neighbourhood","AppointmentDay","ScheduledDay","SMS_received"],axis=1,inplace=True)

def applyFunc(row):
    r=1
    if(row["Gender"]=="M"):
        r=0
    row["Gender"]=r
    r=0
    if(row['No-show']=="Yes"):
        r=1
    row['No-show']=r
    if(row['Handcap']>=2):
        row['Handcap']=1
    return row

data=data.apply(applyFunc,axis=1)
data.drop_duplicates(keep="first",inplace=True)
data.dropna(axis=1,inplace=True)
data=data[~(data['Age']==-1)]

data['Age']/=100

data.to_csv("../dataset/final_dataset.csv",index=False)