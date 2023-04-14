import os
import pandas as pd
import numpy as np

all_dataset=os.listdir("../MLdataset")

dfs=[]
for i in range(len(all_dataset)):
    df=pd.read_csv(f"../MLdataset/{all_dataset[i]}")
    df['City']=i
    df['City_name']=all_dataset[i].split(".")[0]
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first",inplace=True)
    dfs.append(df)

df=pd.concat(dfs).reset_index().drop(['index'],axis=1)

df.to_csv("../MLdataset/final_dataset.csv",index=False)
