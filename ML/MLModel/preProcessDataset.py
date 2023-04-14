import pandas as pd
import numpy as np

if __name__=="__main__":
    df=pd.read_csv("../MLdataset/final_dataset.csv")

    # print(df.columns.tolist())


    # location=df[['Location',"City_name"]].drop_duplicates(keep="first")

    # location.to_csv("../MLdataset/location.csv",index=False)

    location=df['Location'].unique().tolist()

    location_dict={}
    for  i in range(len(location)):
        location_dict[location[i]]=i

    drop_coumns=["City_name","Wifi","Wardrobe","MaintenanceStaff","ShoppingMall","Intercom","ATM","School","StaffQuarter","Cafeteria","MultipurposeRoom","Hospital","LiftAvailable","VaastuCompliant","Microwave","GolfCourse"]

    df.drop(drop_coumns,inplace=True,axis=1)
    df.replace(9,np.nan,inplace=True)
    df.dropna(inplace=True)

    def handleLocation(row):
        row["Location"]=location_dict[row["Location"]]
        return row
            
    df=df.apply(handleLocation,axis=1)

    print(df.columns.to_list())

    # df.to_csv("../MLdataset/dataset.csv",index=False)