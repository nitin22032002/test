import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from backend.settings import BASE_DIR
furninsh_columns=['Gasconnection', 'AC', 'BED', 'TV', 'DiningTable', 'Sofa']

def furnishingRoom(row):
    furnish=furninsh_columns
    semi_furnish=['BED','Gasconnection']
    furnish_status=0
    for item in furnish:
        if(row[item]!=1):
            furnish_status=0
            break
        else:
            furnish_status=2
    for item in semi_furnish:
        if(row[item]!=1):
            furnish_status=0
            break
        else:
            furnish_status=max(furnish_status,1)
    row['furnishing']=furnish_status
    vals=['ATM','SwimmingPool','Gymnasium','LandscapedGardens','LiftAvailable','IndoorGames','SportsFacility','ClubHouse','24X7Security','PowerBackup']
    for item in vals:
        if(row[item]==9):
            row[item]=2
    return row

maxAmt=10**6

areaMax=10**3

df=pd.read_csv(f"{BASE_DIR}/ML/MLdataset/Bangalore.csv")

df=df.apply(furnishingRoom,axis=1)

df.drop(['Location','MaintenanceStaff','Wardrobe','Refrigerator','GolfCourse','Microwave','WashingMachine','JoggingTrack','RainWaterHarvesting','ShoppingMall','Intercom','School','VaastuCompliant','StaffQuarter','Cafeteria','Hospital',"Children'splayarea",'Wifi','Resale','MultipurposeRoom']+furninsh_columns,axis=1,inplace=True)

df['Area']/=areaMax

df['Price']/=maxAmt

df.to_csv(f"{BASE_DIR}/ML/MLModel/output.csv",index=False)

print("Dataset Pre Process")