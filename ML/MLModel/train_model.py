from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pickle

final_data=pd.read_csv("../MLdataset/dataset.csv")
y=final_data['Price'].to_numpy()
X=final_data.drop(['Price'],axis=1).to_numpy()
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=39)
extra_tree_model = KNeighborsRegressor(n_neighbors=20)
extra_tree_model.fit(train_x, train_y)
output=extra_tree_model.predict(test_x)
ans=np.sqrt(mean_squared_error(test_y,output))
print(ans)
pickle.dump(extra_tree_model,open("model.pkl","wb"))