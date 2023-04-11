from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

final_data=pd.read_csv("output.csv")
y=final_data['Price'].to_numpy()
X=final_data.drop(['Price'],axis=1)
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.4,random_state=39)
models=[LinearSVR(random_state=45,max_iter=2000),KNeighborsRegressor(n_neighbors=20),LinearRegression()]
models.append(StackingRegressor([("linear",LinearSVR(random_state=45,max_iter=2000)),("knn",KNeighborsRegressor(n_neighbors=30)),("pod",LinearRegression())]))
for item in models:
    item.fit(train_x,train_y)
    print(pow(mean_squared_error(test_y,item.predict(test_x)),0.5))


