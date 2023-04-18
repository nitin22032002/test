import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import BernoulliNB
from lazypredict.Supervised import LazyClassifier
import matplotlib.pyplot as plt

data=pd.read_csv("../dataset/final_dataset.csv")
print(data.columns.tolist())

y=data['No-show'].to_numpy()
X=data.drop(['No-show'],axis=1).to_numpy()

train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=45)
model=BernoulliNB()
model.fit(train_x,train_y)
print(model.score(test_x,test_y))
pickle.dump(model,open("model.pkl","wb"))