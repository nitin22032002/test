# since got a better accuracy using KNN finding the optimal value of n_neighbors which increases the accuracy 
import train_test
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x_train=train_test.x_train
y_train=train_test.y_train
x_test=train_test.x_test
y_test=train_test.y_test

Knn_accuracy_arr = []

for k in range(1,31):
    KNNClassifier=KNeighborsClassifier(n_neighbors=k)
    KNNClassifier.fit(x_train,y_train)
    y_pred_KNN = KNNClassifier.predict(x_test)
    KNNAcc = accuracy_score(y_pred_KNN, y_test)
    print(f'for k={k} accuracy = '+'{:.4f}%'.format(KNNAcc*100))
    Knn_accuracy_arr.append(KNNAcc*100)

K=np.argmax(Knn_accuracy_arr)+1

print(f'Best choice of k: {K}')

kNNClassifier=KNeighborsClassifier(n_neighbors=K)
kNNClassifier.fit(x_train,y_train)
y_pred_kNN=kNNClassifier.predict(x_test)
kNNAcc=accuracy_score(y_pred_kNN,y_test)
pickle.dump(kNNClassifier, open('model.pkl', 'wb'))
print(f'When k={K} we get an accuracy of {kNNAcc*100}%')

valk=[]
for i in range(1,31):
    s='For k='+str(i)
    valk.append(s)

# fig=plt.figure(figsize=(14,10))
# plt.barh(valk,Knn_accuracy_arr)
# plt.ylabel('value of k')
# plt.xlabel('accuracy in percentage')
# plt.title('Value of k vs accuracy')
# plt.show()
