������������
ѧ�ţ�2016011607
#ɢ��ͼ����ͼ���۲�����֮��Ĺ�ϵ
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr=pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(20,20),marker='o',hist_kwds={'bins':20},
s=60,alpha=.8,cmap=mglearn.cm3)

# ����sklearn.neighbors����k���ں������з���
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)

#����ģ��
test_score=np.mean(y_pred==y_test)
print(y_pred)
print('accuracy is :{:.2f}'.format(test_score)

Out:
> [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
> accuracy is :0.97

